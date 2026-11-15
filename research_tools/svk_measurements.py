"""Match verified image features to the original SVK data splits."""

import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from r29_repro import cli, kernel, measure, state


METADATA_RECEIPT_SHA256 = "3f2e9461f2a025b1e4e8c5e59c0837d58e319b0d3be3eeddb13438e545c6a772"


def read_chunk(root, descriptor, shapes):
    path = root.joinpath(*measure.safe_member(descriptor["path"]).parts)
    measure.need(path.resolve().is_relative_to(root.resolve()), "Contained feature chunk")
    measure.check(path, descriptor)
    with np.load(path, allow_pickle=False) as packet:
        measure.need(set(packet.files) == set(shapes), "Exact R5 feature packet array inventory")
        arrays = {name: packet[name] for name in packet.files}
    for name, (shape, dtype) in shapes.items():
        array = arrays[name]
        measure.need(array.shape == shape and array.dtype == dtype and np.isfinite(array).all(),
                     f"Invalid R5 feature coordinates in {path.name}: {name}")
    return arrays


class Measurements:
    def __init__(self, metadata, store):
        self.metadata = Path(metadata)
        self.store = Path(store)
        measure.need(cli.sha(self.metadata / "EXPORT_RECEIPT.json") == METADATA_RECEIPT_SHA256,
                     "Expected authenticated R5 split and measurement metadata")
        receipt = cli.read(self.store / "COMPLETE.json")
        measure.need(receipt["schema"] == "R29_SHARED_STORE_V1" and receipt["status"] == "COMPLETE"
                     and receipt["sources"] == state.source_pins(), "Expected a source-bound completed R5 store")
        for key, path in (("plan", "PLAN.json"), ("mobile", "mobile/COMPLETE.json"), ("sd", "sd/COMPLETE.json")):
            measure.need(receipt[key]["path"] == path, "Expected fixed R5 store layout")
            measure.check(self.store / path, receipt[key])
        plan = measure.load_plan(self.store / "PLAN.json")
        measure.need(plan["coverage"] == "complete_requested_roles" and receipt["rows"] == len(plan["rows"])
                     and {"train", "calibration", "test"} <= set(plan["roles"]),
                     "Expected a complete TRAIN/CAL/TEST measurement plan")
        export_files = cli.read(self.metadata / "EXPORT_RECEIPT.json")["files"]
        for name in ("contexts.json.gz", "documents.json", "layout.json", "registered_geometry.jsonl.gz"):
            measure.need(plan["metadata"][name] == export_files[name], "Expected the released measurement metadata")
            measure.check(self.metadata / name, export_files[name])
        self.source, _ = cli.context(self.metadata, plan["contexts"][0])
        source_contexts = {item["id"]: item for item in self.source["contexts"]}
        self.contexts = {}
        for key in plan["contexts"]:
            item = source_contexts[key]
            self.contexts[key] = dict(item, **self.source["outer_folds"][str(item["context"][0])])
        self.documents = {item["id"]: item for item in cli.read(self.metadata / "documents.json")["rows"]}
        measure.need(plan["field_order"] == self.source["field_order"], "Expected archived SVK field order")
        self.layout = kernel.Layout("IDSpace_SVK_R29", tuple(plan["field_order"]), 512, 512,
                                    sha256(json.dumps(receipt["extractor"], sort_keys=True,
                                                      separators=(",", ":")).encode()).hexdigest())
        self.offset = np.load(self.metadata / "R29_OFFSET_FP64.npy", allow_pickle=False)
        self.input_sha256 = cli.sha(self.store / "COMPLETE.json")
        self._check_plan_rows(plan)
        self._read_arrays(plan, receipt)

    def _check_plan_rows(self, plan):
        _, _, wanted, recovered = measure.request(self.metadata, plan["contexts"], plan["roles"])
        rows = {row["id"]: row for row in plan["rows"]}
        measure.need(rows.keys() == wanted
                     and {i for i, row in rows.items() if "recovered" in row["maps"]} == recovered,
                     "Expected the exact published measurement and recovery rosters")
        seen = set()
        with gzip.open(self.metadata / "registered_geometry.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                geometry = json.loads(line)
                identity = geometry["id"]
                if identity in rows:
                    maps = {"original": geometry["maps"]}
                    if identity in recovered:
                        maps["recovered"] = geometry["recovery_maps"]
                    measure.need(identity not in seen and rows[identity]["maps"] == maps,
                                 "Expected the exact archived sampling map")
                    seen.add(identity)
        measure.need(seen == wanted, "Geometry covers every measured image")

    def _read_arrays(self, plan, receipt):
        rows = plan["rows"]
        self.positions = {item["id"]: i for i, item in enumerate(rows)}
        self.original_valid = np.array([item["maps"]["original"] is not None for item in rows])
        recovered_ids = [item["id"] for item in rows if "recovered" in item["maps"]]
        self.recovery_positions = {identity: i for i, identity in enumerate(recovered_ids)}
        n, m, d = len(rows), self.layout.m, self.layout.d
        measure.need(len(recovered_ids) == plan["required_recovered_variants"], "Expected R5 recovery inventory")
        self.sd = np.empty((n, self.layout.dg), dtype=np.float32)
        self.mobile = np.empty((n, d), dtype=np.float32)
        self.local = np.empty((n, m, d), dtype=np.float32)
        self.recovered = np.empty((len(recovered_ids), m, d), dtype=np.float32)

        sources, documents = receipt["sources"], self.documents
        data = {}
        for label, schema in (("mobile", "R29_SHARED_MOBILE_V1"), ("sd", "R29_SHARED_SD_V1")):
            stage = cli.read(self.store / label / "COMPLETE.json")
            measure.need(stage["schema"] == schema and stage["status"] == "COMPLETE"
                         and stage["rows"] == n and stage["plan_sha256"] == receipt["plan"]["sha256"]
                         and stage["sources"] == sources, f"Expected completed R5 {label} stage for this plan")
            data[label] = stage
        measure.need(data["sd"]["mobile_sha256"] == receipt["mobile"]["sha256"], "Expected same R5 SD/Mobile parent")
        first = 0
        for a, b in zip(data["mobile"]["chunks"], data["sd"]["chunks"], strict=True):
            ids = a["ids"]
            stop = first + len(ids)
            measure.need(0 < len(ids) <= 8 and a["start"] == b["start"] == first
                         and ids == b["ids"] == [row["id"] for row in rows[first:stop]]
                         and [w["id"] for w in a["rasters"]] == ids, "Expected paired R5 chunk IDs and raster order")
            for identity, row in zip(ids, rows[first:stop], strict=True):
                doc = documents[identity]
                measure.need(all(doc[key] == row[key] for key in ("member", "sha256", "bytes"))
                             and doc["original_local_valid"] == (row["maps"]["original"] is not None),
                             "Expected immutable image identity and original field availability")
            shape = (len(ids), m, d), np.float32
            validity = (len(ids),), np.bool_
            mobile = read_chunk(self.store / "mobile", a["features"], {
                "grid512": ((len(ids), d), np.float32),
                "original_local512": shape, "recovered_local512": shape,
                "original_valid": validity, "recovered_valid": validity,
            })
            sd = read_chunk(self.store / "sd", b["features"], {"sd512": ((len(ids), self.layout.dg), np.float32)})
            selector = slice(first, stop)
            measure.need(np.array_equal(mobile["original_valid"], self.original_valid[selector])
                         and np.array_equal(mobile["recovered_valid"], [i in self.recovery_positions for i in ids]),
                         "Expected original and recovered feature validity")
            self.sd[selector] = sd["sd512"]
            self.mobile[selector] = mobile["grid512"]
            self.local[selector] = mobile["original_local512"]
            for j, identity in enumerate(ids):
                if identity in self.recovery_positions:
                    self.recovered[self.recovery_positions[identity]] = mobile["recovered_local512"][j]
            first = stop
        measure.need(first == n, "Expected every registered measurement")

    def batch(self, context, role):
        c = self.contexts[context]
        ids = tuple(cli.selected_ids(c, role))
        ix = np.array([self.positions[identity] for identity in ids], dtype=np.int64)
        measure.need(len(set(ids)) == len(ids), "Expected unique role IDs")
        local = self.local[ix]
        valid = self.original_valid[ix]
        positions = {identity: i for i, identity in enumerate(ids)}
        for identity in measure.recovery_ids(c, role):
            j = positions[identity]
            measure.need(not valid[j], "Expected a recovery only for an originally missing bag")
            local[j] = self.recovered[self.recovery_positions[identity]]
            valid[j] = True
        batch = kernel.FeatureBatch(self.layout, ids, tuple(self.documents[i]["group"] for i in ids),
                                    self.sd[ix], local, self.mobile[ix], valid)
        batch.validate()
        return batch

    def training(self, context):
        c = self.contexts[context]
        batch = self.batch(context, "train")
        for row, identity, group in zip(c["train"], batch.ids, batch.groups, strict=True):
            measure.need(row["id"] == identity and row["group"] == group
                         and self.documents[identity]["label"] == (row["role"] == "F"),
                         "Expected aligned TRAIN labels, components and role metadata")
        return cli.train_batch(batch, c)
