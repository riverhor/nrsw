"""Regenerate the added test descriptors from staged publisher images."""

import numpy as np

from .cohort import METADATA_SHA, image_rows, metadata, read, require, write
from ..r62_svk import digest


# Keep the deterministic physical padding and pixel transforms of R5.
SEGMENT_ROWS = 640


def spans(total):
    return range(0, total, SEGMENT_ROWS)


def segment(folder, method, number):
    return folder / method / f"segment_{number:02d}"


def mobile_rows(rows, images, weights, guard):
    from r29_repro import measure
    from r29_repro.native_encoders import NativeMobile

    encoder = NativeMobile(weights, guard)
    try:
        packets, rasters = [], []
        for start in range(0, len(rows), 8):
            arrays, witnesses = measure.measure_mobile_rows(
                rows[start:start + 8], encoder, lambda row: measure.decode(row, images), guard,
            )
            packets.append(arrays)
            rasters.extend(witnesses)
        return {
            name: np.concatenate([packet[name] for packet in packets])
            for name in ("grid512", "original_local512", "original_valid")
        }, rasters, encoder.contract()
    finally:
        encoder.close()


def sd_rows(rows, images, snapshot, guard, parent):
    from r29_repro import measure
    from r29_repro.native_encoders import NativeSD

    encoder = NativeSD(snapshot, guard)
    try:
        features = np.empty((len(rows), 512), np.float32)
        for i, (row, witness) in enumerate(zip(rows, parent["rasters"], strict=True)):
            with measure.decode(row, images) as image:
                require(witness["id"] == row["id"] and witness["size"] == list(image.size)
                        and witness["rgb_sha256"] == measure.sha256(image.tobytes()), "SD/Mobile RGB mismatch")
                features[i] = encoder.one(image)
        contract = encoder.contract()
    finally:
        encoder.close()
    require(encoder.completed == len(rows) and contract["leaf_cleanup"]["close_completed"],
            "Expected one completed SD operation per image with mmap closure")
    return {"sd512": features}, [], contract


def segment_rows(meta_path, number):
    meta = metadata(meta_path)
    starts = spans(len(meta["rows"]))
    require(0 <= number < len(starts), "Segment out of range")
    start = starts[number]
    return image_rows({"rows": meta["rows"][start:start + SEGMENT_ROWS]})


def measure_segment(meta_path, images, out, method, number, weights):
    from r29_repro.native_encoders import Offline, ResourceGuard
    from r29_repro.state import source_pins

    staged = read(images / "STAGING.json")
    require(staged["schema"] == "NRSW_R62_ADDED_SVK_PUBLISHER_IMAGES_V1"
            and staged["metadata_sha256"] == METADATA_SHA and staged["rows"] == 12116,
            "Missing authenticated added image stage")
    rows = segment_rows(meta_path, number)
    output = segment(out, method, number)
    output.mkdir(parents=True, exist_ok=False)
    parent = None
    with Offline() as offline:
        guard = ResourceGuard(900)
        if method == "mobile":
            arrays, rasters, contract = mobile_rows(rows, images, weights, guard)
        else:
            parent = segment(out, "mobile", number) / "COMPLETE.json"
            parent_info = read(parent)
            require(parent_info["ids"] == [row["id"] for row in rows]
                    and parent_info["metadata_sha256"] == METADATA_SHA
                    and parent_info["feature_sha256"] == digest(parent.with_name("features.npz")),
                    "Unexpected added Mobile parent")
            arrays, rasters, contract = sd_rows(rows, images, weights, guard, parent_info)
        require(offline.attempts == 0, "Added extraction must stay offline")
    guard()
    path = output / "features.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write(output / "COMPLETE.json", {
        "schema": "NRSW_R62_ADDED_SVK_FEATURE_SEGMENT_V1",
        "method": method, "segment": number, "ids": [row["id"] for row in rows],
        "metadata_sha256": METADATA_SHA, "sources": source_pins(), "feature_sha256": digest(path),
        "mobile_parent_sha256": digest(parent) if parent else None,
        "rasters": rasters, "contract": contract, "resource": guard.receipt(),
    })
    print(f"Added {method} segment {number}: {len(rows)} images", flush=True)


def load_features(root, meta):
    from r29_repro.state import source_pins

    rows = meta["rows"]
    sources = source_pins()
    arrays = {}
    receipts = {}
    for number, first in enumerate(spans(len(rows))):
        wanted = [str(r["id"]) for r in rows[first:first + SEGMENT_ROWS]]
        expected = {
            "grid512": ((len(wanted), 512), np.float32),
            "sd512": ((len(wanted), 512), np.float32),
            "original_local512": ((len(wanted), 11, 512), np.float32),
            "original_valid": ((len(wanted),), np.bool_),
        }
        parent = None
        for method, names in (
            ("mobile", ("grid512", "original_local512", "original_valid")), ("sd", ("sd512",)),
        ):
            folder = segment(root, method, number)
            seal = folder / "COMPLETE.json"
            receipt = read(seal)
            packet = folder / "features.npz"
            require(receipt["schema"] == "NRSW_R62_ADDED_SVK_FEATURE_SEGMENT_V1"
                    and receipt["method"] == method and receipt["segment"] == number
                    and receipt["ids"] == wanted and receipt["metadata_sha256"] == METADATA_SHA
                    and receipt["sources"] == sources and receipt["feature_sha256"] == digest(packet)
                    and receipt["mobile_parent_sha256"] == parent, "Unexpected added feature segment")
            parent = receipts[f"{method}/{number}"] = digest(seal)
            with np.load(packet, allow_pickle=False) as values:
                require(set(values.files) == set(names), "Unexpected added feature columns")
                for name in names:
                    data = values[name]
                    shape, dtype = expected[name]
                    require(data.dtype == dtype and data.shape == shape and np.isfinite(data).all(),
                            f"Invalid added {name} descriptor")
                    arrays.setdefault(name, []).append(data)
    output = {name: np.concatenate(parts) for name, parts in arrays.items()}
    require(np.array_equal(output["original_valid"], [r["quad_maps"] is not None for r in rows]),
            "Expected every final added availability flag")
    return output, receipts
