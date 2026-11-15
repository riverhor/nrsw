"""Fit, calibrate and evaluate methods on the same re-encoded SVK images."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

from nrsw.core import center_residuals, genuine_normalization, pool_aligned
from r29_repro import cli, kernel
from .. import svk_estimators
from .measurements import SEGMENT_ROWS, checked_plan, digest, ensure, read, save
from .statistics import TARGETS, merge_metrics, metrics

METHODS = ("JOINT_ALIGNED", "JOINT_UNIFORM", "GLOBAL_ONLY", "NRSW_COMMON_UNIT", "TAG_DIAGONAL")
RANKS = np.array([499, 484, 462]) - 1
COMPARISONS = ((0, 1), (0, 2), (3, 4))
BLOCK = 160


class Store:
    def __init__(self, root, metadata):
        plan, _ = checked_plan(root)
        measurement = read(root / "MEASUREMENTS_COMPLETE.json")
        expected_segments = (plan["image_count"] + SEGMENT_ROWS - 1) // SEGMENT_ROWS
        ensure(measurement["bindings"] == digest(root / "BINDINGS.json")
               and measurement["rows"] == plan["image_count"] == 16260
               and measurement["segments"] == expected_segments
               and all(len(measurement["receipts"][name]) == expected_segments for name in ("mobile", "sd")),
               "Re-extracted input seal")
        self.layout = kernel.Layout("IDSpace_SVK_R29", tuple(plan["field_order"]), 512, 512,
                                    measurement["bindings"])
        self.rows = {row["id"]: row for row in read(metadata / "documents.json")["rows"]}
        n = measurement["rows"]
        self.sd = np.empty((n, 512), np.float32)
        self.mobile = np.empty_like(self.sd)
        self.local = np.zeros((n, 11, 512), np.float32)
        self.original = np.zeros(n, bool)
        self.recovered = {}
        cursor = 0
        for segment in range(measurement["segments"]):
            parents = {stage: root / stage / f"{segment:03d}" for stage in ("mobile", "sd")}
            seals = {}
            for stage, folder in parents.items():
                path = folder / "COMPLETE.json"
                ensure(digest(path) == measurement["receipts"][stage][segment], "Exact feature segment seal")
                seals[stage] = read(path)["chunks"]
            for index, (a, b) in enumerate(zip(seals["mobile"], seals["sd"], strict=True)):
                ids = a["ids"]
                sl = slice(cursor, cursor + len(ids))
                planned_rows = plan["rows"][sl]
                ensure(ids == b["ids"] == [row["id"] for row in planned_rows]
                       and ids == [str(i) for i in range(sl.start, sl.stop)]
                       and a["path"] == b["path"] == f"{index:04d}.npz"
                       and a["payload_sha256"] == b["payload_sha256"] and a["rasters"] == b["rasters"],
                       "Same image identities and transformed rasters")
                ma, sb = parents["mobile"] / a["path"], parents["sd"] / b["path"]
                ensure(digest(ma) == a["sha256"] and digest(sb) == b["sha256"], "Exact feature packets")
                with np.load(ma, allow_pickle=False) as m, np.load(sb, allow_pickle=False) as s:
                    raw = {key: m[key] for key in m.files}
                    raw["sd512"] = s["sd512"]
                    count = len(ids)
                    for key in ("sd512", "grid512", "original_local512", "recovered_local512"):
                        shape = (count, 11, 512) if "local" in key else (count, 512)
                        ensure(raw[key].dtype == np.float32 and raw[key].shape == shape
                               and np.isfinite(raw[key]).all(), "Exact finite FP32 feature block")
                    for variant in ("original", "recovered"):
                        actual = raw[variant + "_valid"]
                        expected = np.array([row["maps"].get(variant) is not None for row in planned_rows])
                        ensure(actual.dtype == np.bool_ and np.array_equal(actual, expected),
                               "Exact archived map availability")
                    self.sd[sl] = raw["sd512"]
                    self.mobile[sl] = raw["grid512"]
                    self.local[sl] = raw["original_local512"]
                    self.original[sl] = raw["original_valid"]
                    for i in np.flatnonzero(raw["recovered_valid"]):
                        self.recovered[ids[i]] = raw["recovered_local512"][i].copy()
                cursor += len(ids)
        ensure(cursor == n and len(self.recovered) == plan["required_recovered_variants"]
               and self.original.tolist() == [self.rows[str(i)]["original_local_valid"] for i in range(n)],
               "Complete unchanged original and recoverable measurement coverage")

    def batch(self, c, role):
        ids = tuple(cli.selected_ids(c, role))
        recovery = c["recovered_train_ids"] if role == "train" else c.get(role + "_recovered_ids", [])
        indices = np.array([int(item) for item in ids])
        local = self.local[indices]
        avail = self.original[indices]
        for item in recovery:
            offset = ids.index(item)
            ensure(not avail[offset], "Original availability remains unchanged")
            local[offset] = self.recovered[item]
            avail[offset] = True
        return kernel.FeatureBatch(self.layout, ids, tuple(self.rows[i]["group"] for i in ids),
                                   self.sd[indices], local, self.mobile[indices], avail)


def local_uniform(batch, native_frame, alpha, model):
    """Replace the local pool only, preserving the native row's total weight."""
    weights = np.repeat((alpha.astype(np.float64).sum(1) / model.layout.m).astype(np.float32)[:, None],
                        model.layout.m, axis=1)
    uniform = pool_aligned(batch, weights, model.delta, model.offset)
    frame = native_frame.copy()
    frame[:, model.layout.slices["pooled_local"]] = uniform[:, model.layout.dg:]
    return frame


def native_train_projection(train, joint):
    alpha = np.zeros((len(train.features), joint.layout.m), np.float32)
    strength = np.zeros(len(train.features), np.float64)
    for teacher in joint.teachers:
        held = teacher.served_fold
        indices = (np.flatnonzero(np.asarray(train.roles) == "BANK") if held is None
                   else np.flatnonzero(train.folds == held))
        alpha[indices], _, strength[indices] = teacher.project(train.features.subset(indices))
    return alpha, strength


def common_frame(batch, pair, local, model, mean, unit, alpha, strength):
    d = batch.layout.dg
    fields = local.reshape(len(batch), -1).astype(np.float64)
    fields[~batch.local_valid] = mean
    pooled = pool_aligned(batch, alpha, model.delta, model.offset)
    return np.column_stack((pair[:, :d], fields, pooled, unit * strength, pair[:, d:]))


@dataclass
class Fitted:
    native: kernel.JointModel
    heads: tuple
    field_mean: np.ndarray
    unit: float
    diagonal: svk_estimators.Route
    training_groups: frozenset

    def score(self, batch, prepared):
        ensure(not self.training_groups.intersection(batch.groups) and batch.layout == self.native.layout,
               "CAL/TEST features must use the same layout and disjoint TRAIN groups")
        ensure(batch is prepared.batch and np.array_equal(prepared.means, self.native.teachers[2].evidence.means)
               and np.array_equal(prepared.offset, self.native.offset)
               and np.array_equal(prepared.delta, self.native.delta), "Same held rows and reference alignment")
        alpha, strength = prepared.spectral(self.native)
        d_alpha, d_strength = self.diagonal.project(batch.local_features, batch.local_valid, prepared.centers)
        native = kernel.assemble_joint(batch, self.native.delta, self.native.offset, alpha, strength,
                                       self.native.strength_unit, self.native.local_mean)
        frames = (native, local_uniform(batch, native, alpha, self.native), prepared.pair.astype(np.float64),
                  common_frame(batch, prepared.pair, prepared.local, self.native,
                               self.field_mean, self.unit, alpha, strength),
                  common_frame(batch, prepared.pair, prepared.local, self.native,
                               self.field_mean, self.unit, d_alpha, d_strength))
        result = np.stack([head.score(frame) for head, frame in zip(self.heads, frames, strict=True)])
        ensure(np.isfinite(result).all(), "Finite paired scores")
        return result


@dataclass
class PreparedQuery:
    batch: kernel.FeatureBatch
    pair: np.ndarray
    local: np.ndarray
    measured: np.ndarray
    centers: dict
    means: np.ndarray
    offset: np.ndarray
    delta: np.ndarray

    def spectral(self, model):
        native = model.teachers[2]
        available = self.batch.local_valid
        alpha = np.zeros((len(self.batch), self.batch.layout.m), dtype=np.float32)
        strength = np.zeros(len(self.batch), dtype=np.float64)
        if available.any():
            centered = self.centers[self.means.tobytes()] if native.evidence.active else None
            alpha[available], peak = svk_estimators.project(native.evidence, self.measured, centered)
            strength[available] = native.strength.standardize(peak)
        return alpha, strength


def prepare_query(batch, model):
    means = model.native.teachers[2].evidence.means
    pair, local = kernel.aligned_parts(batch, model.native.delta, model.native.offset)
    measured = batch.local_features[batch.local_valid]
    centers = {means.tobytes(): center_residuals(measured, means)} if len(measured) else {}
    return PreparedQuery(batch, pair, local, measured, centers, means, model.native.offset, model.native.delta)


def fit(train, reference, *, frozen_original_offset=None):
    batch = train.features
    model, frame = kernel.fit_joint(train, reference, frozen_offset=frozen_original_offset)
    alpha, strength = native_train_projection(train, model)
    ensure(np.array_equal(frame, kernel.assemble_joint(batch, model.delta, model.offset, alpha,
                              strength, model.strength_unit, model.local_mean)), "Native OOF feature parity")
    pair, local = kernel.aligned_parts(batch, model.delta, model.offset)
    genuine = train.labels == 0
    mean = local[genuine & batch.local_valid].astype(np.float64).mean(axis=0).ravel()
    ensure(np.array_equal(mean, model.local_mean), "Shared signed-field imputation")
    uniform = np.full((len(batch), batch.layout.m), np.float32(1 / batch.layout.m), dtype=np.float32)
    pooled = pool_aligned(batch, uniform, model.delta, model.offset)
    _, rms = genuine_normalization(pooled[genuine])
    unit = rms / np.sqrt(pooled.shape[1])
    (routes,), ((d_alpha, d_strength),) = svk_estimators.fit_routes(train, {}, arms=("TAG_DIAGONAL",))
    frames = (local_uniform(batch, frame, alpha, model), pair.astype(np.float64),
              common_frame(batch, pair, local, model, mean, unit, alpha, strength),
              common_frame(batch, pair, local, model, mean, unit, d_alpha, d_strength))
    heads = (model.head,) + tuple(kernel.fit_head(x, train.labels) for x in frames)
    return Fitted(model, heads, mean, unit, routes[-1], frozenset(batch.groups))

def query_blocks(model, batch):
    for lo in range(0, len(batch), BLOCK):
        subset = batch.subset(np.arange(lo, min(lo + BLOCK, len(batch))))
        yield prepare_query(subset, model)


def score(model, batch, blocks=None):
    if blocks is None:
        blocks = query_blocks(model, batch)
    return np.concatenate([model.score(block.batch, block) for block in blocks], axis=1)


def save_scores(path, ids, scores, **extra):
    with path.open("xb") as stream:
        np.savez_compressed(stream, ids=ids, scores=scores, **extra)
    return {"file": path.name, "sha256": digest(path)}


def apply_cutoffs(cal):
    ensure(cal.shape == (len(METHODS), 500), "Exact genuine CAL500 scores per arm")
    return np.sort(cal, axis=1)[:, RANKS]


def run(root, output, metadata, qualification=False):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    store = Store(root, metadata)
    source, _ = cli.context(metadata, "f0_d3_o0")
    reference = np.load(metadata / "HISTORICAL_MEAN_FP64.npy", allow_pickle=False)
    fitted = []
    for fold in range(4):
        contexts = [dict(row, **source["outer_folds"][str(fold)]) for row in source["contexts"]
                    if row["context"][0] == fold]
        ensure(len(contexts) == 96, "All original fitted contexts per fold")
        if qualification:
            contexts = contexts[:1]
        cal = store.batch(contexts[0], "calibration")
        test = store.batch(contexts[0], "test")
        ensure(len(cal) == 500 and len(set(cal.groups)) == 500
               and all(store.rows[item]["label"] == 0 for item in cal.ids), "Distinct genuine CAL500")
        ensure(not set(cal.groups).intersection(test.groups), "CAL and TEST groups disjoint")
        trained = []
        for c in contexts:
            ensure(c.get("calibration_recovered_ids", []) == contexts[0].get("calibration_recovered_ids", [])
                   and c.get("test_recovered_ids", []) == contexts[0].get("test_recovered_ids", []),
                   "Same fold-specific query maps in all contexts")
            train = cli.train_batch(store.batch(c, "train"), c)
            ensure(not set(train.features.groups).intersection(cal.groups)
                   and not set(train.features.groups).intersection(test.groups), "Owned TRAIN roles")
            model = fit(train, reference)
            if trained:
                ensure(np.array_equal(model.native.offset, trained[0].native.offset)
                       and np.array_equal(model.native.delta, trained[0].native.delta),
                       "Common transformed N50 alignment")
            trained.append(model)
        cal_seals, cut_rows = [], []
        cal_blocks = tuple(query_blocks(trained[0], cal))
        for c, model in zip(contexts, trained, strict=True):
            ensure(tuple(cli.selected_ids(c, "calibration")) == cal.ids, "Fixed fold calibration order")
            scored = score(model, cal, cal_blocks)
            cuts = apply_cutoffs(scored)
            cut_rows.append(cuts)
            cal_seals.append(save_scores(output / (c["id"] + "_cal.npz"), cal.ids, scored, cuts=cuts))
        calibration = output / f"fold_{fold}_calibration.json"
        save(calibration, dict(contexts=[c["id"] for c in contexts], scores=cal_seals,
                               ranks=(RANKS + 1).tolist(), measurement=digest(root / "MEASUREMENTS_COMPLETE.json")))
        fitted.append((contexts, trained, test, cut_rows, digest(calibration)))
        print(f"Calibrated fold {fold + 1}/4 ({len(trained)} matched models)", flush=True)
    save(output / "CALIBRATION_COMPLETE.json", dict(fold_receipts=[item[-1] for item in fitted]))
    records = {}
    folds = []
    masses = []
    for fold, (contexts, trained, test, cut_rows, calibration_sha) in enumerate(fitted):
        labels = np.asarray([store.rows[item]["label"] for item in test.ids])
        test_seals, stats = [], []
        test_blocks = tuple(query_blocks(trained[0], test))
        for c, model, cuts in zip(contexts, trained, cut_rows, strict=True):
            ensure(tuple(cli.selected_ids(c, "test")) == test.ids, "Fixed fold evaluation order")
            scored = score(model, test, test_blocks)
            test_seals.append(save_scores(output / (c["id"] + "_test.npz"), test.ids, scored))
            p, mass = metrics(labels, test.groups, scored, cuts)
            stats.append(p)
        folds.append(np.mean(stats, axis=0))
        masses.append(mass)
        records[str(fold)] = dict(contexts=[c["id"] for c in contexts], calibration=calibration_sha,
                                  test=test_seals, class_masses=mass.tolist(),
                                  context_points=np.stack(stats).tolist())
        print(f"Verified fold {fold + 1}/4 ({len(trained)} matched models)", flush=True)
    values = merge_metrics(folds, masses)
    estimators = Path(svk_estimators.__file__)
    source_paths = (Path(kernel.__file__), estimators, estimators.with_name("svk_evidence_alternatives.py"),
                    Path(__file__).with_name("statistics.py"))
    save(output / "RESULTS.json", dict(status="NUMERICAL_QUALIFICATION" if qualification else "COMPLETE",
         methods=METHODS, targets=TARGETS.tolist(),
         metric_order=["AUROC", "ROC_FNR_1", "ROC_FNR_5", "ROC_FNR_10", "CAL_FPR_1", "CAL_FPR_5",
                       "CAL_FPR_10", "CAL_FNR_1", "CAL_FNR_5", "CAL_FNR_10"],
         point=values.tolist(), contrasts_first_minus_second=[
             dict(methods=[METHODS[i], METHODS[j]], point=(values[i] - values[j]).tolist()) for i, j in COMPARISONS
         ], folds=records, seconds=round(time.monotonic() - started, 3), programme=digest(__file__),
         source_hashes={path.name: digest(path) for path in source_paths},
         historical_reference_array_hash=hashlib.sha256(reference.tobytes()).hexdigest(),
         measurements=digest(root / "MEASUREMENTS_COMPLETE.json")))
