"""Freeze original SVK models and thresholds before scoring common JPEG."""

import json
from zipfile import ZipFile

import numpy as np
from r29_repro import cli, kernel

from ..svk_measurements import Measurements
from .evaluation import EVIDENCE, SUBMISSION_SHA256
from .measurements import digest, ensure, save
from .refit import METHODS, RANKS, Store, apply_cutoffs, fit, query_blocks, score
from .statistics import accumulate, class_weighted_parts, draw_counts, merge_metrics, metrics


def jpeg_batch(store, context, original):
    encoded = store.batch(context, "test")
    ensure(encoded.ids == original.ids and encoded.groups == original.groups
           and np.array_equal(encoded.local_valid, original.local_valid),
           "Common-JPEG and original measurements must share roles and field recovery")
    # Only encoder inputs differ. The detector retains its original layout,
    # fitted coordinate transforms, evidence settings and classifier heads.
    return kernel.FeatureBatch(original.layout, encoded.ids, encoded.groups, encoded.global_features,
                               encoded.local_features, encoded.global_mobile, encoded.local_valid)


def score_models(models, batch):
    blocks = tuple(query_blocks(models[0], batch))
    return np.stack([score(model, batch, blocks) for model in models])


def flip_mass(labels, component, weights, multiplicities, original, encoded, cuts):
    old = original[:, :, None] > cuts[:, None, :]
    new = encoded[:, :, None] > cuts[:, None, :]
    diffs = np.stack((~old & new, old & ~new), axis=-1)
    result = np.empty((len(multiplicities), len(METHODS), 2, 3, 2))
    for arm in range(len(METHODS)):
        for label in (0, 1):
            select = labels == label
            values = np.zeros((multiplicities.shape[1], 3, 2))
            np.add.at(values, component[select], diffs[arm, select] * weights[select, None, None])
            result[:, arm, label] = (multiplicities @ values.reshape((-1, 6))).reshape((-1, 3, 2))
    return result


def fit_calibration(original, contexts, reference, output):
    prepared = {}
    calibration = {}
    for fold, roster in contexts.items():
        keys = [item["id"] for item in roster]
        models = [fit(original.training(key), reference, frozen_original_offset=original.offset) for key in keys]
        held = original.batch(keys[0], "calibration")
        ensure(len(held) == len(set(held.groups)) == 500
               and all(original.documents[key]["label"] == 0 for key in held.ids),
               "Exact original CAL500 independent genuine components")
        values = score_models(models, held)
        cuts = np.stack([apply_cutoffs(row) for row in values])
        path = output / f"cal{fold}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, ids=held.ids, contexts=keys, scores=values, cuts=cuts)
        calibration[str(fold)] = dict(file=path.name, sha256=digest(path))
        prepared[fold] = models, cuts
        print(f"Original models calibrated in fold {fold + 1}/4", flush=True)
    save(output / "CALIBRATION_COMPLETE.json", dict(ranks=(1 + RANKS).tolist(), packets=calibration,
                                                    native_measurements=original.input_sha256))
    return prepared


def run(root, metadata, native_measurements, submission=None, qualification=False):
    output = root / ("frozen-qualification" if qualification else "frozen")
    output.mkdir(parents=True, exist_ok=False)
    original = Measurements(metadata, native_measurements)
    jpeg = Store(root, metadata)
    source, _ = cli.context(metadata, "f0_d3_o0")
    if qualification:
        contexts = {0: [original.contexts["f0_d3_o0"]]}
    else:
        contexts = {fold: [dict(row, **source["outer_folds"][str(fold)]) for row in source["contexts"]
                            if row["context"][0] == fold] for fold in range(4)}
        ensure(all(len(items) == 96 for items in contexts.values()), "All original contexts in each fold")
    reference = np.load(metadata / "HISTORICAL_MEAN_FP64.npy", allow_pickle=False)
    fitted = fit_calibration(original, contexts, reference, output)

    bootstrap_groups, counts = draw_counts(original.documents[key]["group"]
                                          for fold in source["outer_folds"].values() for key in fold["test_ids"])
    index = {group: i for i, group in enumerate(bootstrap_groups)}
    counts = np.vstack((np.ones((1, len(index)), dtype=np.int32), counts))
    total_mass = np.zeros((2, len(counts)))
    totals = np.zeros((2, len(counts), len(METHODS), 9))
    flips = np.zeros((len(counts), len(METHODS), 2, 3, 2))
    point_by_encoding = ([], [])
    point_masses, packets = [], []
    for fold, roster in contexts.items():
        native = original.batch(roster[0]["id"], "test")
        encoded = jpeg_batch(jpeg, roster[0], native)
        ensure(set(original.batch(roster[0]["id"], "calibration").groups).isdisjoint(native.groups),
               "Original CAL and TEST must be component-disjoint")
        models, cuts = fitted[fold]
        original_scores = score_models(models, native)
        jpeg_scores = score_models(models, encoded)
        path = output / f"fold{fold}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, ids=encoded.ids, contexts=[item["id"] for item in roster],
                                scores=jpeg_scores, original_scores=original_scores, cuts=cuts)
        packets.append(dict(file=path.name, sha256=digest(path)))
        labels = np.array([original.documents[identity]["label"] for identity in native.ids])
        groups = np.asarray(native.groups)
        component, weight, mult, mass = class_weighted_parts(labels, groups, index, counts)
        total_mass += mass
        points = ([], [])
        for a, b, cut in zip(original_scores, jpeg_scores, cuts, strict=True):
            for encoding, rows in enumerate((a, b)):
                point, point_mass = metrics(labels, groups, rows, cut)
                points[encoding].append(point)
                totals[encoding] += accumulate(labels, component, weight, mult, mass, rows, cut) / len(roster)
            flips += flip_mass(labels, component, weight, mult, a, b, cut) / len(roster)
        for combined, estimates in zip(point_by_encoding, points, strict=True):
            combined.append(np.mean(estimates, axis=0))
        point_masses.append(point_mass)
        print(f"Original models scored both encodings in fold {fold + 1}/4", flush=True)

    totals /= total_mass[[1, 1, 1, 0, 0, 0, 1, 1, 1]].T[:, None, :]
    flips /= total_mass.T[:, None, :, None, None]
    changes = totals[1] - totals[0]
    flip_delta = flips[..., 0] - flips[..., 1]
    ensure(np.allclose(flip_delta[:, :, 0], changes[..., 3:6], rtol=0, atol=2e-14)
           and np.allclose(-flip_delta[:, :, 1], changes[..., 6:], rtol=0, atol=2e-14),
           "Paired threshold crossings reproduce both observed risk changes")
    auroc = [merge_metrics(points, point_masses)[:, 0].tolist() for points in point_by_encoding]
    result = dict(status="QUALIFICATION" if qualification else "COMPLETE", methods=METHODS, packets=packets,
                  point_original=totals[0, 0].tolist(), point_common_jpeg=totals[1, 0].tolist(),
                  delta=changes[0].tolist(), delta_interval=np.quantile(changes[1:], [.025, .975], axis=0).tolist(),
                  AUROC=dict(original=auroc[0], encoded=auroc[1]), crossing=flips[0].tolist(),
                  crossing_interval=np.quantile(flips[1:], [.025, .975], axis=0).tolist(),
                  calibration=digest(output / "CALIBRATION_COMPLETE.json"),
                  common_jpeg_measurements=digest(root / "MEASUREMENTS_COMPLETE.json"))
    save(output / "RESULTS.json", result)
    if submission is not None:
        ensure(not qualification, "Choose the full run to compare with the paper")
        verify(output, result, submission)


def verify(output, result, submission):
    ensure(digest(submission) == SUBMISSION_SHA256, "The submission archive checksum does not match")
    with ZipFile(submission) as archive:
        expected = json.loads(archive.read(EVIDENCE + "frozen_aggregate.json"))
    ensure(result["methods"] == tuple(expected["methods"]), "Expected the registered five frozen methods")
    errors = {}
    for key in ("point_original", "point_common_jpeg", "delta", "delta_interval", "crossing", "crossing_interval"):
        max_error = np.abs(np.asarray(result[key]) - np.asarray(expected[key])).max()
        ensure(max_error <= 2e-9, f"Fixed-model sensitivity differs from the paper at {key}: {max_error}")
        errors[key] = float(max_error)
    for encoding in ("original", "encoded"):
        key = f"AUROC/{encoding}"
        error = np.abs(np.asarray(result["AUROC"][encoding]) - np.asarray(expected["AUROC"][encoding])).max()
        ensure(error <= 2e-9, f"Fixed-model sensitivity differs from the paper at {key}: {error}")
        errors[key] = float(error)
    save(output / "VERIFICATION.json", dict(status="PASS_R62_FROZEN_PARITY", maximum_absolute_errors=errors,
                                            result_sha256=digest(output / "RESULTS.json")))
    print(errors)
