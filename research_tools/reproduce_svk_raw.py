"""Refit and evaluate the Slovak ID detectors using verified image features."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits


CONTRASTS = {
    "spectral_weights_given_diagonal_strength": (("DD", 1), ("SD", -1)),
    "spectral_weights_given_spectral_strength": (("DS", 1), ("SS", -1)),
    "spectral_strength_given_diagonal_weights": (("DD", 1), ("DS", -1)),
    "spectral_strength_given_spectral_weights": (("SD", 1), ("SS", -1)),
    "interaction_weight_gain_with_spectral_minus_diagonal_strength": (
        ("DD", -1), ("SD", 1), ("DS", 1), ("SS", -1),
    ),
    "total_SS_vs_DD": (("DD", 1), ("SS", -1)),
    "mean_of_two_weights_effects": (("DD", .5), ("SD", -.5), ("DS", .5), ("SS", -.5)),
    "mean_of_two_strength_effects": (("DD", .5), ("DS", -.5), ("SD", .5), ("SS", -.5)),
}
CROSSED_ALIASES = {"SS": "NRSW_COMMON_UNIT", "DD": "TAG_DIAGONAL", "SD": "SD", "DS": "DS"}
ENDPOINTS = ("roc_FNR", "own_cut_FPR", "own_cut_FNR")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": path.name, "sha256": digest(path)}


def write_arrays(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": path.name, "sha256": digest(path)}


def fit_panel(data, contexts, output):
    from svk_estimators import fit_model, require

    models, roster = {}, {}
    for fold, keys in contexts.items():
        teachers = {}
        fold_models = []
        for i, key in enumerate(keys, start=1):
            training = data.training(key)
            model, frames = fit_model(training, data.offset, teacher_cache=teachers)
            require(not fold_models or np.array_equal(model.delta, fold_models[0].delta),
                    "Archived contexts have the same genuine anchor alignment")
            fold_models.append(model)
            roster[key] = {
                "training_groups": len(model.train_groups),
                "unit": model.unit,
                "delta_sha256": sha256(model.delta.tobytes()).hexdigest(),
                "train_frame_sha256": {
                    arm: sha256(frame.tobytes()).hexdigest() for arm, frame in frames.items()
                },
                "covariance_relative_infinity_residual": {
                    arm: head.stats["covariance_relative_infinity_residual"]
                    for arm, head in model.heads.items()
                },
            }
            if i % 16 == 0 or i == len(keys):
                print(f"FIT fold {fold}: {i}/{len(keys)} contexts", flush=True)
        models[fold] = fold_models
    return models, write(output / "FIT_COMPLETE.json", {
        "schema": "R62_SVK_RAW_FIT_V1",
        "measurement_store_sha256": data.input_sha256,
        "contexts": roster,
    })


def score_fold(batch, models, fold, role):
    from svk_estimators import BLOCK, require

    if any(model.train_groups.intersection(batch.groups) for model in models):
        raise ValueError(f"Fold {fold} {role} has a component used to fit a scored model")
    result = np.empty((len(models), len(models[0].methods), len(batch)), dtype=np.float64)
    for first in range(0, len(batch), BLOCK):
        stop = min(first + BLOCK, len(batch))
        prepared = models[0].prepare(batch.subset(np.arange(first, stop)))
        for i, model in enumerate(models):
            result[i, :, first:stop] = prepared.score(model)
        if stop == len(batch) or stop % 512 == 0:
            print(f"{role.upper()} fold {fold}: {stop}/{len(batch)} images", flush=True)
    require(np.isfinite(result).all(), "Expected only finite scores")
    return result


def calibrate(data, contexts, models, output, fit):
    from svk_estimators import RANKS, require

    limits, packets = {}, {}
    for fold, keys in contexts.items():
        batch = data.batch(keys[0], "calibration")
        require(len(batch) == len(set(batch.groups)) == 500
                and all(data.documents[i]["label"] == 0 for i in batch.ids),
                "Own CAL contains exactly 500 component-distinct genuine images")
        test_groups = {data.documents[i]["group"] for i in data.contexts[keys[0]]["test_ids"]}
        require(not test_groups.intersection(batch.groups), "CAL and TEST component ownership")
        value = score_fold(batch, models[fold], fold, "calibration")
        limits[fold] = np.sort(value, axis=-1)[..., np.array(RANKS) - 1]
        # Each threshold is the specified ascending order statistic, even
        # if more than one calibration score equals it.
        cuts = limits[fold]
        require(np.all((value[..., None] < cuts[..., None, :]).sum(axis=2) < np.array(RANKS))
                and np.all((value[..., None] <= cuts[..., None, :]).sum(axis=2) >= np.array(RANKS)),
                "Expected verified CAL500 rank decisions")
        packets[fold] = write_arrays(output / f"cal{fold}.npz", ids=batch.ids, contexts=keys,
                                     scores=value, cuts=cuts)
    return limits, write(output / "CALIBRATION_COMPLETE.json", {
        "schema": "R62_SVK_RAW_CAL_V1", "fit": fit,
        "ascending_ranks": RANKS, "strict_decision": "score > cut", "packets": packets,
    })


def test(data, contexts, models, limits, output):
    packets, metrics_inputs = {}, {}
    for fold, keys in contexts.items():
        batch = data.batch(keys[0], "test")
        labels = np.array([data.documents[i]["label"] for i in batch.ids], dtype=np.uint8)
        scores = score_fold(batch, models[fold], fold, "test")
        packets[fold] = write_arrays(output / f"test{fold}.npz", ids=batch.ids, contexts=keys,
                                     scores=scores, labels=labels, groups=batch.groups,
                                     local_valid=batch.local_valid)
        metrics_inputs[fold] = (labels, np.asarray(batch.groups), scores, limits[fold])
    return metrics_inputs, packets


def evaluate(inputs, methods, full):
    from jpeg.statistics import accumulate, class_weighted_parts, draw_counts, metrics, merge_metrics
    from svk_estimators import require

    points, masses = [], []
    if full:
        all_groups = np.concatenate([value[1] for value in inputs.values()])
        groups, counts = draw_counts(all_groups)
        member_index = {group: i for i, group in enumerate(groups)}
        replicates = np.zeros((len(counts), len(methods), 9))
        denominator = np.zeros((2, len(counts)))
        require(len(all_groups) == 16210 and set(inputs) == set(range(4)), "Expected complete SVK TEST population")
    for fold, (labels, groups, scores, cuts) in inputs.items():
        fold_points, mass = [], None
        if full:
            require(len(scores) == 96, "Ninety-six fits per outer fold")
            parts = class_weighted_parts(labels, groups, member_index, counts)
            denominator += parts[-1]
        for i, (score, cut) in enumerate(zip(scores, cuts, strict=True), start=1):
            point, mass = metrics(labels, groups, score, cut)
            fold_points.append(point)
            if full:
                replicates += accumulate(labels, *parts, score, cut) / len(scores)
                if i % 32 == 0:
                    print(f"BOOTSTRAP fold {fold}: {i}/{len(scores)} contexts", flush=True)
        points.append(np.stack(fold_points).mean(axis=0))
        masses.append(mass)
    if full:
        require(np.allclose(np.sum(masses, axis=0), [1411, 1183], atol=1e-10, rtol=0),
                "Expected the published genuine and forged TEST component masses")
    else:
        replicates = None
    values = merge_metrics(points, masses)[:, 1:].reshape(len(methods), 3, 3)
    if full:
        expanded_mass = np.repeat(denominator[[1, 0, 1]].T, 3, axis=1)
        replicates /= expanded_mass[:, None, :]
        replicates = replicates.reshape(len(counts), len(methods), 3, 3)
    lookup = {arm: i for i, arm in enumerate(methods)}
    result = {
        "methods": {arm: dict(zip(ENDPOINTS, point.tolist(), strict=True))
                    for arm, point in zip(methods, values, strict=True)},
        "comparisons": {},
        "crossed_effects": {},
        "paired_component_replicates": 5000 if full else 0,
    }

    def summarize(terms, scale):
        point = sum(factor * values[lookup[arm]] for arm, factor in terms)
        draws = None if replicates is None else sum(factor * replicates[:, lookup[arm]] for arm, factor in terms)
        interval = None if draws is None else scale * np.quantile(draws, [.025, .975], axis=0).transpose(1, 2, 0)
        return {name: {
            "point": (scale * value).tolist(),
            "pointwise95": None if interval is None else interval[i].tolist(),
        } for i, (name, value) in enumerate(zip(ENDPOINTS, point, strict=True))}

    spectral = "NRSW_COMMON_UNIT"
    for arm in ("TAG_DIAGONAL", "GENUINE_ENERGY", "CONCAT_NO_EVIDENCE"):
        result["comparisons"][arm] = summarize(((spectral, 1), (arm, -1)), 1)
    for name, terms in CONTRASTS.items():
        result["crossed_effects"][name] = summarize(tuple((CROSSED_ALIASES[arm], c) for arm, c in terms), 100)
    return result


def verify(result, native_path, crossed_path):
    from svk_estimators import require

    def compare(actual, expected, name, tolerance=2e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        require(actual.shape == expected.shape, f"Saved array shape differs for {name}")
        error = float(np.max(abs(actual - expected)))
        differences[name] = error
        require(np.allclose(actual, expected, atol=tolerance, rtol=0),
                f"Saved result differs for {name} (maximum absolute error {error})")

    differences = {}
    if native_path:
        reference = json.loads(native_path.read_text(encoding="utf-8"))
        for arm, expected in reference["methods"].items():
            observed = result["methods"][arm]
            compare([observed[name] for name in ENDPOINTS], [expected["roc_FNR"], *expected["fixed_cut"]], arm)
        for arm, expected in reference["paired"].items():
            observed = result["comparisons"][arm]
            compare([observed[name]["point"] for name in ENDPOINTS],
                    [expected["roc_FNR_difference"], *expected["fixed_cut_difference"]], f"{arm} contrasts")
            compare([observed[name]["pointwise95"] for name in ENDPOINTS],
                    [expected["roc_FNR_pointwise95"], *expected["fixed_cut_pointwise95"]], f"{arm} intervals")
    if crossed_path:
        reference = json.loads(crossed_path.read_text(encoding="utf-8"))
        for effect, observed in result["crossed_effects"].items():
            compare([observed[name]["point"] for name in ENDPOINTS],
                    [reference["effects"][name][effect]["point_pp"] for name in ENDPOINTS], effect)
            compare([observed[name]["pointwise95"] for name in ENDPOINTS],
                    [reference["effects"][name][effect]["percentile_95_pp"] for name in ENDPOINTS],
                    f"{effect} intervals")
    return differences


def verify_witness(output, result, methods):
    """Compare the one-context outputs to released aggregate points and array hashes."""
    from svk_estimators import require

    path = Path(__file__).with_name("svk_native_witness.json")
    reference = json.loads(path.read_text(encoding="utf-8"))
    context = reference["context"]
    require(tuple(reference["method_order"]) == methods, "Expected the archived method order")
    for method in methods:
        for endpoint in ENDPOINTS:
            require(np.allclose(
                result["methods"][method][endpoint], reference["methods"][method][endpoint], atol=2e-9, rtol=0
            ), f"Witness operating point differs for {method} at {endpoint}")

    def checksum(values):
        return sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()

    with (
        np.load(output / "cal0.npz", allow_pickle=False) as cal,
        np.load(output / "test0.npz", allow_pickle=False) as test,
    ):
        require(cal["contexts"].tolist() == test["contexts"].tolist() == [context],
                "Witness uses the first archived SVK context")
        require(cal["scores"].shape[-1] == reference["scope"]["calibration"]
                and test["scores"].shape[-1] == reference["scope"]["test"],
                "Expected the complete witness CAL and TEST rosters")
        actual = {
            "calibration_scores": checksum(cal["scores"][0]),
            "calibration_cuts": checksum(cal["cuts"][0]),
            "test_scores": checksum(test["scores"][0]),
        }
    write(output / "WITNESS_PARITY.json", {
        "status": "PASS_OPERATING_POINTS",
        "context": context,
        "operating_point_tolerance": 2e-9,
        "exact_arrays": {key: value == reference["sha256"][key] for key, value in actual.items()},
        "actual_array_sha256": actual,
        "reference_sha256": digest(path),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r5-runtime", type=Path, required=True, help="Extracted, authenticated R5 runtime")
    p.add_argument("--measurements", type=Path, required=True, help="Verified image features")
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--reference-native", type=Path, help="Saved complete-estimator results for --all")
    p.add_argument("--reference-crossed", type=Path, help="Saved results for the weight and strength comparison, used with --all")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="All 384 fits from the paper")
    mode.add_argument("--context", action="append", help="Repeat to refit explicit context IDs")
    args = p.parse_args()
    if not args.all and (args.reference_native or args.reference_crossed):
        p.error("Use --all to compare with the paper's aggregate results")
    package = (args.r5_runtime / "package").resolve()
    if not (package / "r29_repro" / "measure.py").is_file():
        p.error("--r5-runtime must contain the extracted R5 package/ and metadata/ folders")
    sys.path.insert(0, str(package))

    from svk_estimators import CROSSED_METHODS, NATIVE_METHODS, require
    from svk_measurements import Measurements

    start = monotonic()
    data = Measurements(args.r5_runtime / "metadata", args.measurements)
    keys = [item["id"] for item in data.source["contexts"]] if args.all else args.context
    require(len(keys) == len(set(keys)) and set(keys) <= data.contexts.keys(),
            "Every selected context must occur once in the measurement plan")
    require(not args.all or len(keys) == 384, "Expected all 384 fits from the paper")
    contexts = {}
    for key in keys:
        fold = data.contexts[key]["context"][0]
        contexts.setdefault(fold, []).append(key)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=2, user_api="blas"):
        models, fit = fit_panel(data, contexts, output)
        limits, calibration = calibrate(data, contexts, models, output, fit)
        metrics_inputs, test_packets = test(data, contexts, models, limits, output)
        methods = NATIVE_METHODS + CROSSED_METHODS
        result = evaluate(metrics_inputs, methods, args.all)
    write(output / "RESULTS.json", {
        "schema": "R62_SVK_RAW_RESULTS_V1",
        "scope": "all_384_contexts" if args.all else "specified_contexts",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numerical_sources_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "reproduce_svk_raw.py", "svk_estimators.py", "svk_evidence_alternatives.py",
            "svk_measurements.py", "jpeg/statistics.py",
        )},
        "methods": methods,
        "contexts_by_fold": contexts,
        "calibration": calibration,
        "test_packets": test_packets,
        "elapsed_seconds": monotonic() - start,
        "results": result,
    })
    if args.reference_native or args.reference_crossed:
        differences = verify(result, args.reference_native, args.reference_crossed)
        write(output / "ARCHIVE_PARITY.json", {
            "status": "PASS", "max_absolute_errors": differences,
            "native_reference_sha256": digest(args.reference_native) if args.reference_native else None,
            "crossed_reference_sha256": digest(args.reference_crossed) if args.reference_crossed else None,
        })
    if keys == ["f0_d3_o0"]:
        verify_witness(output, result, methods)
    print(output / "RESULTS.json", flush=True)


if __name__ == "__main__":
    main()
