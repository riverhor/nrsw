"""Compare the JPEG experiment with the paper and recompute paired intervals."""

import json
from zipfile import ZipFile

import numpy as np
from r29_repro import cli

from .measurements import digest, ensure, read, save
from .refit import COMPARISONS, METHODS, RANKS, apply_cutoffs
from .statistics import accumulate, class_weighted_parts, draw_counts, merge_metrics, metrics, summarize


SUBMISSION_SHA256 = "c5cbbd107a9774fd604eb311a59c7d87184d8e5b68a70569655ea21313f6b9d1"
EVIDENCE = "reviewer_evidence_r52/shared_jpeg/"


def score_file(root, descriptor, expected):
    name = descriptor["file"]
    ensure(name == expected and digest(root / name) == descriptor["sha256"], "Changed CAL/TEST scores")
    return np.load(root / name, allow_pickle=False)


def native_scores(folder, result, calibration, fold, roster, held, labels, groups):
    test_receipt = result["test_packets"][fold]
    cal_receipt = calibration["packets"][fold]
    test_name, cal_name = f"test{fold}.npz", f"cal{fold}.npz"
    ensure(test_receipt["path"] == test_name and digest(folder / test_name) == test_receipt["sha256"]
           and cal_receipt["path"] == cal_name and digest(folder / cal_name) == cal_receipt["sha256"],
           "Changed original-encoding CAL/TEST scores")
    with np.load(folder / test_name, allow_pickle=False) as test, np.load(folder / cal_name, allow_pickle=False) as cal:
        ensure(test["ids"].tolist() == held["test_ids"] and cal["ids"].tolist() == held["calibration_ids"]
               and test["contexts"].tolist() == cal["contexts"].tolist() == roster
               and np.array_equal(test["labels"], labels) and np.array_equal(test["groups"], groups)
               and np.array_equal(np.sort(cal["scores"], axis=-1)[..., RANKS], cal["cuts"]),
               "Same original-encoding and JPEG fitted contexts, held roles and cutoff ranks")
        values, cuts = test["scores"][:, :2], cal["cuts"][:, :2]
    ensure(values.shape == (96, 2, len(labels)) and cuts.shape == (96, 2, 3),
           "Expected native spectral and diagonal arms")
    return values, cuts


def run(root, metadata, baseline=None):
    refit = root / "refit"
    result = read(refit / "RESULTS.json")
    ensure(result["status"] == "COMPLETE" and result["methods"] == list(METHODS),
           "Expected the full predefined JPEG refit")
    roles, _ = cli.context(metadata, "f0_d3_o0")
    docs = {record["id"]: record for record in read(metadata / "documents.json")["rows"]}
    named_groups, counts = draw_counts(docs[item]["group"] for fold in roles["outer_folds"].values()
                                      for item in fold["test_ids"])
    member_index = {group: index for index, group in enumerate(named_groups)}
    sums = np.zeros((len(counts), len(METHODS), 9))
    base_sum = np.zeros((len(counts), 2, 9))
    total_mass = np.zeros((2, len(counts)))
    fold_points, base_points, point_masses = [], [], []
    cal_receipts = read(refit / "CALIBRATION_COMPLETE.json")["fold_receipts"]
    if baseline is not None:
        original_result = read(baseline / "RESULTS.json")
        original_cal = baseline / "CALIBRATION_COMPLETE.json"
        ensure(original_result["schema"] == "R62_SVK_RAW_RESULTS_V1"
               and original_result["scope"] == "all_384_contexts"
               and original_result["methods"][:2] == list(METHODS[3:])
               and digest(original_cal) == original_result["calibration"]["sha256"],
               "Expected all fits and calibration thresholds from the original-encoding experiment")
        original_calibration = read(original_cal)
    for fold in range(4):
        f = str(fold)
        held = roles["outer_folds"][f]
        y = np.array([docs[item]["label"] for item in held["test_ids"]])
        g = np.array([docs[item]["group"] for item in held["test_ids"]])
        group, weight, mult, mass = class_weighted_parts(y, g, member_index, counts)
        total_mass += mass
        fold_result = result["folds"][f]
        cal_file = refit / f"fold_{fold}_calibration.json"
        ensure(digest(cal_file) == cal_receipts[fold] == fold_result["calibration"], "Changed calibration seal")
        cal = read(cal_file)
        roster = [item["id"] for item in roles["contexts"] if item["context"][0] == fold]
        ensure(cal["contexts"] == fold_result["contexts"] == roster and cal["ranks"] == (RANKS + 1).tolist()
               and len(cal["scores"]) == len(fold_result["test"]) == len(roster) == 96, "Changed folded score roster")
        points, original = [], []
        if baseline is not None:
            original_scores, original_cuts = native_scores(baseline, original_result, original_calibration,
                                                          f, roster, held, y, g)
        for index, context in enumerate(roster):
            with score_file(refit, cal["scores"][index], f"{context}_cal.npz") as c, \
                    score_file(refit, fold_result["test"][index], f"{context}_test.npz") as t:
                ensure(c["ids"].tolist() == held["calibration_ids"] and t["ids"].tolist() == held["test_ids"]
                       and np.array_equal(c["cuts"], apply_cutoffs(c["scores"])), "Changed roles or cutoff ranks")
                scores, cuts = t["scores"], c["cuts"]
            point, point_mass = metrics(y, g, scores, cuts)
            points.append(point)
            sums += accumulate(y, group, weight, mult, mass, scores, cuts) / len(roster)
            if baseline is not None:
                scores, cuts = original_scores[index], original_cuts[index]
                original.append(metrics(y, g, scores, cuts)[0])
                base_sum += accumulate(y, group, weight, mult, mass, scores, cuts) / len(roster)
        fold_points.append(np.mean(points, axis=0))
        point_masses.append(point_mass)
        if baseline is not None:
            base_points.append(np.mean(original, axis=0))
        print(f"Bootstrapped JPEG fold {fold + 1}/4", flush=True)

    denominators = total_mass[[1, 1, 1, 0, 0, 0, 1, 1, 1]].T[:, None, :]
    estimates = merge_metrics(fold_points, point_masses)
    draws = sums / denominators
    ensure(np.allclose(estimates, result["point"], rtol=0, atol=2e-14), "Refit points changed")
    contrasts = [
        dict(methods=[METHODS[a], METHODS[b]], imposed=summarize(
            estimates[a, 1:] - estimates[b, 1:], draws[:, a] - draws[:, b]))
        for a, b in COMPARISONS
    ]
    if baseline is not None:
        native = merge_metrics(base_points, point_masses)
        native_gap = native[0, 1:] - native[1, 1:]
        native_draws = base_sum / denominators
        native_dist = native_draws[:, 0] - native_draws[:, 1]
        contrasted = contrasts[-1]
        contrasted["native"] = summarize(native_gap, native_dist)
        contrasted["change"] = summarize(estimates[3, 1:] - estimates[4, 1:] - native_gap,
                                         draws[:, 3] - draws[:, 4] - native_dist)
    path = root / "PAIRED_RESULTS.json"
    save(path, dict(status="COMPLETE", intervals="paired percentile 95%, conditional and pointwise",
                    bootstrap_draws=len(counts), refit_results_hash=digest(refit / "RESULTS.json"),
                    original_results_sha256=digest(baseline / "RESULTS.json") if baseline is not None else None,
                    contrasts=contrasts, metric_order=result["metric_order"][1:]))
    print(path)


def verify(root, submission):
    ensure(digest(submission) == SUBMISSION_SHA256, "The submission archive checksum does not match")
    with ZipFile(submission) as archive:
        expected = json.loads(archive.read(EVIDENCE + "refit_aggregate.json"))
        paired = json.loads(archive.read(EVIDENCE + "paired_results.json"))
    observed = read(root / "refit" / "RESULTS.json")
    ensure(observed["status"] == "COMPLETE" and observed["methods"] == expected["methods"]
           and observed["metric_order"] == expected["metric_order"], "Expected all methods and metrics from the paper")
    delta = np.abs(np.asarray(observed["point"]) - np.asarray(expected["point"]))
    # Use the same numerical tolerance as the saved-score check.
    ensure(delta.max() <= 2e-9, f"Reported common-JPEG result differs: largest absolute error {delta.max()}")
    intervals = root / "PAIRED_RESULTS.json"
    interval_checks = {}
    if intervals.is_file():
        actual = read(intervals)
        ensure(actual["refit_results_hash"] == digest(root / "refit" / "RESULTS.json")
               and actual["metric_order"] == paired["interval_metric_order"]
               and len(actual["contrasts"]) == len(paired["contrasts"]), "Changed paired result scope")
        for ref, obs in zip(paired["contrasts"], actual["contrasts"], strict=True):
            ensure(ref["methods"] == obs["methods"], "Paired comparison does not match the paper")
            for key in ("imposed", "native", "change"):
                if key not in obs:
                    continue
                error = max(np.abs(np.asarray(ref[key][field]) - np.asarray(obs[key][field])).max()
                            for field in ("point", "percentile95"))
                ensure(error <= 2e-9, f"Reported interval differs for {obs['methods']} {key}: {error}")
                interval_checks["/".join(obs["methods"]) + "/" + key] = float(error)
    receipt = dict(status="PASS_R62_PARITY", maximum_point_absolute_error=float(delta.max()),
                   interval_maximum_absolute_errors=interval_checks, result_sha256=digest(root / "refit" / "RESULTS.json"))
    save(root / "VERIFICATION.json", receipt)
    print(receipt)
