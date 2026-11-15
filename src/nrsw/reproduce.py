"""Recompute the R62 CrossVerify headline from archived scores."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np


R62_SHA256 = "c5cbbd107a9774fd604eb311a59c7d87184d8e5b68a70569655ea21313f6b9d1"
TARGETS = (1, 5, 10)
BUDGET = 15
BUDGET_DIR = "crossverify_label_budget"
MODES = ("pooled_71_root_sensitivity", "fixed_fold_primary")
METHODS = (
    ("NRSW", f"{BUDGET_DIR}/sealed_scores/nrsw_v2", f"{BUDGET_DIR}/RESULTS.json", None),
    ("CrossVerify_ResNet18", f"{BUDGET_DIR}/sealed_scores/resnet", f"{BUDGET_DIR}/RESULTS.json",
     "mean_nrsw_minus_resnet_pp"),
    ("DINOv3_frozen_dense", f"{BUDGET_DIR}/sealed_scores/dinov3", f"{BUDGET_DIR}/DINOV3_RESULTS.json",
     "nrsw_minus_dinov3_pp"),
    ("FLiD_text_shared_regions", f"{BUDGET_DIR}/sealed_scores/flid", f"{BUDGET_DIR}/FLID_RESULTS.json",
     "nrsw_minus_flid_pp"),
    ("GLOBAL_ONLY", "global_evidence/budget", "global_evidence/budget/RESULTS.json",
     "nrsw_minus_global_only_pp"),
    ("FROZEN_SHARED_CORE", "global_evidence/shared", "global_evidence/shared/RESULTS.json",
     "nrsw_minus_shared_core_pp"),
)
NATIVE_SVK = "submission_evidence/svk/RESULTS.json"
CROSSED_SVK = "reviewer_evidence_r52/weight_strength/independent_outcome_audit.json"
JPEG_SVK = "reviewer_evidence_r52/shared_jpeg/refit_aggregate.json"
JPEG_SVK_PAIRED = "reviewer_evidence_r52/shared_jpeg/paired_results.json"
RAW_RUNTIME = "submission_evidence/reproduction/NRSW_R29_portable_reproduction_r5.zip"


def _require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


class _Evidence:
    """Read selected R62 members in memory after checking their published hashes."""

    def __init__(self, archive):
        self.archive = archive
        self.manifest = json.loads(archive.read("PACKAGE_MANIFEST_R62.json"))["members"]

    def read(self, name):
        data = self.archive.read(name)
        record = self.manifest[name]
        _require(len(data) == record["bytes"] and sha256(data).hexdigest() == record["sha256"],
                 f"R62 member integrity check failed for {name}")
        return data

    def json(self, name):
        return json.loads(self.read(name))


def _roc_fnr(scores, labels, records, draws):
    """Interpolate the ROC between whole score ties, keeping each record together."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    tie_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    genuine = labels[order] == 0
    result = np.empty((len(draws), len(TARGETS)))
    for start in range(0, len(draws), 256):
        sample = draws[start:start + 256, records[order]]
        fp = np.column_stack((np.zeros(len(sample), dtype=int), np.cumsum(sample * genuine, axis=1)[:, tie_ends]))
        tp = np.column_stack((np.zeros(len(sample), dtype=int), np.cumsum(sample * ~genuine, axis=1)[:, tie_ends]))
        row = np.arange(len(sample))
        for index, target in enumerate(TARGETS):
            # The last point at or below the target includes any vertical ROC steps.
            budget = target * fp[:, -1]
            scaled_fp = 100 * fp
            low = (scaled_fp <= budget[:, None]).sum(axis=1) - 1
            frac = (budget - scaled_fp[row, low]) / (scaled_fp[row, low + 1] - scaled_fp[row, low])
            misses = tp[:, -1] - tp[row, low] - frac * (tp[row, low + 1] - tp[row, low])
            result[start:start + len(sample), index] = 100 * misses / tp[:, -1]
    return result


def _counts(generator, records, replicates):
    return np.vstack((np.ones(records, dtype=int),
                      generator.multinomial(records, np.full(records, 1 / records), replicates)))


def _check_close(recomputed, archived, description):
    archived = np.asarray(archived)
    _require(recomputed.shape == archived.shape and np.allclose(recomputed, archived, rtol=0, atol=2e-9),
             f"Archived R62 result differs for {description}")


def _run_crossverify(evidence):
    design = evidence.json(f"{BUDGET_DIR}/DESIGN.json")
    reports = {name: evidence.json(name) for name in dict.fromkeys(method[2] for method in METHODS)}
    reference = reports[f"{BUDGET_DIR}/RESULTS.json"]
    replicates = reference["bootstrap_replicates"]
    seed = reference["bootstrap_seed"]
    _require(replicates == 5000 and all(report["bootstrap_replicates"] == replicates
             and report["bootstrap_seed"] == seed for report in reports.values()), "R62 bootstrap design mismatch")
    _require(all(report["targets_fpr"] == [target / 100 for target in TARGETS]
                 for report in reports.values()), "R62 ROC target mismatch")

    held_order = design["held_canonical_order"]
    rows_by_id = {row["id"]: row for fold in design["test_folds"].values() for row in fold}
    _require(len(held_order) == len(rows_by_id) == 284 and set(held_order) == set(rows_by_id),
             "R62 held-record roster mismatch")
    labels = np.array([rows_by_id[identity]["label"] for identity in held_order])
    roots = sorted({row["component"] for row in rows_by_id.values()})
    root_ids = {root: index for index, root in enumerate(roots)}
    membership = np.array([root_ids[rows_by_id[identity]["component"]] for identity in held_order])
    _require(len(roots) == 71 and np.isin(labels, (0, 1)).all() and all(
        np.all(np.bincount(membership[labels == label], minlength=len(roots)) == per_root)
        for label, per_root in ((0, 1), (1, 3))), "R62 should have one genuine and three forgeries per record")

    rng = np.random.Generator(np.random.PCG64(seed))
    position = {identity: index for index, identity in enumerate(held_order)}
    pooled_counts = np.zeros((replicates + 1, len(roots)), dtype=int)
    folds = []
    for rows in design["test_folds"].values():
        indices = np.array([position[row["id"]] for row in rows])
        root_indices, fold_membership = np.unique(membership[indices], return_inverse=True)
        _require(not pooled_counts[0, root_indices].any(), "R62 record belongs to multiple held folds")
        counts = _counts(rng, len(root_indices), replicates)
        pooled_counts[:, root_indices] = counts
        folds.append((indices, labels[indices], fold_membership, counts))
    _require(len(folds) == 6 and pooled_counts[0].all(), "R62 held-fold partition mismatch")

    contexts = [row["id"] for row in design["contexts"]]
    selected = [index for index, row in enumerate(design["contexts"]) if row["budget"] == BUDGET]
    _require(len(selected) == 3, "R62 should have three fitted draws at 15 forged labels")
    analyses = {name: {} for name in MODES}
    for method, directory, _, _ in METHODS:
        with np.load(BytesIO(evidence.read(f"{directory}/TEST_SCORES.npz")), allow_pickle=False) as packet:
            scores = packet["scores"]
            _require(packet["context_ids"].tolist() == contexts and packet["ids"].tolist() == held_order
                     and scores.shape == (len(contexts), len(held_order)) and np.isfinite(scores).all(),
                     f"R62 score packet has incorrect rows for {method}")

        pooled = np.stack([_roc_fnr(vector, labels, membership, pooled_counts) for vector in scores[selected]])
        fixed = np.zeros_like(pooled)
        for indices, fold_labels, records, counts in folds:
            weight = counts.shape[1] / len(roots)
            for draw, vector in enumerate(scores[selected]):
                fixed[draw] += weight * _roc_fnr(vector[indices], fold_labels, records, counts)
        analyses[MODES[0]][method] = pooled
        analyses[MODES[1]][method] = fixed

    reproduced = {}
    for name, analysis in analyses.items():
        nrsw = analysis["NRSW"]
        methods = {}
        for method, _, report, difference_key in METHODS:
            archived = reports[report]["achieved_fpr_analyses"][name]["budgets"][str(BUDGET)]
            values = analysis[method]
            mean = values[:, 0].mean(axis=0)
            _check_close(values[:, 0], archived["draw_fnr_pct"][method], f"{name} {method} per-draw FNR")
            _check_close(mean, archived["mean_fnr_pct"][method], f"{name} {method} mean FNR")
            result = {"fnr_percent": mean.tolist()}
            if difference_key is not None:
                differences = (nrsw - values).mean(axis=0)
                interval = np.percentile(differences[1:], (2.5, 97.5), axis=0).T
                _check_close(differences[0], archived[difference_key], f"{name} {method} paired difference")
                _check_close(interval, archived["pointwise_95pct_interval_pp"], f"{name} {method} paired interval")
                result.update(nrsw_minus_method_pp=differences[0].tolist(), pointwise_95_percent_pp=interval.tolist())
            methods[method] = result
        reproduced[name] = methods
    return dict(held_images=284, records=71, forged_training_images_per_fit=BUDGET,
                fitted_draws_per_method=len(selected), paired_bootstrap_replicates=replicates, analyses=reproduced)


def _percent(values):
    return (100 * np.asarray(values)).tolist()


def _read_svk(evidence):
    native = evidence.json(NATIVE_SVK)
    crossed = evidence.json(CROSSED_SVK)
    jpeg = evidence.json(JPEG_SVK)
    jpeg_paired = evidence.json(JPEG_SVK_PAIRED)
    spectral, diagonal = "NRSW_COMMON_UNIT", "TAG_DIAGONAL"
    native_pair = native["paired"][diagonal]
    jpeg_pair = next(row["imposed"] for row in jpeg_paired["contrasts"] if row["methods"] == [spectral, diagonal])
    jpeg_indices = [jpeg["metric_order"].index(f"ROC_FNR_{target}") for target in TARGETS]
    interval_indices = [jpeg_paired["interval_metric_order"].index(f"ROC_FNR_{target}") for target in TARGETS]
    jpeg_rates = {method: _percent([row[index] for index in jpeg_indices])
                  for method, row in zip(jpeg["methods"], jpeg["point"], strict=True)}
    native_rates = {method: _percent(row["roc_FNR"]) for method, row in native["methods"].items()}
    _check_close(np.subtract(native_rates[spectral], native_rates[diagonal]),
                 _percent(native_pair["roc_FNR_difference"]), "SVK native aggregate contrast")
    _check_close(np.subtract(jpeg_rates[spectral], jpeg_rates[diagonal]),
                 _percent([jpeg_pair["point"][index] for index in interval_indices]), "SVK JPEG aggregate contrast")
    return {
        "scope": "Verified archived aggregates. Per-view scores for these three analyses are absent from the submission ZIP.",
        "original_encoding": {
            "fnr_percent": native_rates,
            "spectral_minus_diagonal_pp": _percent(native_pair["roc_FNR_difference"]),
            "pointwise_95_percent_pp": _percent(native_pair["roc_FNR_pointwise95"]),
        },
        "crossed_weights_and_strength_gain_pp": crossed["effects"]["roc_FNR"],
        "common_jpeg_refit": {
            "fnr_percent": jpeg_rates,
            "spectral_minus_diagonal_pp": _percent([jpeg_pair["point"][index] for index in interval_indices]),
            "pointwise_95_percent_pp": _percent([jpeg_pair["percentile95"][index] for index in interval_indices]),
        },
    }


@contextmanager
def _r62_evidence(path):
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
        _require(digest == R62_SHA256, f"Expected the R62 submission ZIP with SHA-256 {R62_SHA256}; got {digest}")
        stream.seek(0)
        with ZipFile(stream) as archive:
            yield _Evidence(archive)


def reproduce(path: Path, svk_scores: Path | None = None) -> dict:
    """Verify the immutable manuscript ZIP and replay the supported headline results."""
    with _r62_evidence(path) as evidence:
        report = dict(revision="R62", archive_sha256=R62_SHA256, roc_fpr_targets_percent=list(TARGETS),
                      crossverify_recomputed_from_scores=_run_crossverify(evidence),
                      svk_archived_aggregates=_read_svk(evidence))
        if svk_scores is not None:
            from .replay_svk import replay

            report["svk_recomputed_from_scores"] = replay(svk_scores, evidence.json(NATIVE_SVK),
                                                          evidence.json(JPEG_SVK), evidence.json(CROSSED_SVK))
        return report


def _display(report):
    def rates(values):
        return " / ".join(f"{value:.2f}" for value in values)

    print("Verified R62 submission ZIP. ROC false-positive-rate targets are 1%, 5%, and 10%.\n")
    cv = report["crossverify_recomputed_from_scores"]
    print("CrossVerify registry card, recomputed from archived test scores")
    print(f"{cv['held_images']} images, {cv['records']} records, {cv['forged_training_images_per_fit']} forged labels per fit, "
          f"{cv['paired_bootstrap_replicates']:,} paired record resamples")
    print(f"{'Method':28} {'Pooled FNR (%)':24} Fold-weighted FNR (%)")
    for method, pooled in cv["analyses"][MODES[0]].items():
        fixed = cv["analyses"][MODES[1]][method]
        print(f"{method:28} {rates(pooled['fnr_percent']):24} {rates(fixed['fnr_percent'])}")
    gaps = [-gap for method in cv["analyses"][MODES[0]].values() for gap in method.get("nrsw_minus_method_pp", ())]
    print(f"NRSW pooled FNR reductions across the five comparators are {min(gaps):.2f} to {max(gaps):.2f} pp.")
    print("Both analyses match the archived rates and every paired pointwise interval.\n")

    svk = report["svk_archived_aggregates"]
    replayed = report.get("svk_recomputed_from_scores")
    print("IDSpace SVK, recomputed from ordinal scores" if replayed else "IDSpace SVK, verified archived aggregates")
    for name, label in (("original_encoding", "Original encoding"), ("common_jpeg_refit", "Common JPEG refit")):
        saved = svk[name]
        if replayed:
            spectral = replayed[name]["NRSW_COMMON_UNIT"]["roc_fnr_percent"]
            diagonal = replayed[name]["TAG_DIAGONAL"]["roc_fnr_percent"]
        else:
            spectral = saved["fnr_percent"]["NRSW_COMMON_UNIT"]
            diagonal = saved["fnr_percent"]["TAG_DIAGONAL"]
        print(f"{label} spectral FNR (%)  {rates(spectral)}")
        print(f"{label} diagonal FNR (%)  {rates(diagonal)}")
        print(f"{label} spectral FNR reduction (pp)  {rates(np.subtract(diagonal, spectral))}")
    effects = svk["crossed_weights_and_strength_gain_pp"]
    strength_only = effects["spectral_strength_given_diagonal_weights"]["point_pp"]
    added_weights = effects["spectral_weights_given_spectral_strength"]["point_pp"]
    if replayed:
        crossed = replayed["crossed_weights_and_strength"]
        strength_only = np.subtract(crossed["DD"]["roc_fnr_percent"], crossed["DS"]["roc_fnr_percent"])
        added_weights = np.subtract(crossed["DS"]["roc_fnr_percent"], crossed["SS"]["roc_fnr_percent"])
    print(f"Spectral strength with diagonal weights, FNR reduction (pp)  "
          f"{rates(strength_only)}")
    print(f"Spectral weights with spectral strength, FNR reduction (pp)  "
          f"{rates(added_weights)}")
    print(replayed["scope"] if replayed else svk["scope"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay the R62 headline results from archived scores.")
    parser.add_argument("archive", type=Path, help="Original NRSW_R62_submission_package.zip")
    parser.add_argument("--svk-scores", type=Path, help="Optional NRSW_R62_SVK_rank_score_replay.zip companion")
    parser.add_argument("--json", action="store_true", help="Print structured rates and pairing intervals")
    parser.add_argument("--extract-r5", type=Path, metavar="ZIP", help="Copy the verified R29 R5 runtime ZIP to a new file and exit")
    args = parser.parse_args(argv)
    if args.extract_r5 is not None and (args.svk_scores is not None or args.json):
        parser.error("--extract-r5 is a separate operation")
    try:
        if args.extract_r5 is not None:
            with _r62_evidence(args.archive) as evidence:
                data = evidence.read(RAW_RUNTIME)
            with args.extract_r5.open("xb") as stream:
                stream.write(data)
            print(f"Copied verified R29 R5 runtime to {args.extract_r5}")
            return 0
        report = reproduce(args.archive, args.svk_scores)
    except (OSError, ValueError, BadZipFile, KeyError) as exc:
        parser.exit(1, f"R62 reproduction failed: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _display(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
