"""Check IDSpace Slovak ID results from saved score ranks."""
from __future__ import annotations

from hashlib import file_digest, sha256
from io import BytesIO
import json
from math import lcm
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .calibration import choose_np_rank


SVK_REPLAY_SHA256 = "fb0ece96f407bb2deb3bff8e40e7697529a0925fda6144f0f82d96b7c4003bf0"
TARGETS = (1, 5, 10)
RANKS = tuple(choose_np_rank(500, target, 100)["rank"] for target in TARGETS)
NATIVE_METHODS = ("NRSW_COMMON_UNIT", "TAG_DIAGONAL", "GENUINE_ENERGY", "CONCAT_NO_EVIDENCE", "SD", "DS")
JPEG_METHODS = ("JOINT_ALIGNED", "JOINT_UNIFORM", "GLOBAL_ONLY", "NRSW_COMMON_UNIT", "TAG_DIAGONAL")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fold_points(ordinal, cuts, labels, membership):
    """Preserve score ties and use exact integer within-component class mass."""
    counts = np.stack([np.bincount(membership[labels == label], minlength=membership.max() + 1)
                       for label in (0, 1)])
    mass = np.count_nonzero(counts, axis=1)
    _require(np.all(mass > 0), "Both classes are required in an SVK fold")
    views = counts[labels, membership]
    units = [lcm(*np.unique(counts[label, counts[label] > 0]).tolist()) for label in (0, 1)]
    weights = np.asarray(units, dtype=np.int64)[labels] // views
    totals = np.asarray(units, dtype=np.int64) * mass
    points = np.zeros((ordinal.shape[1], 3, len(TARGETS)))
    for fit_scores, fit_cuts in zip(ordinal, cuts, strict=True):
        for method, (scores, limits) in enumerate(zip(fit_scores, fit_cuts, strict=True)):
            order = np.argsort(scores, kind="stable")[::-1]
            sorted_scores = scores[order]
            ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
            forged = labels[order] == 1
            ordered_weights = weights[order]
            fp = np.r_[0, np.cumsum(ordered_weights * ~forged)[ends]] * 100
            tp = np.r_[0, np.cumsum(ordered_weights * forged)[ends]]
            budgets = np.asarray(TARGETS) * totals[0]
            low = np.searchsorted(fp, budgets, side="right") - 1
            fractions = (budgets - fp[low]) / (fp[low + 1] - fp[low])
            fnr = (totals[1] - tp[low] - fractions * (tp[low + 1] - tp[low])) / totals[1]
            ready = scores[:, None] > limits
            false_positives = (weights[labels == 0] @ ready[labels == 0]) / totals[0]
            false_negatives = (weights[labels == 1] @ ~ready[labels == 1]) / totals[1]
            points[method] += (fnr, false_positives, false_negatives)
    return points / len(ordinal), mass


def _metric_fold(packet, variant, labels, membership, methods):
    scores, calibration = packet[f"{variant}_test"], packet[f"{variant}_cal"]
    _require(scores.shape == (96, len(methods), len(labels)) and calibration.shape == (96, len(methods), 500)
             and scores.dtype == calibration.dtype == np.uint16, f"Incorrect {variant} score array")
    cuts = np.sort(calibration, axis=-1)[..., np.array(RANKS) - 1]
    return _fold_points(scores, cuts, labels, membership)


def _check(actual, reference, name):
    reference = np.asarray(reference)
    _require(actual.shape == reference.shape and np.allclose(actual, reference, rtol=0, atol=2e-9),
             f"The recalculated IDSpace result differs from the saved result for {name}")


def _result(values, methods):
    result = {}
    for method, (roc_fnr, cut_fpr, cut_fnr) in zip(methods, 100 * values, strict=True):
        result[method] = dict(roc_fnr_percent=roc_fnr.tolist(), fixed_cut_fpr_percent=cut_fpr.tolist(),
                              fixed_cut_fnr_percent=cut_fnr.tolist())
    return result


def replay(path: Path, native_reference: dict, jpeg_reference: dict, crossed_reference: dict) -> dict:
    """Recompute ROC and fixed-threshold error rates from saved score ranks."""
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
        _require(digest == SVK_REPLAY_SHA256, f"The IDSpace score ZIP must have SHA-256 {SVK_REPLAY_SHA256}. Found {digest}")
        stream.seek(0)
        with ZipFile(stream) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
            native = np.zeros((len(NATIVE_METHODS), 3, len(TARGETS)))
            jpeg = np.zeros((len(JPEG_METHODS), 3, len(TARGETS)))
            full_mass = np.zeros(3, dtype=int)
            total = 0
            for fold in range(4):
                name = f"fold{fold}.npz"
                data = archive.read(name)
                _require(sha256(data).hexdigest() == manifest["files"][name], f"Checksum mismatch for {name} in the IDSpace score ZIP")
                with np.load(BytesIO(data), allow_pickle=False) as packet:
                    labels, membership = packet["label"], packet["component"]
                    _require(labels.ndim == 1 and membership.shape == labels.shape and np.isin(labels, (0, 1)).all(),
                             "The IDSpace score file has invalid labels or groups")
                    native_point, mass = _metric_fold(packet, "native", labels, membership, NATIVE_METHODS)
                    jpeg_point, _ = _metric_fold(packet, "jpeg", labels, membership, JPEG_METHODS)
                weights = mass[[1, 0, 1]]
                native += native_point * weights[:, None]
                jpeg += jpeg_point * weights[:, None]
                full_mass += weights
                total += len(labels)
    _require(total == 16210 and full_mass.tolist() == [1183, 1411, 1183], "The IDSpace score file has an unexpected number of test images or groups")
    native /= full_mass[:, None]
    jpeg /= full_mass[:, None]
    for method, point in zip(NATIVE_METHODS[:4], native[:4], strict=True):
        reference = native_reference["methods"][method]
        _check(point, [reference["roc_FNR"], *reference["fixed_cut"]], method)
    indices = [[jpeg_reference["metric_order"].index(f"{name}_{target}") for target in TARGETS]
               for name in ("ROC_FNR", "CAL_FPR", "CAL_FNR")]
    for method, point in zip(JPEG_METHODS, jpeg, strict=True):
        row = np.asarray(jpeg_reference["point"][jpeg_reference["methods"].index(method)])
        _check(point, row[indices], f"common JPEG {method}")

    spectral, diagonal, weights_only, strength_only = native[[0, 1, 4, 5]]
    effects = {"spectral_strength_given_diagonal_weights": diagonal - strength_only,
               "spectral_strength_given_spectral_weights": weights_only - spectral,
               "spectral_weights_given_diagonal_strength": diagonal - weights_only,
               "spectral_weights_given_spectral_strength": strength_only - spectral,
               "total_SS_vs_DD": diagonal - spectral}
    for effect, points in effects.items():
        for kind, point in zip(("roc_FNR", "own_cut_FPR", "own_cut_FNR"), 100 * points, strict=True):
            _check(point, crossed_reference["effects"][kind][effect]["point_pp"], effect)

    return dict(archive_sha256=digest, held_images=total, connected_components=int(full_mass[1]),
                fitted_contexts=384, original_encoding=_result(native[:4], NATIVE_METHODS[:4]),
                crossed_weights_and_strength=_result(native[[0, 4, 5, 1]], ("SS", "SD", "DS", "DD")),
                common_jpeg_refit=_result(jpeg, JPEG_METHODS),
                scope="ROC and fixed-cut point estimates were recomputed. SVK intervals are the archived estimates.")
