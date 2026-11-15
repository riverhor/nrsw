"""Exact binomial-tail NP calibration; forged iff score > threshold."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math

import numpy as np


def choose_np_rank(n: int, alpha_num: int, alpha_den: int,
                   delta_num: int = 1, delta_den: int = 20) -> dict:
    """Archived exact-integer rule, with no rounded tail decisions."""
    if any(type(x) is not int for x in (n, alpha_num, alpha_den, delta_num, delta_den)) or n < 0:
        raise ValueError("integer nonnegative n and integer rational parameters required")
    if not (0 < alpha_num < alpha_den and 0 < delta_num < delta_den):
        raise ValueError("eta and delta must be strictly between zero and one")
    empty = dict(rank=None, bound=None, previous_bound=None)
    if n == 0:
        return empty
    p, q = alpha_num, alpha_den
    denominator, term, tail = q ** n, (q - p) ** n, 0
    rank = numerator = None
    for k in range(n, 0, -1):
        tail += term
        if tail * delta_den > delta_num * denominator:
            if rank is None:
                return empty
            return dict(rank=rank, bound=numerator / denominator, previous_bound=tail / denominator)
        rank, numerator = k, tail
        if k > 1:
            term, remainder = divmod(term * k * p, (n - k + 1) * (q - p))
            if remainder:
                raise ArithmeticError("nonintegral binomial recurrence")
    return dict(rank=rank, bound=numerator / denominator, previous_bound=1.0)


def _probability(value, name: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a probability or rational string") from exc
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def calibration_key(eta) -> str:
    value = _probability(eta, "eta")
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class CalibrationResult:
    eta: float
    delta: float
    n: int
    rank: int | None
    threshold: float | None
    bound: float | None
    previous_bound: float | None
    status: str
    eta_fraction: str
    delta_fraction: str

    @property
    def available(self) -> bool:
        return self.threshold is not None

    def to_dict(self) -> dict:
        return asdict(self)

    def predict(self, scores) -> np.ndarray:
        values = np.asarray(scores, np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("scores must be a finite one-dimensional array")
        if not self.available:
            raise ValueError("calibration unavailable: no finite NP rank; request scores or more genuine calibration documents")
        return (values > self.threshold).astype(np.int64)


def calibrate_scores(normal_scores, eta=0.1, delta=0.05) -> CalibrationResult:
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("normal calibration scores must be finite and one-dimensional")
    a, d = _probability(eta, "eta"), _probability(delta, "delta")
    rule = choose_np_rank(len(scores), a.numerator, a.denominator, d.numerator, d.denominator)
    rank = rule["rank"]
    threshold = None if rank is None else float(np.sort(scores, kind="stable")[rank - 1])
    return CalibrationResult(float(a), float(d), len(scores), rank, threshold,
                             rule["bound"], rule["previous_bound"],
                             "AVAILABLE" if rank is not None else "UNAVAILABLE_NO_FINITE_RANK",
                             str(a), str(d))


def component_error_rates(scores, labels, groups, threshold: float) -> dict:
    """Equal mass per class-present connected group, then per same-class view."""
    scores, labels, groups = np.asarray(scores), np.asarray(labels), np.asarray(groups)
    if scores.ndim != 1 or labels.shape != scores.shape or groups.shape != scores.shape:
        raise ValueError("scores, labels and groups must be matching vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all() or not math.isfinite(threshold):
        raise ValueError("finite scores/threshold and known binary labels required")
    predicted = scores > threshold
    result = {"FN": int(np.sum((labels == 1) & ~predicted)),
              "FP": int(np.sum((labels == 0) & predicted)),
              "TP": int(np.sum((labels == 1) & predicted)),
              "TN": int(np.sum((labels == 0) & ~predicted))}
    for label, name in ((1, "FNR"), (0, "FPR")):
        rates = []
        for group in np.unique(groups[labels == label]):
            selected = (groups == group) & (labels == label)
            rates.append(float(np.mean(predicted[selected] != label)))
        result[name] = float(np.mean(rates)) if rates else None
        result[name + "_groups"] = len(rates)
    return result
