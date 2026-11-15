"""Residual evidence, pooling and classifier normalization."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .schema import FeatureDataset

def normal_means(bags: np.ndarray) -> np.ndarray:
    bags = np.asarray(bags, np.float64)
    if bags.ndim != 3 or len(bags) < 2 or not np.isfinite(bags).all():
        raise ValueError("at least two complete finite genuine bags required")
    n, m, d = bags.shape
    return np.asarray([[math.fsum(float(bags[i, j, a]) for i in range(n)) / n
                        for a in range(d)] for j in range(m)], np.float64)


def center_residuals(bags: np.ndarray, means: np.ndarray, *, field_only: bool = False) -> np.ndarray:
    """Center each field on its genuine mean and optionally remove document variation."""
    bags, means = np.asarray(bags, np.float64), np.asarray(means, np.float64)
    if bags.ndim != 3 or bags.shape[1:] != means.shape or means.shape[0] < 2:
        raise ValueError("bags and field means have incompatible shapes")
    residual = bags - means[None]
    if field_only:
        return residual
    n, m, d = residual.shape
    common = np.asarray([[math.fsum(float(v) for v in residual[i, :, a]) / m
                          for a in range(d)] for i in range(n)], np.float64).reshape(n, d)
    return residual - common[:, None, :]


@dataclass
class StrengthReference:
    mean: float
    std: float
    active: bool
    rows: int

    def standardize(self, peak: np.ndarray) -> np.ndarray:
        peak = np.asarray(peak, np.float64)
        if not np.isfinite(peak).all() or np.any(peak < 0):
            raise ValueError("peak evidence must be finite and nonnegative")
        return (peak - self.mean) / self.std if self.active else np.zeros_like(peak)


@dataclass
class EvidenceModel:
    means: np.ndarray
    factor: np.ndarray
    studentizers: np.ndarray
    eigenvalues: np.ndarray
    leading_vector: np.ndarray
    leading_studentizers: np.ndarray
    active: bool
    stats: dict

    def evaluate(self, bags: np.ndarray, *, leading: bool = False) -> dict:
        bags = np.asarray(bags, np.float64)
        if bags.ndim != 3 or bags.shape[1:] != self.means.shape or not np.isfinite(bags).all():
            raise ValueError("complete finite local descriptors required by evidence model")
        n, m, _ = bags.shape
        if self.active:
            residual = center_residuals(bags, self.means,
                                        field_only=self.stats.get("residual_mode") == "field_only")
            if leading:
                projected = residual @ self.leading_vector
                energy = projected * projected / self.leading_studentizers[None]
            else:
                projected = residual @ self.factor
                energy = np.sum(projected * projected, axis=2) / self.studentizers[None]
        else:
            energy = np.zeros((n, m), np.float64)
        if not np.isfinite(energy).all() or np.any(energy < 0):
            raise ArithmeticError("nonfinite or negative fitted energies")
        # Match the precision used by the pooled descriptors.
        alpha = np.empty((n, m), np.float32)
        for i, row in enumerate(energy.tolist()):
            total = math.fsum(row)
            alpha[i] = [v / total for v in row] if total else [1.0 / m] * m
        return {"energy": energy, "alpha": alpha,
                "peak": np.max(energy, axis=1) if n else np.empty(0, np.float64)}


def fit_evidence(normal_bags: np.ndarray, forged_bags: np.ndarray,
                 candidate_mask: np.ndarray, *, field_only: bool = False) -> EvidenceModel:
    """Candidate-matched moments, finite-reference correction, positive modes."""
    normal = np.asarray(normal_bags, np.float64)
    forged = np.asarray(forged_bags, np.float64)
    candidates = np.asarray(candidate_mask)
    if normal.ndim != 3 or forged.ndim != 3 or normal.shape[1:] != forged.shape[1:]:
        raise ValueError("genuine and forged bags must have matching [N,m,d] shapes")
    n_n, m, d = normal.shape
    n_f = len(forged)
    if n_n < 2 or n_f < 1 or m < 2 or d < 1:
        raise ValueError("each evidence fit needs >=2 genuine, >=1 forged, >=2 fields")
    if candidates.dtype != np.bool_ or candidates.shape != (n_f, m) or np.any(~candidates.any(axis=1)):
        raise ValueError("every fitting forgery needs a nonempty boolean candidate mask")
    if not np.isfinite(normal).all() or not np.isfinite(forged).all():
        raise ValueError("evidence fits require complete finite local descriptors")
    means = normal_means(normal)
    residual = center_residuals(np.concatenate((normal, forged)), means, field_only=field_only)
    rn, rf = residual[:n_n], residual[n_n:]
    pi = candidates.astype(np.float64) / candidates.sum(axis=1)[:, None]
    p = pi.mean(0)
    # Retain native sqrt(m*p) Gram order and original m denominators.
    nr = (rn * np.sqrt(m * p)[None, :, None]).reshape(-1, d)
    fr = (rf * np.sqrt(m * pi)[:, :, None]).reshape(-1, d)
    cn = (nr.T @ nr) / (m * (n_n - 1))
    cf = (fr.T @ fr) / (m * n_f)
    tau = math.fsum(float(cn[a, a]) for a in range(d)) / d
    kappa = 1.0 + 1.0 / n_n
    stats = {"normal_count": n_n, "fake_count": n_f, "fields": m, "dimension": d,
             "kappa": kappa, "tau": tau, "field_marginal": p.tolist(),
             "positive_modes": 0, "normal_denominator": m * (n_n - 1),
             "fake_denominator": m * n_f}
    if field_only:
        stats["residual_mode"] = "field_only"
    result = EvidenceModel(means, np.empty((d, 0), np.float64), np.empty(0, np.float64),
                           np.empty(0, np.float64), np.empty(0, np.float64),
                           np.empty(0, np.float64), False, stats)
    if tau == 0:
        stats["status"] = "UNIFORM_ZERO_NORMAL_TRACE"
        return result
    if not math.isfinite(tau) or tau < 0:
        raise ArithmeticError("invalid genuine trace")
    contrast, metric = cf - kappa * cn, kappa * cn + tau * np.eye(d)
    lower = np.linalg.cholesky(metric)
    left = np.linalg.solve(lower, contrast)
    whitened = np.linalg.solve(lower, left.T).T
    asym = float(np.linalg.norm(whitened - whitened.T)) / max(float(np.linalg.norm(whitened)), 1e-300)
    if not math.isfinite(asym) or asym > 1e-8:
        raise ArithmeticError("whitened contrast exceeds native symmetry tolerance")
    values, vectors = np.linalg.eigh((whitened + whitened.T) * 0.5)
    positive = values > 0.0
    result.eigenvalues = values
    stats.update(leading_eigenvalue=float(values[-1]), positive_modes=int(positive.sum()),
                 whitened_asymmetry_relative=asym)
    if not positive.any():
        stats["status"] = "UNIFORM_NONPOSITIVE_CONTRAST"
        return result
    leading = np.linalg.solve(lower.T, vectors[:, -1])
    leading = leading / math.sqrt(float(leading @ metric @ leading))
    pn = rn @ leading
    lead_den = np.asarray([kappa * math.fsum(float(row[j]) ** 2 for row in pn) / (n_n - 1)
                           + tau * float(leading @ leading) for j in range(m)], np.float64)
    factor = np.linalg.solve(lower.T, vectors[:, positive]) * np.sqrt(values[positive])[None]
    projected = rn @ factor
    trace = float(np.sum(factor * factor))
    den = kappa * np.sum(projected * projected, axis=(0, 2)) / (n_n - 1) + tau * trace
    if not np.isfinite(den).all() or not (den > 0).all():
        raise ArithmeticError("invalid genuine field studentizers")
    result.factor, result.studentizers = factor, den
    result.leading_vector, result.leading_studentizers = leading, lead_den
    result.active = True
    stats.update(status="FULL_POSITIVE", factor_trace_W=trace)
    return result


def fit_strength_reference(evidence: EvidenceModel, bank_bags: np.ndarray) -> StrengthReference:
    if len(bank_bags) < 2:
        raise ValueError("at least two local-eligible genuine bank documents are required")
    peak = evidence.evaluate(bank_bags)["peak"]
    mean = float(np.mean(peak))
    std = float(np.sqrt(np.mean((peak - mean) ** 2)))
    return StrengthReference(mean, std, evidence.active and std > 0, len(peak))


def pool_aligned(dataset: FeatureDataset, alpha: np.ndarray,
                 delta: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Native FP64 alignment -> FP32, then archived four-accumulator reduction."""
    dg, m, d = dataset.dimensions
    alpha = np.asarray(alpha)
    if alpha.dtype != np.float32 or alpha.shape != (len(dataset), m):
        raise ValueError("pooling alpha must be FP32 [N,m]")
    if delta.shape != (m, d) or offset.shape != (dg + d,):
        raise ValueError("alignment dimensions disagree with feature contract")
    raw = np.concatenate((dataset.global_features, dataset.global_mobile), axis=1)
    out = (raw.astype(np.float64) - offset).astype(np.float32)
    positions = np.flatnonzero(dataset.local_valid)
    if len(positions):
        raw_local = np.concatenate((np.repeat(dataset.global_features[positions, None], m, axis=1),
                                    dataset.local_features[positions]), axis=2)
        aligned = raw_local.astype(np.float64) - offset
        aligned[:, :, dg:] -= delta[None]
        aligned = aligned.astype(np.float32)
        accum = [np.zeros((len(positions), dg + d), np.float32) for _ in range(4)]
        for j in range(m):
            product = np.multiply(aligned[:, j], alpha[positions, j, None], dtype=np.float32)
            np.add(accum[j % 4], product, out=accum[j % 4])
        for j in (1, 2, 3):
            np.add(accum[0], accum[j], out=accum[0])
        out[positions] = accum[0]
    if not np.isfinite(out).all():
        raise ArithmeticError("nonfinite aligned representation")
    return out


@dataclass
class AnalyticHead:
    mean: np.ndarray
    rms: float
    w: np.ndarray
    b: float
    stats: dict

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError("invalid classifier representation")
        score = ((features - self.mean) / self.rms) @ self.w + self.b
        if not np.isfinite(score).all():
            raise ArithmeticError("nonfinite classifier scores")
        return score


def genuine_normalization(features: np.ndarray) -> tuple[np.ndarray, float]:
    genuine = np.asarray(features, np.float64)
    mean = genuine.mean(0)
    rms = float(np.sqrt(np.mean(np.sum((genuine - mean) ** 2, axis=1))))
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("genuine vector RMS is zero; native analytic head has no epsilon repair")
    return mean, rms
