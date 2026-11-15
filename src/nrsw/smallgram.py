"""Fit the genuine-covariance classifier in document space."""
import math

import numpy as np


def _oas(count, width, trace, trace_squared):
    mean_variance = trace / width
    if width == 1:
        # The covariance is already isotropic in a one-dimensional field.
        return 1.0, mean_variance
    numerator = trace_squared + trace * trace
    denominator = (count + 1) * (trace_squared - trace * trace / width)
    if not math.isfinite(denominator) or denominator < 0.0:
        raise ArithmeticError("invalid covariance traces for OAS shrinkage")
    return (1.0 if denominator == 0.0 else min(numerator / denominator, 1.0)), mean_variance


def fit_small_gram(features, labels):
    """Solve the regularized covariance equation without a feature-square matrix."""
    n, width = features.shape
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    means = [np.mean(features[group], axis=0) for group in groups]
    residuals = np.concatenate([features[group] - mean for group, mean in zip(groups, means)])
    raw_gram = residuals @ residuals.T
    gram = (raw_gram + raw_gram.T) * 0.5
    stats, scales = [], []
    offset = 0
    ridge = 0.0
    for label, group in enumerate(groups):
        count = len(group)
        block = gram[offset:offset + count, offset:offset + count]
        trace = float(np.trace(block)) / count
        trace_squared = float(np.sum(block * block)) / (count * count)
        shrinkage, mean_variance = _oas(count, width, trace, trace_squared)
        stats.append(dict(label=label, documents=count, covariance_denominator=count,
                          trace_S_FP64=trace, trace_S_squared_FP64=trace_squared,
                          mean_variance_FP64=mean_variance, shrinkage_FP64=shrinkage,
                          zero_within_class_covariance=trace == 0.0))
        scales.extend([math.sqrt((1.0 - shrinkage) / count) if label == 0 else 0.0] * count)
        ridge += shrinkage * mean_variance if label == 0 else 0.0
        offset += count

    difference = means[1] - means[0]
    if ridge == 0.0:
        if np.any(difference):
            raise ValueError("genuine training covariance is zero while the class means differ")
        coefficients = np.zeros(width, dtype=np.float64)
        intercept = small_residual = coefficient_residual = 0.0
        status = "DEGENERATE_ZERO_POOLED_COVARIANCE_MINNORM"
    else:
        scale = np.asarray(scales, dtype=np.float64)
        system = gram * np.outer(scale, scale) + ridge * np.eye(n, dtype=np.float64)
        right = scale * (residuals @ difference)
        factor = np.linalg.cholesky(system)
        solution = np.linalg.solve(factor.T, np.linalg.solve(factor, right))
        coefficients = (difference - residuals.T @ (scale * solution)) / ridge
        intercept = -0.5 * float((means[1] + means[0]) @ coefficients)
        small_residual = float(np.max(np.abs(system @ solution - right)))
        action = ridge * coefficients + residuals.T @ ((scale * scale) * (residuals @ coefficients))
        coefficient_residual = float(np.max(np.abs(action - difference)))
        status = "IDENTIFIABLE_SHRUNK_COVARIANCE"
    if not np.isfinite(coefficients).all() or not all(map(math.isfinite, (intercept, small_residual, coefficient_residual))):
        raise ArithmeticError("nonfinite classifier fit")
    diagnostics = dict(class_stats=stats, pooled_isotropic_variance_FP64=ridge, status=status,
                       small_system_residual_maxabs_FP64=small_residual,
                       covariance_equation_residual_maxabs_FP64=coefficient_residual)
    return coefficients, intercept, diagnostics
