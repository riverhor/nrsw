"""Check the leading generalized eigenpair of the evidence contrast."""
import math


def eigen_checks(delta, metric, vector, eigenvalue):
    dimension = len(vector)
    av = [math.fsum(float(delta[a][b]) * vector[b] for b in range(dimension))
          for a in range(dimension)]
    bv = [math.fsum(float(metric[a][b]) * vector[b] for b in range(dimension))
          for a in range(dimension)]
    norm = math.fsum(vector[a] * bv[a] for a in range(dimension))
    residual = math.sqrt(math.fsum((a - eigenvalue * b) ** 2 for a, b in zip(av, bv)))
    scale = max(math.sqrt(math.fsum(x * x for x in av)),
                abs(eigenvalue) * math.sqrt(math.fsum(x * x for x in bv)), 1e-300)
    relative = residual / scale
    tolerance = 1e-8
    if not (math.isfinite(norm) and abs(norm - 1.0) <= tolerance
            and math.isfinite(relative) and relative <= tolerance):
        raise ArithmeticError("the leading eigenpair did not satisfy the spectral solve tolerance")
    return {"generalized_eigen_relative_residual": relative, "B_norm_squared": norm,
            "relative_tolerance": tolerance}
