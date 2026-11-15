"""Joint detector for a registered document layout."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

from .core import (
    AnalyticHead, EvidenceModel, StrengthReference, center_residuals,
    fit_evidence as _fit_evidence, fit_strength_reference,
    genuine_normalization, pool_aligned,
)
from .smallgram import fit_small_gram
from .spectral_checks import eigen_checks

METHOD = "JOINT_ALIGNED_TARGET_LAYOUT_V1"
FIELD_DOCUMENT_METHOD = "JOINT_FIELD_RESIDUAL_TARGET_LAYOUT_V1"


@dataclass(frozen=True)
class Layout:
    layout_id: str
    fields: tuple[str, ...]
    dg: int
    d: int
    extractor_id: str

    def validate(self):
        if (not self.layout_id or not self.extractor_id or not isinstance(self.fields, tuple)
                or len(self.fields) < 2 or len(set(self.fields)) != len(self.fields)
                or any(not isinstance(v, str) or not v for v in self.fields)
                or type(self.dg) is not int or type(self.d) is not int
                or self.dg < 1 or self.d < 1):
            raise ValueError("explicit layout/unique field order/extractor and positive dimensions required")

    @property
    def m(self):
        return len(self.fields)

    @property
    def width(self):
        return 2 * self.dg + (self.m + 2) * self.d + 1

    @property
    def slices(self):
        a = self.dg
        b = a + self.m * self.d
        c = b + self.dg
        e = c + self.d
        return dict(g=slice(0, a), fields=slice(a, b), pooled_g=slice(b, c),
                    pooled_local=slice(c, e), strength=e, h=slice(e + 1, self.width))


@dataclass(frozen=True)
class FeatureBatch:
    """Unlabeled descriptors. Unavailable local rows may contain NaN."""
    layout: Layout
    ids: tuple[str, ...]
    groups: tuple[str, ...]
    global_features: np.ndarray
    local_features: np.ndarray
    global_mobile: np.ndarray
    local_valid: np.ndarray

    def __len__(self):
        return len(self.ids)

    @property
    def dimensions(self):
        return self.layout.dg, self.layout.m, self.layout.d

    def validate(self):
        self.layout.validate()
        n, s = len(self), self.layout
        if (not isinstance(self.ids, tuple) or not isinstance(self.groups, tuple)
                or len(self.groups) != n or len(set(self.ids)) != n
                or any(not isinstance(v, str) or not v for v in self.ids + self.groups)):
            raise ValueError("unique nonempty string document IDs and explicit component IDs required")
        for name, value, shape in (("g", self.global_features, (n, s.dg)),
                                   ("h", self.global_mobile, (n, s.d)),
                                   ("z", self.local_features, (n, s.m, s.d))):
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
                raise ValueError(f"{name} must be native FP32 with declared dimensions")
        if (not isinstance(self.local_valid, np.ndarray) or self.local_valid.dtype != np.bool_
                or self.local_valid.shape != (n,)):
            raise ValueError("local_valid must be boolean [N]")
        if (not np.isfinite(self.global_features).all() or not np.isfinite(self.global_mobile).all()
                or not np.isfinite(self.local_features[self.local_valid]).all()):
            raise ValueError("all globals and all complete local bags must be finite; no dropped global failures")

    def subset(self, indices):
        positions = np.asarray(indices, dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= len(self)):
            raise ValueError("subset indices out of bounds")
        return FeatureBatch(self.layout, tuple(self.ids[i] for i in positions),
                            tuple(self.groups[i] for i in positions),
                            self.global_features[positions], self.local_features[positions],
                            self.global_mobile[positions], self.local_valid[positions])


@dataclass(frozen=True)
class TrainBatch:
    features: FeatureBatch
    roles: tuple[str, ...]
    folds: np.ndarray
    candidates: tuple[tuple[int, ...], ...]

    @property
    def labels(self):
        return np.asarray([role == "F" for role in self.roles], np.int64)

    def validate(self):
        self.features.validate()
        n, m = len(self.features), self.features.layout.m
        if (not isinstance(self.roles, tuple) or len(self.roles) != n
                or any(role not in ("N", "F", "BANK") for role in self.roles)):
            raise ValueError("training requires anchor, forged support or bank rows")
        if (not isinstance(self.folds, np.ndarray) or self.folds.dtype != np.int64
                or self.folds.shape != (n,)):
            raise ValueError("preassigned int64 seed folds required")
        if not isinstance(self.candidates, tuple) or len(self.candidates) != n:
            raise ValueError("explicit candidate tuple per training row required")
        if len(set(self.features.groups)) != n:
            raise ValueError("one document per independent training component is required by this protocol")
        for role, fold, tags in zip(self.roles, self.folds.tolist(), self.candidates):
            if (role == "BANK" and fold != -1) or (role != "BANK" and fold not in (0, 1)):
                raise ValueError("seed folds must be 0/1 before eligibility; BANK uses -1")
            if not isinstance(tags, tuple):
                raise ValueError("candidate fields must be tuples")
            if role == "F":
                if (len(tags) not in (1, 2) or len(set(tags)) != len(tags)
                        or any(type(j) is not int or not 0 <= j < m for j in tags)):
                    raise ValueError("each support forgery requires one/two distinct valid candidate fields")
            elif tags:
                raise ValueError("genuine documents do not carry edited-field tags")
        if self.roles.count("F") < 2:
            raise ValueError("native receiving head needs at least two forged documents")


def fit_evidence(normal, forged, tags, *, field_only=False):
    """Fit the spectral contrast from genuine and annotated forged fields."""
    normal, forged = np.asarray(normal, np.float64), np.asarray(forged, np.float64)
    if normal.ndim != 3 or forged.ndim != 3 or normal.shape[1:] != forged.shape[1:]:
        raise ValueError("matching complete [N,m,d] evidence bags required")
    m = normal.shape[1]
    if len(tags) != len(forged):
        raise ValueError("one candidate tuple per complete forged fitting row required")
    mask = np.zeros((len(forged), m), bool)
    for i, candidates in enumerate(tags):
        if (not isinstance(candidates, tuple) or len(candidates) not in (1, 2)
                or len(set(candidates)) != len(candidates)
                or any(type(j) is not int or not 0 <= j < m for j in candidates)):
            raise ValueError("invalid one/two-field candidate set")
        mask[i, list(candidates)] = True
    fitted = _fit_evidence(normal, forged, mask, field_only=field_only)
    if not np.isfinite(fitted.eigenvalues).all():
        raise ArithmeticError("nonfinite eigenvalues")
    if fitted.active:
        # Rebuild the contrast to check its leading eigenpair.
        rn, rf = (center_residuals(v, fitted.means, field_only=field_only) for v in (normal, forged))
        pi = mask.astype(np.float64) / mask.sum(1)[:, None]
        nr = (rn * np.sqrt(m * pi.mean(0))[None, :, None]).reshape(-1, normal.shape[2])
        fr = (rf * np.sqrt(m * pi)[:, :, None]).reshape(-1, normal.shape[2])
        cn = nr.T @ nr / (m * (len(normal) - 1))
        cf = fr.T @ fr / (m * len(forged))
        kappa, tau = fitted.stats["kappa"], fitted.stats["tau"]
        metric = kappa * cn + tau * np.eye(normal.shape[2])
        fitted.stats["rank1_checks"] = eigen_checks(
            cf - kappa * cn, metric, fitted.leading_vector.tolist(), float(fitted.eigenvalues[-1]))
        if not (np.isfinite(fitted.leading_studentizers).all()
                and (fitted.leading_studentizers > 0).all()):
            raise ArithmeticError("invalid leading-mode studentizers")
    return fitted


def fit_head(features, labels):
    """Native two-class small-Gram solve, including zero forged covariance weight."""
    x, y = np.asarray(features, np.float64), np.asarray(labels)
    if (x.ndim != 2 or x.shape[1] < 1 or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all()
            or min(int((y == c).sum()) for c in (0, 1)) < 2):
        raise ValueError("finite representation and at least two rows in each class required")
    mean, rms = genuine_normalization(x[y == 0])
    normalized = (x - mean) / rms
    coefficients, intercept, stats = fit_small_gram(normalized, y)
    residual = normalized[y == 0] - normalized[y == 0].mean(0)
    difference = normalized[y == 1].mean(0) - normalized[y == 0].mean(0)
    rho = stats["class_stats"][0]["shrinkage_FP64"]
    ridge = stats["pooled_isotropic_variance_FP64"]
    action = ridge * coefficients + (1 - rho) / len(residual) * residual.T @ (residual @ coefficients)
    error = float(np.max(np.abs(action - difference))) / max(float(np.max(np.abs(difference))), 1e-300)
    if not math.isfinite(error) or error > 1e-8:
        raise ArithmeticError("native head covariance equation residual exceeds 1e-8")
    stats["covariance_relative_infinity_residual"] = error
    return AnalyticHead(mean, rms, coefficients, intercept, stats)


def aligned_parts(batch, delta, offset):
    """Native FP64 offset, then FP64 field correction, then FP32 cast."""
    batch.validate()
    s = batch.layout
    if (not isinstance(delta, np.ndarray) or delta.dtype != np.float64 or delta.shape != (s.m, s.d)
            or not isinstance(offset, np.ndarray) or offset.dtype != np.float64
            or offset.shape != (s.dg + s.d,) or not np.isfinite(delta).all()
            or not np.isfinite(offset).all()):
        raise ValueError("finite FP64 alignment with exact schema dimensions required")
    global_pair = (np.concatenate((batch.global_features, batch.global_mobile), 1).astype(np.float64) - offset).astype(np.float32)
    local = np.zeros((len(batch), s.m, s.d), np.float32)
    good = batch.local_valid
    values = batch.local_features[good].astype(np.float64) - offset[s.dg:]
    values -= delta[None]
    local[good] = values.astype(np.float32)
    return global_pair, local


def assemble_joint(batch, delta, offset, alpha, standardized, unit, local_mean):
    """[g, signed fields, actual FP32 pooled g/local, scaled T, always h]."""
    pair, fields = aligned_parts(batch, delta, offset)
    s = batch.layout
    alpha, standardized, local_mean = map(np.asarray, (alpha, standardized, local_mean))
    if (alpha.dtype != np.float32 or alpha.shape != (len(batch), s.m)
            or not np.isfinite(alpha).all() or (alpha < 0).any()
            or (alpha[batch.local_valid].sum(1, dtype=np.float64) <= 0).any()
            or standardized.dtype != np.float64 or standardized.shape != (len(batch),)
            or not np.isfinite(standardized).all() or (standardized[~batch.local_valid] != 0).any()
            or not math.isfinite(unit) or unit <= 0
            or local_mean.dtype != np.float64 or local_mean.shape != (s.m * s.d,)
            or not np.isfinite(local_mean).all()):
        raise ValueError("invalid native alpha/strength/unit/imputation input")
    pooled = pool_aligned(batch, alpha, delta, offset)
    signed = fields.reshape(len(batch), s.m * s.d).astype(np.float64)
    signed[~batch.local_valid] = local_mean
    result = np.column_stack((pair[:, :s.dg], signed, pooled,
                              unit * standardized, pair[:, s.dg:]))
    if result.shape != (len(batch), s.width) or result.dtype != np.float64 or not np.isfinite(result).all():
        raise ArithmeticError("invalid final joint representation")
    return result


@dataclass(frozen=True)
class Teacher:
    evidence: EvidenceModel
    strength: StrengthReference
    fit_ids: tuple[str, ...]
    bank_ids: tuple[str, ...]
    served_fold: int | None

    def project(self, batch):
        """No query labels, tags, component metadata or images are consulted."""
        batch.validate()
        s = batch.layout
        if self.evidence.means.shape != (s.m, s.d):
            raise ValueError("teacher field schema dimension mismatch")
        alpha = np.zeros((len(batch), s.m), np.float32)
        leading = alpha.copy()
        strength = np.zeros(len(batch), np.float64)
        good = batch.local_valid
        values = self.evidence.evaluate(batch.local_features[good])
        alpha[good] = values["alpha"]
        leading[good] = self.evidence.evaluate(batch.local_features[good], leading=True)["alpha"]
        strength[good] = self.strength.standardize(values["peak"])
        return alpha, leading, strength


@dataclass(frozen=True)
class JointModel:
    layout: Layout
    delta: np.ndarray
    offset: np.ndarray
    historical_mean: np.ndarray
    teachers: tuple[Teacher, Teacher, Teacher]
    strength_unit: float
    local_mean: np.ndarray
    head: AnalyticHead
    training_identity: str
    metadata: dict

    def transform(self, batch):
        if batch.layout != self.layout:
            raise ValueError("Document features must use the model's layout, semantic order of fields, and encoders")
        alpha, _, strength = self.teachers[2].project(batch)
        return assemble_joint(batch, self.delta, self.offset, alpha, strength,
                              self.strength_unit, self.local_mean)

    def score(self, batch):
        return self.head.score(self.transform(batch))


def _training_identity(training, historical_mean, offset, offset_source):
    batch = training.features
    digest = hashlib.sha256()
    digest.update(repr((batch.layout, batch.ids, batch.groups, training.roles,
                        training.candidates, offset_source)).encode())
    for value in (batch.global_features, batch.local_features[batch.local_valid], batch.global_mobile,
                  batch.local_valid, training.folds, historical_mean, offset):
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_joint(training: TrainBatch, historical_mean: np.ndarray, *, frozen_offset=None,
              original: JointModel | None = None):
    """Fit two complementary field estimators, a serving estimator and a classifier.

    Returns (model, OOF_training_features). model.transform serves NEW rows with
    the full teacher; its output for seed training rows is not their OOF frame.
    Every genuine anchor needs complete fields. Forged support and bank rows
    with missing fields still contribute to the classifier through global features.
    An original fit from the same training inputs supplies the common coordinate
    parameters for refitting the document branch with field-only residuals.
    """
    if not isinstance(training, TrainBatch):
        raise TypeError("training must be a TrainBatch")
    training.validate()
    batch, roles, labels = training.features, np.asarray(training.roles), training.labels
    s = batch.layout
    reference = np.asarray(historical_mean)
    if reference.dtype != np.float64 or reference.shape != (s.dg + s.d,) or not np.isfinite(reference).all():
        raise ValueError("The genuine reference mean must be a 64-bit vector of the expected length")
    if not batch.local_valid[roles == "N"].all():
        raise ValueError("every fixed N reference must have complete fields; context unavailable, no subset offset")
    ni = np.flatnonzero((roles == "N") & batch.local_valid)
    fi = np.flatnonzero((roles == "F") & batch.local_valid)
    bi = np.flatnonzero((roles == "BANK") & batch.local_valid)
    if len(ni) < 2 or len(fi) < 1 or len(bi) < 2:
        raise ValueError("full teacher needs >=2 complete N, >=1 complete F, >=2 complete BANK")
    field_only = original is not None
    if field_only:
        if (not isinstance(original, JointModel) or original.layout != s or frozen_offset is not None
                or not np.array_equal(reference, original.historical_mean)
                or original.metadata.get("method") != METHOD):
            raise ValueError("field-only refit requires a matching original joint model")
        delta, offset = original.delta.copy(), original.offset.copy()
        offset_source = original.metadata["offset_source"]
    else:
        delta = (batch.local_features[ni].astype(np.float64).mean(0)
                 - batch.global_mobile[ni].astype(np.float64).mean(0)[None])
        offset = np.concatenate((batch.global_features[ni], batch.global_mobile[ni]), 1).astype(np.float64).mean(0) - reference
        offset_source = "target_N_minus_historical_mean"
        if frozen_offset is not None:
            supplied = np.asarray(frozen_offset)
            if supplied.dtype != np.float64 or supplied.shape != offset.shape or not np.isfinite(supplied).all():
                raise ValueError("Explicit finite FP64 archived coordinate offset required")
            offset = supplied.copy()
            offset_source = "archived_exact_R29"
    identity = _training_identity(training, reference, offset, offset_source)
    if field_only:
        if identity != original.training_identity:
            raise ValueError("field-only refit must use the same training rows, features, folds and tags as the original")
        identity = hashlib.sha256(f"{FIELD_DOCUMENT_METHOD}:{identity}".encode()).hexdigest()
    alpha = np.zeros((len(batch), s.m), np.float32)
    leading = alpha.copy()
    strength = np.zeros(len(batch), np.float64)
    teachers = []
    for held in (0, 1, None):
        normal = ni if held is None else ni[training.folds[ni] != held]
        forged = fi if held is None else fi[training.folds[fi] != held]
        if len(normal) < 2 or len(forged) < 1:
            raise ValueError("fixed complement lacks >=2 complete N and >=1 complete F; no redraw")
        evidence = fit_evidence(batch.local_features[normal], batch.local_features[forged],
                                tuple(training.candidates[i] for i in forged), field_only=field_only)
        norm = fit_strength_reference(evidence, batch.local_features[bi])
        fit_positions = np.flatnonzero(np.isin(np.arange(len(batch)), np.r_[normal, forged]))
        teacher = Teacher(evidence, norm, tuple(batch.ids[i] for i in fit_positions),
                          tuple(batch.ids[i] for i in bi), held)
        teachers.append(teacher)
        serving = np.flatnonzero(roles == "BANK") if held is None else np.flatnonzero(training.folds == held)
        alpha[serving], leading[serving], strength[serving] = teacher.project(batch.subset(serving))
    genuine_available = (labels == 0) & batch.local_valid
    if field_only:
        unit, mean = original.strength_unit, original.local_mean.copy()
    else:
        lead_pool = pool_aligned(batch, leading, delta, offset)
        _, lead_rms = genuine_normalization(lead_pool[labels == 0])
        unit = lead_rms / math.sqrt(s.dg + s.d)
        _, fields = aligned_parts(batch, delta, offset)
        if genuine_available.sum() < 2:
            raise ValueError("at least two complete genuine classifier rows for imputation")
        mean = fields[genuine_available].reshape(int(genuine_available.sum()), s.m * s.d).astype(np.float64).mean(0)
    frame = assemble_joint(batch, delta, offset, alpha, strength, unit, mean)
    head = fit_head(frame, labels)
    metadata = dict(method=FIELD_DOCUMENT_METHOD if field_only else METHOD,
                    dimensions=dict(m=s.m, dg=s.dg, d=s.d, joint=s.width),
                    role_counts={r: training.roles.count(r) for r in ("N", "F", "BANK")},
                    complete_counts=dict(N=len(ni), F=len(fi), BANK=len(bi)),
                    alignment_ids=tuple(batch.ids[i] for i in ni),
                    imputation_ids=tuple(batch.ids[i] for i in np.flatnonzero(genuine_available)),
                    folds=tuple(training.folds.tolist()), historical_core_weights_used=False,
                    strength_unit_source=("original_joint_model" if field_only
                                          else "OOF_seed/full_BANK_R1_genuine_RMS/sqrt(dg+d)"),
                    calibration_or_test_input=False,
                    offset_source=offset_source)
    model = JointModel(s, delta, offset, reference.copy(), tuple(teachers), unit, mean,
                       head, identity, metadata)
    return model, frame
