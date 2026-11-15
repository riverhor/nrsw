"""Field-specific document refit and optional MobileNet and native-pixel heads."""
from dataclasses import dataclass
import hashlib

import numpy as np

from .kernel import AnalyticHead, JointModel, TrainBatch, fit_head, fit_joint
from .schema import PIXEL_DIMENSION


METHOD = "NRSW_FIELD_CLASSIFIERS_V1"
PIXEL_WINDOW_SIDES = (12, 24, 48)


def ink_descriptor(gray):
    """Measure intermediate-dark pixels in a native-resolution uint8 grayscale crop."""
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("a native grayscale crop must be a uint8 matrix")
    height, width = gray.shape
    if min(height, width) < max(PIXEL_WINDOW_SIDES):
        return None

    import cv2

    counts = np.cumsum(np.bincount(gray.ravel(), minlength=256))
    positions = (gray.size - 1) * np.array([.01, .95])
    low_bins = np.searchsorted(counts, positions.astype(int), side="right")
    high_bins = np.searchsorted(counts, np.ceil(positions).astype(int), side="right")
    low, high = low_bins + (positions % 1) * (high_bins - low_bins)
    darkness = np.clip((high - gray.astype(np.float32)) / max(high - low, 32.), 0., 1.).astype(np.float32)
    nearest_darkness = cv2.dilate(darkness, np.ones((5, 5), dtype=np.uint8), borderType=cv2.BORDER_REPLICATE)
    intermediate = np.maximum(0., 1. - np.abs(darkness - .44) / .26)
    unsupported = intermediate * np.clip((.80 - nearest_darkness) / .20, 0., 1.)
    descriptor = [float(unsupported.mean(dtype=np.float64))]
    for side in PIXEL_WINDOW_SIDES:
        average = cv2.boxFilter(unsupported, cv2.CV_32F, (side, side), anchor=(0, 0), normalize=True)
        valid = average[:height - side + 1, :width - side + 1]
        descriptor.extend((np.quantile(valid, .99), valid.max()))
    return np.asarray(descriptor, dtype=np.float64)


@dataclass(frozen=True)
class TailReference:
    upper: float
    spread: float

    def standardize(self, scores):
        standardized = (scores - self.upper) / self.spread
        if not np.isfinite(standardized).all():
            raise ArithmeticError("field classifier produced a nonfinite standardized score")
        return standardized


def train_tail_reference(scores, *, use_std_floor=False):
    """Fit the score origin and scale on genuine training scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("a score reference needs at least two finite genuine training scores")
    median, upper = np.quantile(scores, (.50, .95))
    spread = upper - median
    if use_std_floor and np.ptp(scores) > 0:
        spread = max(spread, scores.std())
    return TailReference(float(upper), float(spread)) if np.isfinite(spread) and spread > 0 else None


@dataclass(frozen=True)
class FieldHead:
    head: AnalyticHead
    reference: TailReference


def _measurements(batch, pixel_features, pixel_available, signed_available):
    shape = (len(batch), batch.layout.m)
    pixels = np.asarray(pixel_features, dtype=np.float64)
    pixel_mask = np.asarray(pixel_available)
    if (pixels.shape != (*shape, PIXEL_DIMENSION) or not np.isfinite(pixels).all() or (pixels < 0).any()
            or pixel_mask.shape != shape or pixel_mask.dtype != np.bool_):
        raise ValueError("pixel fields require finite nonnegative [N,m,7] descriptors and a boolean [N,m] mask")
    if signed_available is None:
        signed_mask = np.broadcast_to(batch.local_valid[:, None], shape)
    else:
        signed_mask = np.asarray(signed_available)
        if signed_mask.shape != shape or signed_mask.dtype != np.bool_:
            raise ValueError("signed field availability requires a boolean [N,m] mask")
        if not np.isfinite(batch.local_features[signed_mask]).all():
            raise ValueError("available signed fields must have finite descriptors")
    return pixels, pixel_mask, signed_mask


def _fit_fields(values, available, labels, candidates, *, pixel=False):
    heads = []
    for j in range(values.shape[1]):
        genuine = np.flatnonzero((labels == 0) & available[:, j])
        positive = np.flatnonzero(candidates[:, j] & available[:, j])
        reference_rows = values[genuine, j]
        branch = None
        if len(genuine) >= 2 and len(positive) >= 2 and np.ptp(reference_rows, axis=0).any():
            positions = np.r_[genuine, positive]
            head = fit_head(values[positions, j], labels[positions])
            reference = train_tail_reference(head.score(reference_rows), use_std_floor=pixel)
            if reference is not None:
                branch = FieldHead(head, reference)
        heads.append(branch)
    return tuple(heads)


@dataclass(frozen=True)
class FieldClassifierModel:
    document: JointModel
    document_reference: TailReference | None
    signed_heads: tuple[FieldHead | None, ...]
    pixel_heads: tuple[FieldHead | None, ...]
    training_identity: str
    metadata: dict

    @property
    def layout(self):
        return self.document.layout

    def score(self, batch, pixel_features, pixel_available, *, signed_available=None):
        """Return the maximum available standardized branch score for each image."""
        scores = self.document.score(batch)
        pixels, pixel_mask, signed_mask = _measurements(batch, pixel_features, pixel_available, signed_available)
        if self.document_reference is None or not any((*self.signed_heads, *self.pixel_heads)):
            return scores

        scores = self.document_reference.standardize(scores)
        usable_pixels = pixel_mask & np.any(pixels > 0, axis=2)
        for values, available, heads in ((batch.local_features, signed_mask, self.signed_heads),
                                         (pixels, usable_pixels, self.pixel_heads)):
            for field, branch in enumerate(heads):
                measured = available[:, field]
                if branch is not None and measured.any():
                    branch_scores = branch.reference.standardize(branch.head.score(values[measured, field]))
                    scores[measured] = np.maximum(scores[measured], branch_scores)
        return scores


def fit_field_classifiers(training: TrainBatch, original: JointModel, pixel_features, pixel_available,
                          *, signed_available=None) -> FieldClassifierModel:
    """Fit field heads and branch references using the original model's training rows."""
    if not isinstance(training, TrainBatch) or not isinstance(original, JointModel):
        raise TypeError("field classifiers require a TrainBatch and original JointModel")
    training.validate()
    batch = training.features
    pixels, pixel_mask, signed_mask = _measurements(batch, pixel_features, pixel_available, signed_available)
    document, _ = fit_joint(training, original.historical_mean, original=original)
    reference_rows = batch.subset(np.flatnonzero(np.asarray(training.roles) == "N"))
    document_reference = train_tail_reference(document.score(reference_rows))

    candidates = np.zeros((len(batch), batch.layout.m), dtype=bool)
    for i, fields in enumerate(training.candidates):
        candidates[i, list(fields)] = True
    labels = training.labels
    signed_heads = _fit_fields(batch.local_features, signed_mask, labels, candidates)
    pixel_heads = _fit_fields(pixels, pixel_mask, labels, candidates, pixel=True)

    digest = hashlib.sha256(f"{METHOD}:{document.training_identity}".encode())
    for values in (signed_mask, batch.local_features[signed_mask], pixel_mask, pixels[pixel_mask]):
        digest.update(values.tobytes())
    metadata = {**original.metadata, "method": METHOD, "original_training_identity": original.training_identity}
    return FieldClassifierModel(document, document_reference, signed_heads, pixel_heads,
                                digest.hexdigest(), metadata)
