"""Train and use a detector with cached document descriptors."""
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from typing import Iterable

import numpy as np

from .calibration import CalibrationResult, calibrate_scores, calibration_key, component_error_rates
from .field_classifiers import FieldClassifierModel, fit_field_classifiers
from .kernel import FeatureBatch, JointModel, Layout, TrainBatch, fit_joint
from .schema import FeatureDataset, canonical_json
from .state import export_primary as save_model, import_primary as load_model

DEFAULT_FPRS = ("0.01", "0.05", "0.10")
TRAIN_ROLES = {"anchor": "N", "support": "F", "bank": "BANK"}


def extractor_identity(extractor: dict) -> str:
    return hashlib.sha256(canonical_json(extractor).encode()).hexdigest()


def feature_batch(dataset: FeatureDataset) -> FeatureBatch:
    dg, _, d = dataset.dimensions
    extractor = extractor_identity(dataset.metadata["extractor"])
    layout = Layout(dataset.metadata["layout_id"], tuple(dataset.metadata["field_order"]), dg, d, extractor)
    return FeatureBatch(layout, tuple(dataset.ids.tolist()), tuple(dataset.groups.tolist()),
                        dataset.global_features, dataset.local_features, dataset.global_mobile,
                        dataset.local_valid)


def _select(dataset: FeatureDataset, roles: Iterable[str]) -> FeatureDataset:
    dataset.validate()
    roles = tuple(roles)
    selected = np.isin(dataset.roles, roles)
    if not selected.any():
        raise ValueError(f"feature packet has no {' or '.join(roles)} rows")
    return dataset.subset(selected)


def _lineage(dataset: FeatureDataset) -> dict:
    return {"ids": sorted(set(dataset.ids)), "groups": sorted(set(dataset.groups)),
            "image_sha256": sorted(set(dataset.image_hashes().values()))}


def _require_disjoint(dataset: FeatureDataset, earlier: dict, name: str) -> None:
    for key, values in _lineage(dataset).items():
        if set(values).intersection(earlier[key]):
            raise ValueError(f"{name} overlap in {key}; keep document components in one split")


def _folds(training: FeatureDataset) -> np.ndarray:
    if training.inner_folds is not None:
        return training.inner_folds
    folds = np.full(len(training), -1, dtype=np.int64)
    for role in ("anchor", "support"):
        indices = np.flatnonzero(training.roles == role)
        order = sorted(indices, key=lambda i: training.groups[i])
        folds[order] = np.arange(len(order)) % 2
    return folds


def fit(dataset: FeatureDataset, historical_mean=None, *, field_classifiers=False) -> JointModel | FieldClassifierModel:
    """Fit on anchor, support and bank rows. The optional mean is ordered [SD, MobileNet]."""
    training = _select(dataset, TRAIN_ROLES)
    if field_classifiers and training.pixel_features is None:
        raise ValueError("field classifiers need pixel measurements; run extract --field-classifiers")
    features = feature_batch(training)
    roles = tuple(TRAIN_ROLES[role] for role in training.roles)
    candidates = tuple(tuple(np.flatnonzero(row).tolist()) for row in training.candidate_mask)
    folds = _folds(training)
    if historical_mean is None:
        anchors = training.roles == "anchor"
        if not anchors.any():
            raise ValueError("training requires genuine anchor documents")
        pair = np.concatenate((features.global_features[anchors], features.global_mobile[anchors]), axis=1)
        historical_mean = pair.astype(np.float64).mean(axis=0)
    train_batch = TrainBatch(features, roles, folds, candidates)
    model, _ = fit_joint(train_batch, np.asarray(historical_mean, np.float64))
    model.metadata["training_lineage"] = _lineage(training)
    if field_classifiers:
        return fit_field_classifiers(train_batch, model, training.pixel_features, training.pixel_available,
                                     signed_available=training.field_available)
    return model


@dataclass(frozen=True)
class Calibration:
    training_identity: str
    lineage: dict
    cuts: dict[str, CalibrationResult]

    def to_dict(self) -> dict:
        return {"schema": "nrsw.calibration.v1", "training_identity": self.training_identity,
                "lineage": self.lineage, "cuts": {key: result.to_dict() for key, result in self.cuts.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        if not isinstance(data, dict) or data.get("schema") != "nrsw.calibration.v1":
            raise ValueError("unsupported calibration format")
        try:
            return cls(data["training_identity"], data["lineage"],
                       {key: CalibrationResult(**result) for key, result in data["cuts"].items()})
        except (KeyError, AttributeError, TypeError) as error:
            raise ValueError("invalid calibration file") from error

    def cut(self, fpr) -> CalibrationResult:
        key = calibration_key(fpr)
        if key not in self.cuts:
            raise ValueError(f"no threshold for FPR {fpr}; calibrate at this budget first")
        cut = self.cuts[key]
        if not cut.available:
            raise ValueError(f"no finite threshold for FPR {fpr} with {cut.n} genuine calibration documents")
        threshold = cut.threshold
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise ValueError("calibration threshold must be a finite number")
        try:
            finite = math.isfinite(threshold)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("calibration threshold must be a finite number")
        return cut


def _score(model, dataset, batch):
    if isinstance(model, FieldClassifierModel):
        if dataset.pixel_features is None:
            raise ValueError("field classifiers need pixel measurements; run extract --field-classifiers")
        return model.score(batch, dataset.pixel_features, dataset.pixel_available,
                           signed_available=dataset.field_available)
    return model.score(batch)


def calibrate(model: JointModel | FieldClassifierModel, dataset: FeatureDataset, fprs=DEFAULT_FPRS,
              delta="0.05") -> Calibration:
    """Choose thresholds from independent genuine calibration documents."""
    genuine = _select(dataset, ("calibration",))
    if len(set(genuine.groups)) != len(genuine):
        raise ValueError("calibration requires one genuine document per independent component")
    _require_disjoint(genuine, model.metadata["training_lineage"], "training/calibration")
    scores = _score(model, genuine, feature_batch(genuine))
    cuts = {calibration_key(fpr): calibrate_scores(scores, fpr, delta) for fpr in fprs}
    if not cuts:
        raise ValueError("provide at least one false-positive budget")
    return Calibration(model.training_identity, _lineage(genuine), cuts)


def score_rows(model: JointModel | FieldClassifierModel, dataset: FeatureDataset, calibration: Calibration | None = None):
    _require_disjoint(dataset, model.metadata["training_lineage"], "training/query")
    if calibration is not None:
        if model.training_identity != calibration.training_identity:
            raise ValueError("calibration belongs to a different fitted model")
        _require_disjoint(dataset, calibration.lineage, "calibration/query")
    batch = feature_batch(dataset)
    return batch, _score(model, dataset, batch)


def predict(model: JointModel | FieldClassifierModel, dataset: FeatureDataset, calibration: Calibration | None = None,
            fpr="0.01") -> dict:
    """Score inference and evaluation rows. Scores above the cut are flagged."""
    query = _select(dataset, ("inference", "evaluation"))
    batch, scores = score_rows(model, query, calibration)
    cut = calibration.cut(fpr) if calibration is not None else None
    rows = [{"id": identifier, "score": float(score), "local_available": bool(local),
             "flagged": None if cut is None else bool(score > cut.threshold)}
            for identifier, score, local in zip(batch.ids, scores, batch.local_valid)]
    return {"layout": model.layout.layout_id, "threshold": None if cut is None else cut.to_dict(), "rows": rows}


def evaluate(model: JointModel | FieldClassifierModel, dataset: FeatureDataset, calibration: Calibration, fpr="0.01") -> dict:
    """Report group-balanced FNR and FPR at the genuine-calibrated cut."""
    test = _select(dataset, ("evaluation",))
    if set(test.labels) != {0, 1}:
        raise ValueError("evaluation needs labeled genuine and forged documents")
    _, scores = score_rows(model, test, calibration)
    cut = calibration.cut(fpr)
    metrics = component_error_rates(scores, test.labels, test.groups, cut.threshold)
    return {"layout": model.layout.layout_id, "calibration": cut.to_dict(), **metrics}
