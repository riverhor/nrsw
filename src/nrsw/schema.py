"""Portable feature packets and explicit registered-layout contracts.

NPZ archives contain numeric or Unicode arrays only; loading never enables
pickle. Descriptors are stored as FP32, as in the archived feature pipeline.
``local_valid`` is document-level: every field must exist to use local evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from zipfile import BadZipFile

import numpy as np

FEATURE_SCHEMA = "nrsw.features.v1"
PIXEL_DIMENSION = 7
ROLES = {"anchor", "bank", "support", "calibration", "evaluation", "inference"}
ARRAY_NAMES = (
    "global_features", "local_features", "global_mobile", "local_valid",
    "labels", "roles", "groups", "ids", "candidate_mask",
)
OPTIONAL_ARRAY_NAMES = ("inner_folds", "pixel_features", "pixel_available", "field_available")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_image_sha256(value: Any, field: str = "image_sha256") -> str:
    """Validate a declared encoded-image digest, returning canonical lowercase."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA256")
    return value.lower()


def reference_hashes(extractor: dict) -> set[str]:
    """Read optional registered-reference image hashes from an extractor contract."""
    if not isinstance(extractor, dict):
        raise ValueError("extractor contract must be an object")
    if "reference_images" not in extractor:
        return set()
    references = extractor["reference_images"]
    if not isinstance(references, list):
        raise ValueError("extractor.reference_images must be a list of objects")
    hashes = set()
    for index, row in enumerate(references):
        if not isinstance(row, dict):
            raise ValueError(f"reference_images[{index}] must be an object")
        hashes.add(validate_image_sha256(row.get("sha256"), f"reference_images[{index}].sha256"))
    return hashes


@dataclass
class FeatureDataset:
    global_features: np.ndarray
    local_features: np.ndarray
    global_mobile: np.ndarray
    local_valid: np.ndarray
    labels: np.ndarray
    roles: np.ndarray
    groups: np.ndarray
    ids: np.ndarray
    candidate_mask: np.ndarray
    metadata: dict
    inner_folds: np.ndarray | None = None
    pixel_features: np.ndarray | None = None
    pixel_available: np.ndarray | None = None
    field_available: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("global_features", "local_features", "global_mobile"):
            a = np.asarray(getattr(self, name))
            if a.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be numeric")
            setattr(self, name, a.astype(np.float32))
        if self.pixel_features is not None:
            self.pixel_features = np.asarray(self.pixel_features, dtype=np.float64)
        for name in ("local_valid", "candidate_mask", "pixel_available", "field_available"):
            if getattr(self, name) is None and name in ("pixel_available", "field_available"):
                continue
            a = np.asarray(getattr(self, name))
            if a.dtype != np.bool_:
                raise ValueError(f"{name} must have boolean dtype")
            setattr(self, name, a)
        for name in ("labels", "inner_folds"):
            a = getattr(self, name)
            if a is None:
                continue
            a = np.asarray(a)
            if a.dtype.kind not in "iu":
                raise ValueError(f"{name} must have integer dtype")
            setattr(self, name, a.astype(np.int64))
        for name in ("roles", "groups", "ids"):
            a = np.asarray(getattr(self, name))
            if a.dtype.kind not in "US":
                raise ValueError(f"{name} must be a Unicode/string array, not object dtype")
            setattr(self, name, a.astype(str))
        self.metadata = json.loads(canonical_json(self.metadata))
        self.validate()

    def __len__(self) -> int:
        return len(self.global_features)

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """Return (global dimension, field count, local dimension)."""
        return self.global_features.shape[1], self.local_features.shape[1], self.local_features.shape[2]

    def validate(self) -> None:
        g, z, h = self.global_features, self.local_features, self.global_mobile
        if g.ndim != 2 or z.ndim != 3 or h.ndim != 2:
            raise ValueError("features must have shapes [N,dg], [N,m,d], [N,d]")
        n, dg = g.shape
        if dg < 1 or z.shape[1] < 2 or z.shape[2] < 1:
            raise ValueError("require dg>=1, m>=2 and d>=1")
        if z.shape[0] != n or h.shape != (n, z.shape[2]):
            raise ValueError("global/local descriptor shapes disagree")
        for name in ("labels", "roles", "groups", "ids", "local_valid"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape [N]")
        if self.candidate_mask.shape != z.shape[:2]:
            raise ValueError("candidate_mask must have shape [N,m]")
        if self.inner_folds is not None and self.inner_folds.shape != (n,):
            raise ValueError("inner_folds must have shape [N]; use -1 outside anchor/support")
        if not np.isfinite(g).all() or not np.isfinite(h).all():
            raise ValueError("global descriptors must be finite for every document")
        if not np.isfinite(z[self.local_valid]).all():
            raise ValueError("eligible local descriptors must contain all finite fields")
        if (self.pixel_features is None) != (self.pixel_available is None):
            raise ValueError("pixel_features and pixel_available must be supplied together")
        if self.pixel_features is not None:
            if self.pixel_features.shape != (*z.shape[:2], PIXEL_DIMENSION) or self.pixel_available.shape != z.shape[:2]:
                raise ValueError("pixel features and availability must have shapes [N,m,7] and [N,m]")
            if not np.isfinite(self.pixel_features).all() or np.any(self.pixel_features < 0):
                raise ValueError("pixel features must be finite and nonnegative")
        if self.field_available is not None:
            if self.field_available.shape != z.shape[:2]:
                raise ValueError("field_available must have shape [N,m]")
            if not np.array_equal(self.field_available.all(axis=1), self.local_valid):
                raise ValueError("local_valid must agree with complete field availability")
            if not np.isfinite(z[self.field_available]).all():
                raise ValueError("available field descriptors must be finite")
        if not np.isin(self.labels, [-1, 0, 1]).all():
            raise ValueError("labels must be -1 (unlabelled), 0 (genuine), or 1 (forged)")
        if not set(self.roles.tolist()) <= ROLES:
            raise ValueError(f"roles must belong to {sorted(ROLES)}")
        if any(not x for x in self.ids) or len(set(self.ids.tolist())) != n:
            raise ValueError("ids must be nonempty and unique")
        if any(not x for x in self.groups):
            raise ValueError("groups must be nonempty dependency-component identifiers")
        for role in ("anchor", "bank", "calibration"):
            if np.any(self.labels[self.roles == role] != 0):
                raise ValueError(f"{role} rows must be labelled genuine (0)")
        support = self.roles == "support"
        if np.any(self.labels[support] != 1):
            raise ValueError("support rows must be labelled forged (1)")
        if np.any(~self.candidate_mask[support].any(axis=1)):
            raise ValueError("every forged support row needs a nonempty candidate set")
        if self.candidate_mask[~support].any():
            raise ValueError("candidate annotations are allowed only on forged support rows")
        # A component may contain multiple views, but cannot cross dataset roles.
        assignments: dict[str, str] = {}
        for group, role in zip(self.groups.tolist(), self.roles.tolist()):
            previous = assignments.setdefault(group, role)
            if role != previous:
                raise ValueError(f"group leakage: component {group!r} crosses {previous}/{role} roles")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        if not isinstance(self.metadata.get("layout_id"), str) or not self.metadata["layout_id"]:
            raise ValueError("metadata.layout_id is required")
        fields = self.metadata.get("field_order")
        if not isinstance(fields, list) or len(fields) != z.shape[1] or not all(isinstance(x, str) and x for x in fields) or len(set(fields)) != len(fields):
            raise ValueError("metadata.field_order must list every field exactly once, in descriptor order")
        if not isinstance(self.metadata.get("extractor"), dict) or not self.metadata["extractor"]:
            raise ValueError("metadata.extractor must describe the frozen feature contract")
        if "schema" in self.metadata and self.metadata["schema"] != FEATURE_SCHEMA:
            raise ValueError("unsupported feature schema")
        images, references = self.image_hashes(), self.reference_image_hashes()
        for identifier, role in zip(self.ids.tolist(), self.roles.tolist()):
            if role in ("support", "calibration", "evaluation") and images.get(identifier) in references:
                raise ValueError(f"reference image leakage: {role} row {identifier!r} is a registered reference image")

    def image_hashes(self) -> dict[str, str]:
        """Known image hashes for selected dataset ids only.

        Subsets intentionally retain the original metadata. Extra receipt rows
        are therefore valid and cannot contribute hashes for unselected ids.
        Feature-only packets without image receipts return an empty mapping.
        """
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        if "row_receipts" not in self.metadata:
            return {}
        receipts = self.metadata["row_receipts"]
        if not isinstance(receipts, list):
            raise ValueError("metadata.row_receipts must be a list of objects")
        selected = set(self.ids.tolist())
        seen, hashes = set(), {}
        for index, row in enumerate(receipts):
            if not isinstance(row, dict):
                raise ValueError(f"row_receipts[{index}] must be an object")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"row_receipts[{index}].id must be a nonempty string")
            if identifier in seen:
                raise ValueError(f"duplicate image receipt id: {identifier!r}")
            seen.add(identifier)
            digest = validate_image_sha256(row.get("image_sha256"), f"row_receipts[{index}].image_sha256")
            if identifier in selected:
                hashes[identifier] = digest
        return hashes

    def reference_image_hashes(self) -> set[str]:
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")
        return reference_hashes(self.metadata.get("extractor"))

    def subset(self, indices) -> "FeatureDataset":
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError("subset indices must be a one-dimensional mask or index list")
        return FeatureDataset(**{name: getattr(self, name)[indices] for name in ARRAY_NAMES},
                              metadata=self.metadata,
                              **{name: None if getattr(self, name) is None else getattr(self, name)[indices]
                                 for name in OPTIONAL_ARRAY_NAMES})

    def contract(self) -> dict:
        return {"layout_id": self.metadata["layout_id"], "field_order": self.metadata["field_order"],
                "extractor": self.metadata["extractor"], "dimensions": list(self.dimensions)}


def save_feature_dataset(dataset: FeatureDataset, path: str | Path) -> Path:
    dataset.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(dataset, name) for name in ARRAY_NAMES}
    for name in OPTIONAL_ARRAY_NAMES:
        if getattr(dataset, name) is not None:
            arrays[name] = getattr(dataset, name)
    arrays["metadata_json"] = np.asarray(canonical_json({**dataset.metadata, "schema": FEATURE_SCHEMA}))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def _load_feature_dataset(path: str | Path) -> FeatureDataset:
    with np.load(Path(path), allow_pickle=False) as packet:
        missing = set(ARRAY_NAMES) | {"metadata_json"}
        missing -= set(packet.files)
        if missing:
            raise ValueError(f"feature packet missing arrays: {sorted(missing)}")
        raw = packet["metadata_json"]
        if raw.shape != () or raw.dtype.kind not in "US":
            raise ValueError("metadata_json must be one Unicode JSON scalar")
        metadata = json.loads(str(raw.item()))
        return FeatureDataset(**{name: packet[name] for name in ARRAY_NAMES}, metadata=metadata,
                              **{name: packet[name] for name in OPTIONAL_ARRAY_NAMES if name in packet})


def load_feature_dataset(path: str | Path) -> FeatureDataset:
    try:
        return _load_feature_dataset(path)
    except (BadZipFile, EOFError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed feature packet: {exc}") from exc
