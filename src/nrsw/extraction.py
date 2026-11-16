"""Extract global and field descriptors from a registered image manifest."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .encoders import MobileEncoder, SDEncoder, runtime_versions, sha256_file
from .field_classifiers import ink_descriptor
from .registration import LayoutRegistrar, MAX_ENCODED, MAX_PIXELS, rectify_map, registration_contract
from .schema import PIXEL_DIMENSION


class InvalidImageError(ValueError):
    """The uploaded bytes do not contain a supported image."""


class ImageAlignmentError(ValueError):
    """The image did not align to the registered document layout."""


def read_manifest(path, field_order, layout_id):
    """Read JSONL, JSON array or CSV; resolve images relative to the manifest."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(records, dict):
            records = records.get("records")
    else:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain at least one image record")
    roles = {"anchor", "bank", "support", "calibration", "evaluation", "inference"}
    parsed = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record {index} is not an object")
        row = dict(record)
        for name in ("image", "id", "group", "role"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Manifest record {index} requires nonempty {name}")
        if row["role"] not in roles:
            raise ValueError(f"Unknown role in record {index}: {row['role']}")
        if row.get("layout_id") != layout_id:
            raise ValueError(f"Record {index} layout_id differs from the supplied layout")
        label = row.get("label", -1)
        try:
            label = int(label) if isinstance(label, str) else label
        except ValueError as exc:
            raise ValueError(f"Invalid label at record {index}") from exc
        if isinstance(label, bool) or label not in (-1, 0, 1):
            raise ValueError(f"Record {index} label must be 0, 1 or -1 (unlabeled)")
        if row["role"] in ("anchor", "bank", "calibration") and label != 0:
            raise ValueError(f"Record {index} {row['role']} requires genuine label 0")
        if row["role"] == "support" and label != 1:
            raise ValueError(f"Record {index} support requires forged label 1")
        candidates = row.get("candidate_fields", [])
        if isinstance(candidates, str):
            candidates = json.loads(candidates) if candidates.strip() else []
        if (not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates)
                or len(candidates) != len(set(candidates)) or not set(candidates) <= set(field_order)):
            raise ValueError(f"Record {index} candidate_fields must be unique names from field_order")
        if row["role"] == "support" and len(candidates) not in (1, 2):
            raise ValueError(f"Record {index} forged support needs one or two candidate fields")
        if row["role"] != "support" and candidates:
            raise ValueError(f"Record {index}: candidate_fields are permitted only for training support, never inference/evaluation")
        inner_fold = row.get("inner_fold")
        if isinstance(inner_fold, str):
            try:
                inner_fold = int(inner_fold) if inner_fold.strip() else None
            except ValueError as exc:
                raise ValueError(f"Invalid inner_fold at record {index}") from exc
        if inner_fold is not None:
            allowed_folds = (0, 1) if row["role"] in ("anchor", "support") else (-1,)
            if type(inner_fold) is not int or inner_fold not in allowed_folds:
                raise ValueError(f"Record {index} {row['role']} inner_fold must be in {allowed_folds}")
        row.update(image=str((path.parent / row["image"]).resolve()), label=label,
                   candidate_fields=candidates, inner_fold=inner_fold)
        if not Path(row["image"]).is_file():
            raise FileNotFoundError(f"Image in record {index} not found: {row['image']}")
        parsed.append(row)
    ids = [r["id"] for r in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest ids must be unique")
    group_roles = {}
    group_folds = {}
    for row in parsed:
        previous = group_roles.setdefault(row["group"], row["role"])
        if previous != row["role"]:
            raise ValueError(f"Connected group {row['group']} spans roles {previous}/{row['role']}")
        if row["role"] in ("anchor", "support"):
            previous_fold = group_folds.setdefault(row["group"], row["inner_fold"])
            if previous_fold != row["inner_fold"]:
                raise ValueError(f"Connected group {row['group']} spans inner folds")
    seed_rows = [row for row in parsed if row["role"] in ("anchor", "support")]
    assigned = sum(row["inner_fold"] is not None for row in seed_rows)
    if assigned not in (0, len(seed_rows)):
        raise ValueError("Assign inner_fold to every anchor and support, or omit it from all of them")
    return parsed


def decode_image(path, expected_sha256=None, *, image_format=None):
    from PIL import Image, ImageOps
    if isinstance(path, bytes):
        payload = path
        source = "uploaded image"
    else:
        source = path
        with Path(path).open("rb") as handle:
            payload = handle.read(MAX_ENCODED + 1)
    if not payload or len(payload) > MAX_ENCODED:
        raise InvalidImageError(f"Image must contain 1..{MAX_ENCODED} encoded bytes: {source}")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise InvalidImageError(f"Image SHA256 mismatch: {source}")
    try:
        with Image.open(io.BytesIO(payload), formats=None if image_format is None else (image_format,)) as original:
            if original.width * original.height > MAX_PIXELS:
                raise InvalidImageError(f"Image exceeds {MAX_PIXELS} decoded pixels: {source}")
            if image_format is not None and getattr(original, "n_frames", 1) != 1:
                raise InvalidImageError("Upload a single image, without animation")
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.load()
    except (OSError, Image.DecompressionBombError) as error:
        if image_format is None:
            raise
        raise InvalidImageError("The file could not be decoded as the declared image type") from error
    return image, digest


class ImageEncoders:
    """Reuse the same visual encoders across images."""

    def __init__(self, *, device="auto", batch_size=1, mobile_batch_size=32, sd_snapshot=None,
                 mobile_weights=None, precision="auto", allow_download=False, allow_zerogpu_torch=False):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.allow_zerogpu_torch = allow_zerogpu_torch
        self.mobile = MobileEncoder(mobile_weights, device=device, batch_size=mobile_batch_size,
                                    allow_download=allow_download)
        self.sd = SDEncoder(sd_snapshot, device=device, precision=precision, batch_size=batch_size,
                            allow_download=allow_download)

    def contract(self, registrar, geometry):
        differences = ["configurable_unpadded_MobileNet_batches", "user_layout_geometry_and_EXIF_support"]
        if self.sd.precision != "fp16" or str(self.sd.device) == "cpu" or self.batch_size != 1:
            differences.append("SD_execution_differs_from_native_CUDA_FP16_batch1")
        return dict(schema="nrsw.paper_visual_frontend.v1", sd=self.sd.contract(), mobile=self.mobile.contract(),
                    registration=geometry, versions=runtime_versions(allow_zerogpu_torch=self.allow_zerogpu_torch),
                    reference_images=[dict(sha256=r["sha256"], quads={f: q.tolist() for f, q in r["quads"].items()})
                                      for r in registrar.references],
                    deterministic_algorithms=True, cudnn_benchmark=False, tf32=False,
                    historical_cache_byte_parity_claimed=False, execution_differences=differences)

    def close(self):
        self.sd.close()


def describe_rows(rows, registrar, encoders, extractor, *, provenance=None, progress=None,
                  field_classifiers=False, require_alignment=False, image_format=None):
    """Measure already validated rows using the chosen geometry and encoder pair."""
    from .schema import FeatureDataset

    count, fields = len(rows), len(registrar.field_order)
    global_features = np.zeros((count, 512), dtype=np.float32)
    global_mobile = np.zeros((count, 512), dtype=np.float32)
    local_features = np.zeros((count, fields, 512), dtype=np.float32)
    local_valid = np.zeros(count, dtype=bool)
    field_available = pixel_features = pixel_available = None
    if field_classifiers:
        field_available = np.zeros((count, fields), dtype=bool)
        pixel_available = np.zeros((count, fields), dtype=bool)
        pixel_features = np.zeros((count, fields, PIXEL_DIMENSION), dtype=np.float64)
    row_receipts = []
    batch_size = encoders.batch_size
    for start in range(0, count, batch_size):
        images, receipts = [], []
        try:
            for index in range(start, min(count, start + batch_size)):
                row = rows[index]
                image, digest = decode_image(row["image"], row.get("sha256"), image_format=image_format)
                images.append(image)
                try:
                    registration = registrar.register(image, digest)
                except ValueError as error:
                    if not require_alignment:
                        raise
                    raise ImageAlignmentError("The image could not be aligned to this layout") from error
                if require_alignment and not registration.local_valid:
                    raise ImageAlignmentError("The image could not be aligned to this layout")
                global_mobile[index] = encoders.mobile.global_descriptor(image)
                if field_classifiers and registration.quads:
                    gray = np.asarray(image.convert("L"))
                    matrices = registration.sampling_matrices
                    tracked = [(j, field) for j, field in enumerate(registrar.field_order) if field in matrices]
                    if tracked:
                        indices, names = zip(*tracked)
                        crops = [rectify_map(image, matrices[name]) for name in names]
                        local_features[index, list(indices)] = encoders.mobile.encode(crops)
                        field_available[index, list(indices)] = True
                        for j, name in tracked:
                            quad = registration.quads[name]
                            left, top = np.floor(quad.min(axis=0)).astype(int)
                            right, bottom = np.ceil(quad.max(axis=0)).astype(int)
                            descriptor = ink_descriptor(gray[top:bottom, left:right])
                            if descriptor is not None:
                                pixel_available[index, j] = True
                                pixel_features[index, j] = descriptor
                    local_valid[index] = registration.local_valid
                elif registration.local_valid:
                    crops = [rectify_map(image, registration.sampling_matrices[field]) for field in registrar.field_order]
                    local_features[index] = encoders.mobile.encode(crops)
                    local_valid[index] = True
                receipts.append(dict(id=row["id"], image_sha256=digest, local_valid=registration.local_valid,
                                     selected_reference=registration.reference_index, geometry=registration.diagnostics))
                if require_alignment:
                    boxes = {}
                    for name, quad in registration.quads.items():
                        left, top = quad.min(axis=0) / image.size
                        right, bottom = quad.max(axis=0) / image.size
                        boxes[name] = [float(left), float(top), float(right), float(bottom)]
                    receipts[-1]["boxes"] = boxes
            global_features[start:start + len(images)] = encoders.sd.encode(images)
            row_receipts.extend(receipts)
        finally:
            for image in images:
                image.close()
        if progress is not None:
            progress(min(count, start + batch_size), count)
    candidates = np.array([[field in row["candidate_fields"] for field in registrar.field_order] for row in rows], dtype=bool)
    inner_folds = None
    if any(row["inner_fold"] in (0, 1) for row in rows):
        inner_folds = np.array([row["inner_fold"] if row["inner_fold"] is not None else -1 for row in rows], dtype=np.int64)
    metadata = dict(layout_id=registrar.layout_id, field_order=registrar.field_order, extractor=extractor,
                    row_receipts=row_receipts, local_eligible=int(local_valid.sum()),
                    global_fallback=int((~local_valid).sum()), **(provenance or {}))
    return FeatureDataset(global_features=global_features, local_features=local_features,
                          global_mobile=global_mobile, local_valid=local_valid,
                          labels=np.array([r["label"] for r in rows], np.int64), roles=np.array([r["role"] for r in rows]),
                          groups=np.array([r["group"] for r in rows]), ids=np.array([r["id"] for r in rows]),
                          candidate_mask=candidates, metadata=metadata, inner_folds=inner_folds,
                          field_available=field_available, pixel_available=pixel_available,
                          pixel_features=pixel_features)


def extract_manifest(manifest_path, layout_path, output_path, *, device="auto", batch_size=1,
                     mobile_batch_size=32, sd_snapshot=None, mobile_weights=None,
                     precision="auto", allow_download=False, progress=None,
                     registrar_factory=LayoutRegistrar, geometry_contract=None, field_classifiers=False):
    """Extract one registered layout to a validated nrsw.features.v1 NPZ.

    `progress`, when provided, is called with completed and total image counts.
    Heavy encoders are instantiated only here. Network access is opt-in.
    Choose a new output file to avoid replacing saved features.
    """
    from .schema import save_feature_dataset
    output = Path(output_path).resolve()
    sidecar = output.with_suffix(".metadata.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Extraction output already exists: {output} / {sidecar}")
    if output.suffix.lower() != ".npz":
        raise ValueError("Feature output must have a .npz suffix")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    layout_path = Path(layout_path).resolve()
    layout = json.loads(layout_path.read_text(encoding="utf-8-sig"))
    registrar = registrar_factory(layout, layout_path.parent)
    geometry = geometry_contract(registrar) if geometry_contract is not None else registration_contract()
    rows = read_manifest(manifest_path, registrar.field_order, registrar.layout_id)
    reference_hashes = {reference["sha256"] for reference in registrar.references}
    image_assignments = {}
    for row in rows:
        digest = sha256_file(row["image"])
        if row.get("sha256") is not None and row["sha256"] != digest:
            raise ValueError(f"Image SHA256 mismatch: {row['image']}")
        if digest in reference_hashes and row["role"] in ("support", "calibration", "evaluation"):
            raise ValueError("layout reference images must be genuine fitting data; they cannot also be support, calibration or evaluation images")
        assignment = (row["group"], row["role"])
        previous = image_assignments.setdefault(digest, assignment)
        if previous != assignment:
            raise ValueError("Identical encoded image occurs in different groups or roles; merge its dependency component before extraction")
        row["sha256"] = digest
    encoders = ImageEncoders(device=device, batch_size=batch_size, mobile_batch_size=mobile_batch_size,
                            sd_snapshot=sd_snapshot, mobile_weights=mobile_weights, precision=precision,
                            allow_download=allow_download)
    try:
        extractor = encoders.contract(registrar, geometry)
        provenance = dict(manifest_sha256=sha256_file(manifest_path), layout_sha256=sha256_file(layout_path))
        dataset = describe_rows(rows, registrar, encoders, extractor, provenance=provenance, progress=progress,
                                field_classifiers=field_classifiers)
        metadata = dataset.metadata
        output.parent.mkdir(parents=True, exist_ok=True)
        save_feature_dataset(dataset, output)
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return dict(output=str(output), metadata_path=str(sidecar), images=len(dataset),
                    local_eligible=metadata["local_eligible"], global_fallback=metadata["global_fallback"],
                    extractor=extractor)
    finally:
        encoders.close()
