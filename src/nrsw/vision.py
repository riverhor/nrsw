"""Extract visual descriptors from a manifest or one document image."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .extraction import extract_manifest
from .registration import (
    MAX_PIXELS,
    LayoutRegistrar,
    RegistrationResult,
    quad_sampling_matrix,
    registration_contract,
    validate_quad,
)


class AlignedLayoutRegistrar:
    """Use fixed field positions on documents already in a common pixel frame."""

    references = ()

    def __init__(self, layout, _base_directory):
        self.layout_id = layout.get("layout_id")
        self.field_order = layout.get("field_order")
        if not isinstance(self.layout_id, str) or not self.layout_id.strip():
            raise ValueError("layout_id must be a nonempty string")
        if (not isinstance(self.field_order, list) or len(self.field_order) < 2
                or not all(isinstance(field, str) and field.strip() for field in self.field_order)
                or len(set(self.field_order)) != len(self.field_order)):
            raise ValueError("field_order must list at least two unique field names")
        size = layout.get("image_size")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int for value in size)
                or size[0] < 3 or size[1] < 2 or size[0] * size[1] > MAX_PIXELS):
            raise ValueError(f"image_size must be [width, height] with at most {MAX_PIXELS:,} pixels")
        self.image_size = tuple(size)
        quads = layout.get("quads")
        if not isinstance(quads, dict) or set(quads) != set(self.field_order):
            raise ValueError("quads must define exactly the fields in field_order")
        if "references" in layout:
            raise ValueError("aligned mode uses image_size and quads; remove references")
        self.quads = {name: validate_quad(quads[name], *size) for name in self.field_order}
        self.sampling = {name: quad_sampling_matrix(quad) for name, quad in self.quads.items()}

    def register(self, image, image_sha256=None):
        if image.size != self.image_size:
            raise ValueError(f"Expected an aligned {self.image_size[0]}x{self.image_size[1]} image "
                             f"after EXIF correction, got {image.width}x{image.height}")
        return RegistrationResult(True, None, None, self.quads, {"status": "local"}, self.sampling)

    def contract(self):
        return {
            "algorithm": "fixed_field_quadrilaterals",
            "image_size": list(self.image_size),
            "quads": {name: quad.tolist() for name, quad in self.quads.items()},
            "coordinate_frame": "EXIF_corrected_RGB_pixel_edges",
            "crop": [224, 224],
            "resample": "Pillow_PERSPECTIVE_BILINEAR_white_border",
        }


def _registrar(layout, base_directory):
    if not isinstance(layout, dict):
        raise ValueError("layout JSON must be an object")
    mode = layout.get("mode", "sift")
    if mode == "aligned":
        return AlignedLayoutRegistrar(layout, base_directory)
    if mode == "sift":
        if "image_size" in layout or "quads" in layout:
            raise ValueError("sift mode places field quadrilaterals inside each reference")
        return LayoutRegistrar(layout, base_directory)
    raise ValueError("layout mode must be 'aligned' or 'sift'")


def _geometry_contract(registrar):
    if isinstance(registrar, AlignedLayoutRegistrar):
        return registrar.contract()
    return registration_contract()


def extract_images(layout_path, output_path, *, manifest_path=None, image_path=None, **encoder_options):
    """Save features for the listed images or one new image."""
    if (manifest_path is None) == (image_path is None):
        raise ValueError("Supply one of manifest_path or image_path")
    options = dict(
        registrar_factory=_registrar,
        geometry_contract=_geometry_contract,
        **encoder_options,
    )
    if manifest_path is not None:
        return extract_manifest(manifest_path, layout_path, output_path, **options)

    layout = json.loads(Path(layout_path).read_text(encoding="utf-8-sig"))
    if not isinstance(layout, dict):
        raise ValueError("layout JSON must be an object")
    row = {
        "image": str(Path(image_path).resolve()),
        "id": "query",
        "group": "query",
        "role": "inference",
        "layout_id": layout.get("layout_id"),
    }
    with TemporaryDirectory(prefix="nrsw-query-") as temporary:
        manifest = Path(temporary) / "query.jsonl"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return extract_manifest(manifest, layout_path, output_path, **options)


def _run_extract(args):
    result = extract_images(
        args.layout, args.output, manifest_path=args.manifest, image_path=args.image,
        field_classifiers=args.field_classifiers, **encoder_options(args),
    )
    return {key: value for key, value in result.items() if key != "extractor"}


def encoder_options(args):
    return dict(device=args.device, precision=args.precision, batch_size=args.batch_size,
                mobile_batch_size=args.mobile_batch_size, sd_snapshot=args.sd_snapshot,
                mobile_weights=args.mobile_weights, allow_download=args.allow_download)


def add_encoder_options(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Stable Diffusion batch size")
    parser.add_argument("--mobile-batch-size", type=int, default=32, help="MobileNet batch size")
    parser.add_argument("--sd-snapshot", type=Path, help="Local Stable Diffusion 1.5 model directory")
    parser.add_argument("--mobile-weights", type=Path, help="Local MobileNetV3-Small weights")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloads of missing encoder weights")


def add_vision_commands(subparsers):
    parser = subparsers.add_parser("extract", help="Extract features from images of one layout")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path, help="JSONL, JSON or CSV image manifest")
    input_group.add_argument("--image", type=Path, help="One unlabelled image")
    parser.add_argument("--layout", required=True, type=Path, help="Layout JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Output feature file (.npz)")
    parser.add_argument("--field-classifiers", action="store_true", help="Extract pixel features for the optional field classifiers")
    add_encoder_options(parser)
    parser.set_defaults(handler=_run_extract)
