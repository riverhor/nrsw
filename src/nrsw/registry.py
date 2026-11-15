"""Route known document layouts to their geometry and calibrated detector."""

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock

from .api import Calibration, extractor_identity, load_model, predict
from .encoders import sha256_file
from .extraction import ImageEncoders, describe_rows
from .field_classifiers import FieldClassifierModel
from .kernel import JointModel
from .registration import MAX_ENCODED, LayoutRegistrar
from .vision import AlignedLayoutRegistrar, _geometry_contract, _registrar

REGISTRY_SCHEMA = "nrsw.layout_registry.v1"


class UnknownLayoutError(ValueError):
    """No detector is configured for the supplied layout ID."""


@dataclass(frozen=True)
class _Route:
    registrar: LayoutRegistrar | AlignedLayoutRegistrar
    model: JointModel | FieldClassifierModel
    calibration: Calibration


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate keys in layout registry")
    return result


def _read_json(path, **options):
    return json.loads(path.read_text(encoding="utf-8-sig"), **options)


def _load_route(layout_id, spec, directory):
    fields = ("layout", "model", "calibration")
    if (not isinstance(layout_id, str) or not layout_id.strip()
            or not isinstance(spec, dict) or set(spec) != set(fields)
            or any(not isinstance(spec[name], str) or not spec[name].strip() for name in fields)):
        raise ValueError(f"registry entry {layout_id!r} needs layout, model and calibration paths")
    layout_path, model_path, calibration_path = (directory / spec[name] for name in fields)
    registrar = _registrar(_read_json(layout_path), layout_path.parent)
    model = load_model(model_path)
    calibration = Calibration.from_dict(_read_json(calibration_path))
    if layout_id != registrar.layout_id or layout_id != model.layout.layout_id:
        raise ValueError(f"registry key {layout_id!r} must match the geometry and fitted model layout IDs")
    if tuple(registrar.field_order) != model.layout.fields or (model.layout.dg, model.layout.d) != (512, 512):
        raise ValueError(f"registry entry {layout_id!r} geometry or feature dimensions differ from the fitted model")
    if calibration.training_identity != model.training_identity:
        raise ValueError(f"registry entry {layout_id!r} calibration belongs to a different fitted model")
    return _Route(registrar, model, calibration)


class LayoutRegistry:
    """Predict from images with caller-supplied layout IDs and one shared encoder pair."""

    def __init__(self, routes, encoder_options):
        self._routes = routes
        self._encoder_options = encoder_options
        self._encoders = None
        self._closed = False
        self._lock = Lock()

    @classmethod
    def from_file(cls, path, **encoder_options):
        """Load the geometry, fitted model and own calibration for each layout."""
        path = Path(path).resolve()
        data = _read_json(path, object_pairs_hook=_unique_object)
        if (not isinstance(data, dict) or set(data) != {"schema", "layouts"}
                or data["schema"] != REGISTRY_SCHEMA or not isinstance(data["layouts"], dict) or not data["layouts"]):
            raise ValueError(f"expected a {REGISTRY_SCHEMA} object with at least one layout")
        routes = {layout_id: _load_route(layout_id, spec, path.parent)
                  for layout_id, spec in data["layouts"].items()}
        return cls(routes, encoder_options)

    @property
    def layout_ids(self):
        """Configured IDs in registry order."""
        return tuple(self._routes)

    def supports(self, layout_id):
        """Tell a host whether it should send this layout to NRSW."""
        return isinstance(layout_id, str) and layout_id in self._routes

    def predict_image(self, layout_id, image_path, *, fpr="0.01", identifier=None, group=None):
        """Score one unlabelled image using the named layout's saved cut."""
        if not self.supports(layout_id):
            known = ", ".join(self.layout_ids)
            raise UnknownLayoutError(f"unknown layout ID {layout_id!r}; configured IDs: {known}")
        if any(value is not None and (not isinstance(value, str) or not value.strip())
               for value in (identifier, group)):
            raise ValueError("identifier and group must be nonempty strings when supplied")

        with self._lock:
            if self._closed:
                raise RuntimeError("layout registry is closed")
            route = self._routes[layout_id]
            route.calibration.cut(fpr)
            image = Path(image_path).resolve()
            if not 1 <= image.stat().st_size <= MAX_ENCODED:
                raise ValueError(f"Image must contain 1..{MAX_ENCODED} encoded bytes: {image}")
            digest = sha256_file(image)
            identifier = identifier if identifier is not None else f"query:{digest}"
            row = dict(image=image, id=identifier, group=identifier if group is None else group,
                       sha256=digest, label=-1, role="inference", candidate_fields=[], inner_fold=None)
            if self._encoders is None:
                self._encoders = ImageEncoders(**self._encoder_options)
            extractor = self._encoders.contract(route.registrar, _geometry_contract(route.registrar))
            if extractor_identity(extractor) != route.model.layout.extractor_id:
                raise ValueError(f"layout {layout_id!r} geometry or encoder settings differ from training")
            if isinstance(route.model, FieldClassifierModel):
                dataset = describe_rows([row], route.registrar, self._encoders, extractor, field_classifiers=True)
            else:
                dataset = describe_rows([row], route.registrar, self._encoders, extractor)
            return predict(route.model, dataset, route.calibration, fpr)

    def close(self):
        """Release the encoder pair and prevent further predictions."""
        with self._lock:
            self._closed = True
            encoders, self._encoders = self._encoders, None
            if encoders is not None:
                encoders.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
