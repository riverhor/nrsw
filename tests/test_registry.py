import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from nrsw import Calibration, LayoutRegistry, UnknownLayoutError, extraction, registry
from nrsw.api import extractor_identity
from nrsw.calibration import calibrate_scores
from nrsw.cli import main
from nrsw.kernel import Layout
from nrsw.vision import AlignedLayoutRegistrar


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def routes(tmp_path, monkeypatch):
    image_type = pytest.importorskip("PIL.Image")
    calls = Counter()

    class MobileStub:
        def __init__(self, *args, **kwargs):
            calls["mobile_instances"] += 1

        def global_descriptor(self, image):
            return np.zeros(512, dtype=np.float32)

        def encode(self, crops):
            return np.asarray([[crop[..., 0].mean()] * 512 for crop in crops], dtype=np.float32)

        def contract(self):
            return {"model": "mobile-test"}

    class SDStub:
        precision = "fp32"
        device = "cpu"

        def __init__(self, *args, **kwargs):
            calls["sd_instances"] += 1

        def encode(self, images):
            calls["sd_forwards"] += 1
            return np.zeros((len(images), 512), dtype=np.float32)

        def contract(self):
            return {"model": "sd-test"}

        def close(self):
            calls["sd_closed"] += 1

    monkeypatch.setattr(extraction, "MobileEncoder", MobileStub)
    monkeypatch.setattr(extraction, "SDEncoder", SDStub)
    encoders = extraction.ImageEncoders(device="cpu")
    source = {"schema": registry.REGISTRY_SCHEMA, "layouts": {}}
    top = [[8, 5], [72, 5], [72, 17], [8, 17]]
    bottom = [[8, 30], [72, 30], [72, 42], [8, 42]]
    models = {}
    for index, (key, quads, threshold) in enumerate((
        ("card-v1", [top, bottom], 30), ("pass-v2", [bottom, top], 190),
    )):
        directory = tmp_path / key
        directory.mkdir()
        layout = dict(layout_id=key, mode="aligned", image_size=[80, 48], field_order=["first", "second"],
                      quads=dict(zip(("first", "second"), quads)))
        write_json(directory / "layout.json", layout)
        registrar = AlignedLayoutRegistrar(layout, directory)
        extractor = encoders.contract(registrar, registrar.contract())
        identity = f"model-{key}"
        lineage = {"ids": [], "groups": [], "image_sha256": []}
        models[key] = SimpleNamespace(
            layout=Layout(key, ("first", "second"), 512, 512, extractor_identity(extractor)),
            training_identity=identity, metadata={"training_lineage": lineage},
            score=lambda batch, offset=index: batch.local_features[:, 0, 0] + offset,
        )
        cut = calibrate_scores(np.full(500, threshold), "0.01")
        write_json(directory / "cuts.json", Calibration(identity, lineage, {"1/100": cut}).to_dict())
        source["layouts"][key] = dict(layout=f"{key}/layout.json", model=f"{key}/model", calibration=f"{key}/cuts.json")
    encoders.close()
    calls.clear()
    monkeypatch.setattr(registry, "load_model", lambda path: models[path.parent.name])
    registry_path = tmp_path / "layouts.json"
    write_json(registry_path, source)
    pixels = np.full((48, 80, 3), 40, dtype=np.uint8)
    pixels[24:, :, 0] = 180
    image = tmp_path / "query.png"
    image_type.fromarray(pixels).save(image)
    return registry_path, image, source, calls


def test_routes_own_geometry_model_and_cut_with_one_encoder_pair(routes, capsys):
    config, image, _, calls = routes
    with LayoutRegistry.from_file(config, device="cpu") as detector:
        assert detector.layout_ids == ("card-v1", "pass-v2")
        assert detector.supports("card-v1") and not detector.supports("legacy")
        with pytest.raises(UnknownLayoutError, match="unknown layout ID 'legacy'"):
            detector.predict_image("legacy", "does-not-exist")
        with pytest.raises(ValueError, match="no threshold"):
            detector.predict_image("card-v1", "does-not-exist", fpr="0.05")
        assert not calls
        a = detector.predict_image("card-v1", image, identifier="new-card")
        b = detector.predict_image("pass-v2", image, identifier="new-pass")
        assert a["layout"] == "card-v1" and a["threshold"]["threshold"] == 30
        assert a["rows"] == [{"id": "new-card", "score": 40.0, "local_available": True, "flagged": True}]
        assert b["layout"] == "pass-v2" and b["threshold"]["threshold"] == 190
        assert b["rows"] == [{"id": "new-pass", "score": 181.0, "local_available": True, "flagged": False}]
        assert calls == {"mobile_instances": 1, "sd_instances": 1, "sd_forwards": 2}
    detector.close()
    assert calls["sd_closed"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        detector.predict_image("card-v1", image)

    common = ["--registry", str(config), "--image", str(image), "--device", "cpu"]
    assert main(["route", "--layout-id", "legacy", *common]) == 2
    assert "unknown layout ID" in capsys.readouterr().err
    assert main(["route", "--layout-id", "pass-v2", "--id", "new-pass", *common]) == 0
    assert json.loads(capsys.readouterr().out) == b


@pytest.mark.parametrize("field,match", [("layout", "match the geometry"), ("calibration", "different fitted model")])
def test_registry_rejects_crossed_artifacts(routes, field, match):
    config, _, source, calls = routes
    source["layouts"]["card-v1"][field] = source["layouts"]["pass-v2"][field]
    write_json(config, source)
    with pytest.raises(ValueError, match=match):
        LayoutRegistry.from_file(config, device="cpu")
    assert not calls


def test_changed_geometry_rejected_before_any_forward(routes):
    config, image, source, calls = routes
    layout_path = config.parent / source["layouts"]["card-v1"]["layout"]
    changed = json.loads(layout_path.read_text(encoding="utf-8"))
    changed["quads"]["first"] = changed["quads"]["second"]
    write_json(layout_path, changed)
    with LayoutRegistry.from_file(config, device="cpu") as detector:
        with pytest.raises(ValueError, match="geometry or encoder settings differ from training"):
            detector.predict_image("card-v1", image)
        assert calls["sd_forwards"] == 0


@pytest.mark.parametrize("threshold", ["30", [30], True, 10**400])
def test_route_rejects_malformed_threshold_without_loading_encoders(routes, capsys, threshold):
    config, image, source, calls = routes
    path = config.parent / source["layouts"]["card-v1"]["calibration"]
    calibration = json.loads(path.read_text(encoding="utf-8"))
    calibration["cuts"]["1/100"]["threshold"] = threshold
    write_json(path, calibration)

    assert main(["route", "--registry", str(config), "--layout-id", "card-v1", "--image", str(image)]) == 2
    assert "calibration threshold must be a finite number" in capsys.readouterr().err
    assert not calls
