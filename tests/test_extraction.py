"""Optional image-contract tests; no neural weights or network required."""
import json
from collections import Counter

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")

from nrsw import registration as geometry
from nrsw.encoders import canonical_sd_image, global_mobile_tiles, hash_mobile, mobile_hash_mapping
from nrsw.extraction import extract_manifest, read_manifest

pytestmark = pytest.mark.vision


def test_pixel_edge_perspective_identity_and_rotation():
    pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    image = Image.fromarray(pixels)
    full = [[0, 0], [10, 0], [10, 8], [0, 8]]
    np.testing.assert_array_equal(geometry.rectify_quad(image, full, (10, 8)), pixels)
    crop = [[2, 1], [8, 1], [8, 6], [2, 6]]
    np.testing.assert_array_equal(geometry.rectify_quad(image, crop, (6, 5)), pixels[1:6, 2:8])
    # A center-coordinate 90-degree rotation must be conjugated before sampling.
    centers = np.array([[0., -1., 7.], [1., 0., 0.], [0., 0., 1.]])
    edges = geometry.center_to_edge_homography(centers)
    np.testing.assert_allclose(edges, [[0., -1., 8.], [1., 0., 0.], [0., 0., 1.]])
    rotated = Image.fromarray(np.rot90(pixels, k=-1))
    transported = geometry.project_quad(edges, full)
    np.testing.assert_array_equal(geometry.rectify_quad(rotated, transported, (10, 8)), pixels)


@pytest.mark.parametrize("role,label,candidates", [("calibration", 0, []), ("evaluation", 0, []), ("support", 1, ["a"])])
def test_reference_images_cannot_be_heldout_or_forged_support(tmp_path, role, label, candidates):
    from nrsw.extraction import extract_manifest
    image = Image.new("RGB", (100, 80), (20, 40, 60))
    image.save(tmp_path / "reference.png")
    quads = {"a": [[10,10],[30,10],[30,30],[10,30]], "b": [[40,40],[60,40],[60,60],[40,60]]}
    (tmp_path / "layout.json").write_text(json.dumps({"layout_id": "test", "field_order": ["a", "b"],
        "references": [{"image": "reference.png", "quads": quads}]}))
    (tmp_path / "manifest.json").write_text(json.dumps([{"id": "reference", "group": "reference", "image": "reference.png",
        "role": role, "label": label, "layout_id": "test", "candidate_fields": candidates}]))
    # Rejected before either neural encoder is instantiated.
    with pytest.raises(ValueError, match="reference images must be genuine fitting"):
        extract_manifest(tmp_path / "manifest.json", tmp_path / "layout.json", tmp_path / "out.npz")


def test_layout_requires_json_object():
    with pytest.raises(ValueError, match="must be an object"):
        geometry.LayoutRegistrar([])


def test_native_checkerboard_geometry_projective_and_unrelated():
    cv2.setNumThreads(2)
    base = np.random.default_rng(1848).integers(0, 256, size=(720, 1024), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), .6)
    truth = np.array([[.95, .04, 70.], [-.03, .96, 55.], [.00005, -.00003, 1.]])
    warped = cv2.warpPerspective(base, truth, (1280, 960), flags=cv2.INTER_LINEAR, borderValue=240)
    result = geometry._register(np, cv2, base, warped, [], False, Counter(), lambda: None)
    assert result["registration_qualified"], result["failure_reasons"]
    probes = np.array([(x, y) for y in np.linspace(0, 719, 5) for x in np.linspace(0, 1023, 5)])
    actual, _ = geometry._project(np, np.array(result["qualified_H_raw"]), probes)
    expected, _ = geometry._project(np, truth, probes)
    assert np.linalg.norm(actual-expected, axis=1).max() < 2.5
    unrelated = np.random.default_rng(7841).integers(0, 256, size=base.shape, dtype=np.uint8)
    bad = geometry._register(np, cv2, base, unrelated, [], False, Counter(), lambda: None)
    assert not bad["registration_qualified"]
    reflected = geometry._register(np, cv2, base, base[:, ::-1].copy(), [], False, Counter(), lambda: None)
    assert not reflected["registration_qualified"]


def test_first_global_reference_precedes_observability(tmp_path, monkeypatch):
    image = Image.fromarray(np.full((80, 100, 3), 128, dtype=np.uint8))
    image.save(tmp_path / "reference.png")
    layout = {"layout_id": "test", "field_order": ["a", "b"], "references": [
        {"image": "reference.png", "quads": {"a": [[10,10],[30,10],[30,30],[10,30]],
                                                "b": [[70,40],[90,40],[90,60],[70,60]]}},
        {"image": "reference.png", "quads": {"a": [[10,10],[30,10],[30,30],[10,30]],
                                                "b": [[40,40],[60,40],[60,60],[40,60]]}}]}
    registrar = geometry.LayoutRegistrar(layout, tmp_path)
    calls = []
    def fixed(*args):
        calls.append(1)
        return dict(registration_qualified=True, qualified_H_raw=[[1,0,20],[0,1,0],[0,0,1]],
                    failure_reasons=[], matching={}, partition={})
    monkeypatch.setattr(geometry, "_register", fixed)
    result = registrar.register(image)
    assert result.reference_index == 0 and not result.local_valid
    assert result.diagnostics["unavailable_fields"] == ["b"]
    assert len(calls) == 1


def test_manifest_rejects_inference_candidate_annotations(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (100, 80)).save(image)
    path = tmp_path / "manifest.json"
    row = dict(image="input.png", id="q1", group="g1", role="inference", layout_id="test", candidate_fields=["a"])
    path.write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="only for training support"):
        read_manifest(path, ["a", "b"], "test")
    row.pop("candidate_fields")
    path.write_text(json.dumps([row]), encoding="utf-8")
    assert read_manifest(path, ["a", "b"], "test")[0]["label"] == -1


def test_frozen_mobile_hash_and_preprocess():
    bucket, sign = mobile_hash_mapping()
    assert len(np.unique(bucket)) == 340
    raw = np.arange(2*576, dtype=np.float32).reshape(2, 576) / 4096
    actual = hash_mobile(raw)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.sqrt(2), rtol=1e-6)
    np.testing.assert_array_equal(hash_mobile(np.zeros((1,576), np.float32)), np.zeros((1,512), np.float32))
    image = Image.new("RGB", (1134,716), (40,80,120))
    assert canonical_sd_image(image).size == (512,512)
    assert len(global_mobile_tiles(image)) == 6


def test_duplicate_encoded_image_cannot_cross_components_before_model_load(tmp_path):
    image = Image.new("RGB", (100, 80), (20, 40, 60))
    image.save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    quads = {"a": [[10,10],[30,10],[30,30],[10,30]], "b": [[40,40],[60,40],[60,60],[40,60]]}
    layout = dict(layout_id="test", field_order=["a", "b"], references=[dict(image="a.png", quads=quads)])
    (tmp_path / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    rows = [dict(image=name, id=name, group=name, role="inference", layout_id="test") for name in ("a.png", "b.png")]
    (tmp_path / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Identical encoded image"):
        extract_manifest(tmp_path / "manifest.json", tmp_path / "layout.json", tmp_path / "out.npz")
