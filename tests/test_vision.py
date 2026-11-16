import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nrsw import encoders, extraction
from nrsw.schema import load_feature_dataset
from nrsw.vision import AlignedLayoutRegistrar, add_vision_commands, extract_images


@pytest.mark.parametrize("release,revision,cuda,accepted", [
    ("2.13.0", "cf30153c4c131c8164ee7798e5022d810682e2cb", "13.0", True),
    ("2.13.0+cu130", "cf30153c4c131c8164ee7798e5022d810682e2cb", "13.0", True),
    ("2.13.0+cpu", "cf30153c4c131c8164ee7798e5022d810682e2cb", None, False),
    ("2.13.0", "cf30153c4c131c8164ee7798e5022d810682e2cb", "12.8", False),
    ("2.13.0", "other-build", "13.0", False),
    ("2.13.1", "cf30153c4c131c8164ee7798e5022d810682e2cb", "13.0", False),
])
def test_zerogpu_torch_alias_is_strict_and_opt_in(monkeypatch, release, revision, cuda, accepted):
    versions = {"torch": "2.13.0", "torchvision": "0.28.0+cu130"}
    monkeypatch.setattr(encoders.importlib.metadata, "version", lambda package: versions.get(package, "1"))
    module = SimpleNamespace(__version__=release, version=SimpleNamespace(git_version=revision, cuda=cuda))
    monkeypatch.setitem(sys.modules, "torch", module)
    default = encoders.runtime_versions()
    assert default["torch"] == "2.13.0"
    if accepted:
        assert encoders.runtime_versions(allow_zerogpu_torch=True) == {**default, "torch": "2.13.0+cu130"}
    else:
        with pytest.raises(ValueError, match="verified PyTorch 2.13.0 CUDA 13.0 build"):
            encoders.runtime_versions(allow_zerogpu_torch=True)


def layout_config():
    return {
        "layout_id": "card-v1",
        "mode": "aligned",
        "image_size": [80, 48],
        "field_order": ["top", "bottom"],
        "quads": {
            "top": [[8, 5], [72, 5], [72, 17], [8, 17]],
            "bottom": [[8, 30], [72, 30], [72, 42], [8, 42]],
        },
    }


def test_aligned_extraction_and_single_query_share_feature_contract(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")

    class MobileStub:
        def __init__(self, *args, **kwargs):
            pass

        def global_descriptor(self, image):
            return np.full(512, np.asarray(image)[..., 0].mean(), dtype=np.float32)

        def encode(self, crops):
            return np.asarray([[crop[..., 0].mean()] * 512 for crop in crops], dtype=np.float32)

        def contract(self):
            return {"model": "mobile-test-stub"}

    class SDStub:
        precision = "fp32"
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, images):
            return np.ones((len(images), 512), dtype=np.float32)

        def contract(self):
            return {"model": "sd-test-stub"}

        def close(self):
            pass

    monkeypatch.setattr(extraction, "MobileEncoder", MobileStub)
    monkeypatch.setattr(extraction, "SDEncoder", SDStub)
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps(layout_config()), encoding="utf-8")
    pixels = np.zeros((48, 80, 3), dtype=np.uint8)
    pixels[:24, :, 0] = 40
    pixels[24:, :, 0] = 180
    image = tmp_path / "image.png"
    Image.fromarray(pixels).save(image)
    manifest = tmp_path / "images.jsonl"
    manifest.write_text(json.dumps({
        "image": image.name, "id": "support-1", "group": "document-1", "layout_id": "card-v1",
        "role": "support", "label": 1, "candidate_fields": ["bottom"], "inner_fold": 1,
    }) + "\n", encoding="utf-8")
    training = tmp_path / "training.npz"
    extract_images(layout, training, manifest_path=manifest, device="cpu")

    cli = argparse.ArgumentParser()
    add_vision_commands(cli.add_subparsers(required=True))
    query = tmp_path / "query.npz"
    args = cli.parse_args(["extract", "--image", str(image), "--layout", str(layout),
                           "--output", str(query), "--device", "cpu"])
    assert args.handler(args)["local_eligible"] == 1
    trained = load_feature_dataset(training)
    predicted = load_feature_dataset(query)
    assert trained.contract() == predicted.contract()
    assert trained.local_valid.tolist() == [True]
    assert trained.candidate_mask.tolist() == [[False, True]]
    assert trained.inner_folds.tolist() == [1]
    assert predicted.labels.tolist() == [-1]
    assert predicted.candidate_mask.tolist() == [[False, False]]
    assert predicted.inner_folds is None
    assert predicted.local_features[0, :, 0].tolist() == [40., 180.]
    assert predicted.metadata["extractor"]["registration"]["quads"] == layout_config()["quads"]
    assert query.with_suffix(".metadata.json").is_file()


def test_aligned_layout_rejects_wrong_frame(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    registrar = AlignedLayoutRegistrar(layout_config(), tmp_path)
    with Image.new("RGB", (79, 48)) as image:
        with pytest.raises(ValueError, match="Expected an aligned 80x48 image"):
            registrar.register(image)


def test_aligned_layout_requires_each_named_field(tmp_path):
    layout = layout_config()
    layout["quads"].pop("top")
    with pytest.raises(ValueError, match="quads must define exactly the fields"):
        AlignedLayoutRegistrar(layout, tmp_path)


def test_manifest_requires_all_or_no_seed_folds(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    base = {"image": image.name, "layout_id": "card-v1", "role": "anchor", "label": 0}
    first = dict(base, id="a", group="a", inner_fold=0)
    second = dict(base, id="b", group="b")
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps([first, second]), encoding="utf-8")
    with pytest.raises(ValueError, match="every anchor and support"):
        extraction.read_manifest(manifest, ["top", "bottom"], "card-v1")
    second.update(role="calibration", inner_fold=1)
    manifest.write_text(json.dumps([first, second]), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration inner_fold"):
        extraction.read_manifest(manifest, ["top", "bottom"], "card-v1")
