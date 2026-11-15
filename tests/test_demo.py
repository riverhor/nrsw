from collections import defaultdict
from hashlib import sha256
import json

import pytest

from nrsw.demo import export_demo


def test_export_keeps_the_scored_documents_and_their_predictions(tmp_path):
    site = tmp_path / "site"
    export_demo(site)
    specimens = site / "specimens"
    data = json.loads((specimens / "examples.json").read_text(encoding="utf-8"))
    script = (specimens / "examples.js").read_text(encoding="utf-8")
    assert json.loads(script.removeprefix("window.NRSW_SPECIMENS = ").rstrip(";\n")) == data
    assert json.loads((site / "examples.json").read_text(encoding="utf-8"))["dataset"] == "synthetic"

    fields = set(data["fields"])
    pairs = defaultdict(set)
    assert [point["fpr"] for point in data["thresholds"]] == [0.01, 0.05, 0.10]
    for example in data["examples"]:
        image = site / example["image"]
        assert image.parent == specimens
        assert image.stem == example["id"]
        assert sha256(image.read_bytes()).hexdigest() == example["image_sha256"]
        assert set(example["boxes"]) == fields
        assert len(example["weights"]) == len(example["energies"]) == len(fields)
        assert sum(example["weights"]) == pytest.approx(1)
        assert set(example["edited_fields"]) <= fields
        assert bool(example["edited_fields"]) == (example["label"] == "forged")
        pairs[example["pair_id"]].add(example["label"])
    assert len(data["examples"]) == 6
    assert len(pairs) == 3
    assert all(labels == {"genuine", "forged"} for labels in pairs.values())
