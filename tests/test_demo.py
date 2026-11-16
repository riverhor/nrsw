from collections import Counter, defaultdict
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
    alterations = Counter()
    same_tables = set()
    assert [point["fpr"] for point in data["thresholds"]] == [0.01, 0.05, 0.10]
    examples = {example["id"]: example for example in data["examples"]}
    assert len(examples) == len(data["examples"])
    assert examples[data["selected_id"]]["label"] == "genuine"
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
        assert example["source_member"].split("/", 1)[0] == example["formation"]
        if example["same_jpeg_tables"]:
            same_tables.add(example["pair_id"])
        if example["label"] == "forged":
            field, = example["edited_fields"]
            alterations[example["formation"], field] += 1
        pairs[example["pair_id"]].add(example["label"])
    assert len(pairs) == 100
    assert len(examples) == 2 * len(pairs)
    assert all(labels == {"genuine", "forged"} for labels in pairs.values())
    assert len(same_tables) == 24
    assert alterations == {
        (method, field): 5 for method in ("fraud_crop_and_replace", "fraud_inpaint_and_rewrite")
        for field in fields - {"face"}
    }

    evaluation = data["evaluation"]
    assert (evaluation["genuine"], evaluation["forged"], evaluation["groups"]) == (1398, 2708, 353)
    assert [method["id"] for method in evaluation["methods"]] == ["nrsw", "diagonal", "genuine", "concat"]
    nrsw, diagonal, *_ = evaluation["methods"]
    for method in evaluation["methods"]:
        assert [rate["target"] for rate in method["rates"]] == [0.01, 0.05, 0.10]
        assert all(0 <= rate[metric] <= 1 for rate in method["rates"] for metric in ("fnr", "fpr", "roc_fnr"))
    assert [100 * (d["roc_fnr"] - n["roc_fnr"]) for n, d in zip(nrsw["rates"], diagonal["rates"], strict=True)] == (
        pytest.approx([9.45, 4.50, 3.13], abs=0.01)
    )
    assert evaluation["study"]["images"] == 16210
    assert evaluation["study"]["common_jpeg"] == pytest.approx([17.13, 10.53, 7.55], abs=0.01)
