from dataclasses import replace
import json

import numpy as np
import pytest

from nrsw import calibrate, evaluate, fit, load_model, predict, save_feature_dataset, save_model
from nrsw.cli import main
from nrsw.demo import example_data, example_run, trace


def test_fitted_model_round_trip(tmp_path):
    training, _, test, model, cuts = example_run()
    save_model(model, tmp_path / "model")
    restored = load_model(tmp_path / "model")
    assert model.layout.width == 2 * 4 + (3 + 2) * 6 + 1
    assert predict(restored, test, cuts) == predict(model, test, cuts)
    assert set(training.ids).isdisjoint(cuts.lineage["ids"])
    assert evaluate(restored, test, cuts)["FNR_groups"] > 0


def test_split_and_layout_checks():
    training, genuine, test, model, cuts = example_run()
    genuine.ids = genuine.ids.astype("U32")
    genuine.ids[0] = training.ids[0]
    with pytest.raises(ValueError, match="training/calibration overlap"):
        calibrate(model, genuine)
    with pytest.raises(ValueError, match="different fitted model"):
        predict(model, test, replace(cuts, training_identity="different"))
    test.metadata["field_order"].reverse()
    with pytest.raises(ValueError, match="semantic order"):
        predict(model, test, cuts)


def test_thresholds_and_missing_fields():
    _, genuine, test, model, cuts = example_run()
    data = trace(model, test, cuts, specimen=True)
    rows = {row["id"]: row for row in data["examples"]}
    assert rows["missing-fields"]["weights"] is None
    assert rows["number-edit"]["weights"][1] == max(rows["number-edit"]["weights"])
    score = rows["borderline"]["score"]
    assert cuts.cut("0.10").threshold < score <= cuts.cut("0.01").threshold
    small = calibrate(model, genuine.subset(np.arange(5)))
    with pytest.raises(ValueError, match="no finite threshold"):
        predict(model, test, small)


def test_feature_fit_ignores_held_rows():
    training, genuine, _ = example_data()
    fields = ("global_features", "local_features", "global_mobile", "local_valid", "labels",
              "roles", "groups", "ids", "candidate_mask")
    combined = replace(training, **{name: np.concatenate((getattr(training, name), getattr(genuine, name)))
                                   for name in fields})
    assert fit(training).training_identity == fit(combined).training_identity


def test_cli_demo_and_static_export(tmp_path):
    demo, site = tmp_path / "demo", tmp_path / "site"
    assert main(["demo", "--output", str(demo)]) == 0
    assert main(["export-demo", "--output", str(site)]) == 0
    assert json.loads((demo / "trace.json").read_text()) == json.loads((site / "examples.json").read_text())
    assert all((site / asset).is_file() for asset in ("index.html", "styles.css", "demo.js", "examples.js"))


@pytest.mark.parametrize("extension", [False, True])
def test_image_command_workflow(tmp_path, monkeypatch, extension):
    from nrsw import vision
    from nrsw.schema import ARRAY_NAMES

    training, genuine, test = example_data()
    rng = np.random.default_rng(772)

    def add_pixel_fields(packet):
        shape = packet.candidate_mask.shape
        return replace(packet, pixel_features=rng.random((*shape, 7)),
                       pixel_available=np.ones(shape, bool),
                       field_available=np.broadcast_to(packet.local_valid[:, None], shape).copy())

    combined = replace(training, **{name: np.concatenate([getattr(d, name) for d in (training, genuine, test)])
                                    for name in ARRAY_NAMES})
    combined, query = add_pixel_fields(combined), add_pixel_fields(test.subset([0]))
    query = replace(query, roles=np.array(["inference"]), labels=np.array([-1]),
                    ids=np.array(["new-submission"]), groups=np.array(["new-group"]))
    modes = []

    def extract(_layout, output, *, manifest_path=None, image_path=None, field_classifiers=False, **_):
        modes.append(field_classifiers)
        packet = combined if manifest_path is not None else query
        if not field_classifiers:
            packet = replace(packet, pixel_features=None, pixel_available=None, field_available=None)
        save_feature_dataset(packet, output)

    monkeypatch.setattr(vision, "extract_images", extract)
    run = tmp_path / "run"
    options = ["--field-classifiers"] if extension else []
    assert main(["fit-images", "--manifest", "roles.jsonl", "--layout", "layout.json",
                 "--output", str(run), "--fpr", "0.10", *options]) == 0
    assert json.loads((run / "evaluation.json").read_text())["1/10"]["calibration"]["eta"] == 0.10
    output = run / "inference.json"
    assert main(["predict-image", "--image", "query.png", "--layout", "layout.json", "--model", str(run / "model"),
                 "--calibration", str(run / "cuts.json"), "--fpr", "0.10", "--output", str(output)]) == 0
    assert modes == [extension, extension]
    assert json.loads(output.read_text())["rows"][0]["id"] == "new-submission"
