"""Small synthetic documents for trying the detector without model weights."""
from pathlib import Path
import json
import shutil

import numpy as np

from .api import Calibration, _select, calibrate, evaluate, fit, save_model, score_rows
from .schema import FeatureDataset, save_feature_dataset

WEB = Path(__file__).with_name("web")
FIELDS = ("name", "number", "valid_until")
SAMPLE_NAMES = {
    "genuine": ("Unaltered card", "All three fields come from the same synthetic reference layout.", None),
    "number-edit": ("Edited card number", "The number descriptor was altered in this example.", "number"),
    "date-edit": ("Edited expiry date", "The expiry descriptor was altered in this example.", "valid_until"),
    "name-edit": ("Edited name", "The name descriptor was altered in this example.", "name"),
    "missing-fields": ("Fields unavailable", "Only the global descriptors are available for this card.", None),
}


def _descriptors(rng, edits):
    n = len(edits)
    layout = np.linspace(-0.6, 0.6, 18).reshape(1, 3, 6)
    common = rng.normal(0, 0.18, (n, 1, 6))
    fields = layout + common + rng.normal(0, 0.13, (n, 3, 6))
    for row, field in enumerate(edits):
        if field >= 0:
            fields[row, field, 0] += rng.normal(1.7, 0.28)
            fields[row, field, 1] -= rng.normal(1.15, 0.19)
    sd = rng.normal(0, 0.28, (n, 4))
    mobile = common[:, 0] + rng.normal(0, 0.14, (n, 6))
    return sd, fields, mobile


def _packet(rng, ids, roles, edits, *, missing=None):
    sd, fields, mobile = _descriptors(rng, edits)
    valid = np.array([identifier != missing for identifier in ids], dtype=bool)
    fields[~valid] = np.nan
    tags = np.zeros((len(ids), len(FIELDS)), dtype=bool)
    for i, role in enumerate(roles):
        if role == "support":
            tags[i, edits[i]] = True
    metadata = {"layout_id": "specimen-card", "field_order": list(FIELDS),
                "extractor": {"name": "nrsw synthetic descriptors", "version": 1}}
    return FeatureDataset(sd, fields, mobile, valid, (np.asarray(edits) >= 0).astype(int),
                          np.asarray(roles), np.asarray(ids), np.asarray(ids), tags, metadata)


def example_data():
    rng = np.random.default_rng(146)
    roles = ["anchor"] * 24 + ["bank"] * 72 + ["support"] * 15
    training = _packet(rng, [f"train-{i:03d}" for i in range(len(roles))], roles,
                       [-1] * 96 + [i % 3 for i in range(15)])
    calibration = _packet(rng, [f"cal-{i:03d}" for i in range(320)], ["calibration"] * 320, [-1] * 320)
    ids = list(SAMPLE_NAMES) + [f"test-{i:03d}" for i in range(160)]
    edits = [-1, 1, 2, 0, -1] + [-1] * 80 + [i % 3 for i in range(80)]
    test = _packet(rng, ids, ["evaluation"] * len(ids), edits, missing="missing-fields")
    return training, calibration, test


def trace(model, dataset, calibration: Calibration | None, *, specimen=False):
    query = _select(dataset, ("inference", "evaluation"))
    batch, scores = score_rows(model, query, calibration)
    cuts = [] if calibration is None else sorted(calibration.cuts.values(), key=lambda cut: cut.eta)
    thresholds = [{"fpr": cut.eta, "threshold": cut.threshold} for cut in cuts if cut.available]
    borderline = None
    if specimen:
        lo = min(point["threshold"] for point in thresholds)
        hi = max(point["threshold"] for point in thresholds)
        near = np.flatnonzero((query.labels == 0) & (scores > lo) & (scores <= hi))
        if near.size:
            borderline = batch.ids[near[0]]
        selected = np.isin(query.ids, list(SAMPLE_NAMES)) | (query.ids == borderline)
        batch, scores = batch.subset(np.flatnonzero(selected)), scores[selected]
    teacher = getattr(model, "document", model).teachers[2]
    values = teacher.evidence.evaluate(batch.local_features[batch.local_valid])
    local_rows = iter(zip(values["alpha"], values["energy"]))
    examples = []
    for identifier, score, valid in zip(batch.ids, scores, batch.local_valid):
        weights, energies = next(local_rows) if valid else (None, None)
        if identifier == borderline:
            identifier, title, note, edited = ("borderline", "Genuine card near the threshold",
                                               "This genuine example changes decision when you select a different false-positive budget.", None)
        else:
            title, note, edited = SAMPLE_NAMES[identifier] if specimen else (identifier, "", None)
        examples.append({"id": identifier, "title": title, "note": note,
                         "edited_fields": [edited] if edited else [], "score": float(score),
                         "weights": weights.astype(float).tolist() if valid else None,
                         "energies": energies.tolist() if valid else None})
    counts = model.metadata["role_counts"]
    return {"dataset": "synthetic" if specimen else "custom", "layout": model.layout.layout_id,
            "fields": list(model.layout.fields), "examples": examples, "thresholds": thresholds,
            "training": {"genuine": counts["N"] + counts["BANK"], "forged": counts["F"],
                         "calibration": cuts[0].n if cuts else 0}}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def browser_data(data):
    return "window.NRSW_EXAMPLES = " + json.dumps(data, ensure_ascii=True, allow_nan=False) + ";\n"


def example_run():
    training, genuine, test = example_data()
    model = fit(training)
    cuts = calibrate(model, genuine)
    return training, genuine, test, model, cuts


def run_demo(output):
    training, genuine, test, model, cuts = example_run()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, packet in (("train", training), ("calibration", genuine), ("test", test)):
        save_feature_dataset(packet, output / f"{name}.npz")
    save_model(model, output / "model")
    write_json(output / "cuts.json", cuts.to_dict())
    report = {fpr: evaluate(model, test, cuts, fpr) for fpr in ("0.01", "0.05", "0.10")}
    write_json(output / "evaluation.json", report)
    write_json(output / "trace.json", trace(model, test, cuts, specimen=True))
    return {"directory": str(output), "evaluation": report}


def export_demo(output):
    _, _, test, model, cuts = example_run()
    destination = Path(output)
    shutil.copytree(WEB, destination, ignore=shutil.ignore_patterns("test*", "__pycache__"))
    data = trace(model, test, cuts, specimen=True)
    write_json(destination / "examples.json", data)
    (destination / "examples.js").write_text(browser_data(data), encoding="utf-8")
    (destination / ".nojekyll").touch()
    return {"directory": str(destination)}
