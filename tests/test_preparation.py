import json
from pathlib import Path
import pytest

from nrsw.preparation import prepare_manifest


def test_prepare_group_exclusion_determinism_and_relative_paths(tmp_path):
    rows = []
    for group in range(15):
        for label in (0, 1):
            image = tmp_path / f"g{group}-{label}.png"; image.write_bytes(b"fixture")
            rows.append({"id": image.stem, "group": f"g{group}", "layout_id": "test", "label": label,
                         "image": image.name, "candidate_fields": ["name"] if label else []})
    source = tmp_path / "all.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "split" / "manifest.jsonl"
    result = prepare_manifest(source, output, anchors=4, bank=2, support=2, calibration=3)
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["counts"] == {"support": 2, "anchor": 4, "bank": 2, "calibration": 3, "evaluation": 8}
    roles = {}
    for row in selected:
        assert (output.parent / row["image"]).is_file()
        assert roles.setdefault(row["group"], row["role"]) == row["role"]
        assert bool(row.get("candidate_fields")) == (row["role"] == "support")
    before = output.read_bytes()
    source.write_text("\n".join(json.dumps(row) for row in reversed(rows)))
    repeated = output.with_name("repeated.jsonl")
    prepare_manifest(source, repeated, anchors=4, bank=2, support=2, calibration=3)
    assert before == repeated.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_manifest(source, output, anchors=4, bank=2, support=2, calibration=3)
    with pytest.raises(ValueError, match="different files"):
        prepare_manifest(source, source)


def test_feasible_allocation_preserves_mixed_genuine_groups(tmp_path):
    rows = []
    for group, labels in [(f"g{i}", (0, 1)) for i in range(7)] + [("f0", (1,)), ("f1", (1,))]:
        for label in labels:
            image = tmp_path / f"{group}-{label}.png"; image.write_bytes(b"fixture")
            rows.append({"id": image.stem, "group": group, "layout_id": "test", "label": label,
                         "image": image.name, "candidate_fields": ["name"] if label else []})
    source = tmp_path / "all.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "prepared.jsonl"
    prepare_manifest(source, output, anchors=4, bank=2, support=2, calibration=1)
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert {r["group"] for r in selected if r["role"] == "support"} == {"f0", "f1"}
    assert len(selected) == 9
