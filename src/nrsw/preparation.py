"""Component-disjoint manifest preparation for a newly registered layout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def prepare_manifest(input_path, output_path, *, anchors=50, bank=225, support=15,
                     calibration=500, seed=2026):
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source == destination:
        raise ValueError("input and prepared output must be different files")
    if destination.exists():
        raise FileExistsError(f"manifest already exists: {destination}")
    counts = {"support": support, "anchor": anchors, "bank": bank, "calibration": calibration}
    for role, count in counts.items():
        if type(count) is not int or count < {"anchor": 4, "bank": 2, "support": 2, "calibration": 1}[role]:
            raise ValueError(f"invalid requested {role} count")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input JSONL is empty")
    ids, paths, groups, layouts = set(), set(), {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each input line must be an image record")
        for name in ("id", "group", "layout_id", "image"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"input records require nonempty {name}")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("prepare requires binary integer labels 0=genuine, 1=forged")
        if row["id"] in ids:
            raise ValueError("duplicate image id in input manifest")
        image = (source.parent / row["image"]).resolve()
        if image in paths:
            raise ValueError("duplicate image path in input manifest")
        if not image.is_file():
            raise FileNotFoundError(f"image not found: {image}")
        candidates = row.get("candidate_fields", [])
        if not isinstance(candidates, list) or not all(isinstance(c, str) and c for c in candidates) or len(set(candidates)) != len(candidates):
            raise ValueError("candidate_fields must be a list of unique field names")
        if candidates and (row["label"] != 1 or len(candidates) > 2):
            raise ValueError("candidate_fields must name one or two fields on a forged document")
        row["image"] = str(image)
        ids.add(row["id"]); paths.add(image); layouts.add(row["layout_id"])
        groups.setdefault(row["group"], []).append(row)
    if len(layouts) != 1:
        raise ValueError("prepare one registered layout at a time")

    def order(value):
        return hashlib.sha256(f"nrsw-prepare-v1|{seed}|{value}".encode()).hexdigest()

    support_groups = sorted((g for g, items in groups.items() if any(
        r["label"] == 1 and r.get("candidate_fields") for r in items)), key=order)
    if len(support_groups) < support:
        raise ValueError(f"need {support} different components with annotated forged images; found {len(support_groups)}")
    required = anchors + bank + calibration
    has_genuine = {g for g, items in groups.items() if any(r["label"] == 0 for r in items)}
    if len(has_genuine) < required:
        raise ValueError(f"need {required} separate genuine components; found {len(has_genuine)}")
    # Preserve enough genuine components even when genuine and forged views
    # share a group. Hash order remains the selection order within this budget.
    mixed_capacity = len(has_genuine) - required
    chosen_support, mixed_used = [], 0
    for group in support_groups:
        if group in has_genuine and mixed_used >= mixed_capacity:
            continue
        chosen_support.append(group)
        mixed_used += group in has_genuine
        if len(chosen_support) == support:
            break
    if len(chosen_support) != support:
        raise ValueError("not enough independent components for the requested genuine and forged role budgets")
    used, selected = set(chosen_support), []
    for group in chosen_support:
        options = [r for r in groups[group] if r["label"] == 1 and r.get("candidate_fields")]
        selected.append(dict(min(options, key=lambda r: order(r["id"])), role="support"))
    genuine_groups = sorted((g for g, items in groups.items() if g not in used and any(
        r["label"] == 0 for r in items)), key=order)
    if len(genuine_groups) < required:
        raise ValueError(f"need {required} separate genuine components after support selection; found {len(genuine_groups)}")
    start = 0
    for role in ("anchor", "bank", "calibration"):
        chosen = genuine_groups[start:start + counts[role]]; start += counts[role]
        used.update(chosen)
        for group in chosen:
            options = [r for r in groups[group] if r["label"] == 0]
            selected.append(dict(min(options, key=lambda r: order(r["id"])), role=role))
    for group in sorted(set(groups) - used, key=order):
        selected.extend(dict(row, role="evaluation") for row in sorted(groups[group], key=lambda r: order(r["id"])))
    for row in selected:
        if row["role"] != "support":
            row.pop("candidate_fields", None)
        try:
            row["image"] = Path(os.path.relpath(row["image"], destination.parent)).as_posix()
        except ValueError:  # Different Windows drive: keep the absolute path.
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return {"output": str(destination), "layout_id": next(iter(layouts)), "seed": seed,
            "counts": {role: sum(r["role"] == role for r in selected) for role in (*counts, "evaluation")},
            "input_images": len(rows), "selected_images": len(selected),
            "unused_views_in_selected_components": len(rows) - len(selected),
            "protocol": "one training/calibration representative per component; remaining components held out"}
