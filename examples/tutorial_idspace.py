"""Prepare the image tutorial from the IDSpace Slovak ID images and saved splits."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rank(identifier):
    return hashlib.sha256(f"nrsw-idspace-image-tutorial-v1|{identifier}".encode()).hexdigest()


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def packet(metadata, images, output):
    receipt = read(metadata / "EXPORT_RECEIPT.json")
    for name in ("layout.json", "documents.json", "contexts.json.gz"):
        with (metadata / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != receipt["files"][name]["sha256"]:
            raise ValueError(f"Metadata checksum mismatch for {name}")
    layout = read(metadata / "layout.json")
    fields = layout["field_order"]
    documents = {row["id"]: row for row in read(metadata / "documents.json")["rows"]}
    with gzip.open(metadata / "contexts.json.gz", "rt", encoding="utf-8") as stream:
        contexts = json.load(stream)
    context = next(row for row in contexts["contexts"] if row["id"] == "f0_d3_o0")
    outer = contexts["outer_folds"]["0"]

    def image_path(document):
        path = (images / document["member"]).resolve(strict=True)
        if not path.is_relative_to(images) or path.stat().st_size != document["bytes"]:
            raise ValueError(f"Unexpected image path or size for record {document['id']}")
        return str(path)

    def row(identifier, role):
        document = documents[identifier]
        return dict(id=identifier, image=image_path(document), group=document["group"],
                    layout_id=layout["layout_id"], role=role, label=document["label"],
                    sha256=document["sha256"])

    train_roles = {"N": "anchor", "BANK": "bank", "F": "support"}
    rows = []
    for entry in context["train"]:
        record = row(entry["id"], train_roles[entry["role"]])
        if entry["role"] in {"N", "F"}:
            record["inner_fold"] = entry["fold"]
        if entry["role"] == "F":
            record["candidate_fields"] = [fields[index] for index in entry["candidates"]]
        rows.append(record)

    used = {record["group"] for record in rows}
    if len(used) != len(rows):
        raise ValueError("The training images must come from distinct source groups")

    def choose(identifiers, count, label, role):
        chosen = []
        for identifier in sorted(identifiers, key=rank):
            document = documents[identifier]
            if document["label"] == label and document["group"] not in used:
                chosen.append(row(identifier, role))
                used.add(document["group"])
                if len(chosen) == count:
                    return chosen
        raise ValueError(f"Insufficient distinct groups for {role} label {label}")

    rows += choose(outer["calibration_ids"], 64, 0, "calibration")
    for label in (0, 1):
        rows += choose(outer["test_ids"], 48, label, "evaluation")
    query, = choose(outer["test_ids"], 1, 1, "inference")
    query["label"] = -1

    anchors = {record["sha256"]: record for record in rows if record["role"] == "anchor"}
    references = []
    for reference in layout["references"]:
        quads = {}
        for window in reference["windows"]:
            left, top, right, bottom = window["TRAIN_union_window_edges"]
            quads[window["field_name"]] = [[left, top], [right, top], [right, bottom], [left, bottom]]
        references.append(dict(image=anchors[reference["sha256"]]["image"], quads=quads))

    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "roles.jsonl", rows)
    write_rows(output / "query.jsonl", [query])
    with (output / "layout.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(layout_id=layout["layout_id"], field_order=fields, references=references), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="folder containing the saved IDSpace splits and layout")
    parser.add_argument("--images", type=Path, required=True, help="folder containing the extracted Slovak ID images")
    parser.add_argument("--output", type=Path, required=True, help="new tutorial input directory")
    args = parser.parse_args()
    packet(args.metadata, args.images.resolve(), args.output)
