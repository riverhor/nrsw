"""Convert saved IDSpace measurements to the feature format used by NRSW."""

import argparse
from pathlib import Path
import sys

import numpy as np
from nrsw import FeatureDataset, save_feature_dataset


def convert(companion, metadata, measurements, output):
    # Import the readers from their unpacked folder to keep `nrsw` from this installation.
    sys.path.insert(0, str(companion.resolve(strict=True)))
    from r29_repro.cli import context, read
    from r29_repro.measure import load_batch

    if output.exists():
        raise FileExistsError(output)
    _, case = context(metadata, "f0_d3_o0")
    documents = {row["id"]: row for row in read(metadata / "documents.json")["rows"]}
    training = {row["id"]: row for row in case["train"]}
    training_roles = {"N": "anchor", "F": "support", "BANK": "bank"}
    held_roles = {"calibration": "calibration", "test": "evaluation"}
    arrays = []
    records = []
    for phase in ("train", "calibration", "test"):
        batch, contract = load_batch(measurements, metadata, case, phase)
        arrays.append((batch.global_features, batch.local_features, batch.global_mobile, batch.local_valid))
        for identifier in batch.ids:
            document = documents[identifier]
            entry = training.get(identifier)
            if phase == "train":
                role = training_roles[entry["role"]]
                fold = entry["fold"]
                candidates = entry["candidates"]
            else:
                role = held_roles[phase]
                fold = -1
                candidates = ()
            records.append((identifier, document["group"], document["label"], role, fold,
                            [index in candidates for index in range(len(contract["field_order"]))],
                            document["sha256"]))

    global_sd, local, global_mobile, valid = [np.concatenate(phase) for phase in zip(*arrays)]
    identifiers, groups, labels, roles, folds, tags, digests = zip(*records)
    data = FeatureDataset(
        global_features=global_sd,
        local_features=local,
        global_mobile=global_mobile,
        local_valid=valid,
        labels=np.array(labels),
        roles=np.array(roles),
        groups=np.array(groups),
        ids=np.array(identifiers),
        candidate_mask=np.array(tags),
        inner_folds=np.array(folds),
        metadata={"layout_id": contract["layout_id"], "field_order": contract["field_order"],
                  "extractor": contract["extractor"],
                  "row_receipts": [dict(id=identifier, image_sha256=digest)
                                   for identifier, digest in zip(identifiers, digests, strict=True)]},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_feature_dataset(data, output)
    print(f"Saved features for {len(data)} documents to {output}")
    print(f"Fit with --historical-mean {metadata / 'HISTORICAL_MEAN_FP64.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path, required=True, help="folder containing the unpacked reproduction tools")
    parser.add_argument("--metadata", type=Path, required=True, help="folder containing the saved IDSpace splits and layout")
    parser.add_argument("--measurements", type=Path, required=True, help="folder containing the saved measurements")
    parser.add_argument("--output", type=Path, required=True, help="output feature file (.npz)")
    args = parser.parse_args()
    convert(args.companion, args.metadata, args.measurements, args.output)
