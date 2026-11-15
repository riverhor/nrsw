"""Join released roles to independently authenticated original and added image bytes."""

import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath

from PIL import Image

from .cohort import METADATA_SHA, read, require
from .refit import load_design


SIZES = {"template": (1134, 716), "scanned": (2480, 3507)}


def original_assets(runtime, documents, root):
    geometry = {}
    with gzip.open(runtime / "metadata/registered_geometry.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["id"]
            require(identity not in geometry and row["sha256"] == documents[identity]["sha256"],
                    "Expected matching original registry/geometry identity")
            geometry[identity] = row
    result = {}
    for identity, doc in documents.items():
        path = PurePosixPath(doc["member"])
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in doc["member"],
                "Expected a contained original image member")
        formation = "scanned" if path.parts[0] == "scanned" else "template"
        row = geometry[identity]
        result[identity] = {
            "id": identity, "path": root.joinpath(*path.parts), "size": SIZES[formation],
            "sha256": doc["sha256"], "bytes": doc["bytes"], "formation": formation,
            "maps": {"original": row["maps"], "recovered": row["recovery_maps"]},
        }
    return result


def added_assets(added, root):
    if root is None:
        return {fold: [] for fold in added}
    seal = read(root / "STAGING.json")
    require(seal["metadata_sha256"] == METADATA_SHA and seal["rows"] == 12116,
            "Expected the authenticated added image stage")
    return {fold: [{
        "id": f'test_scale_20261107:{row["id"]}', "path": root / f'{row["id"]}.img',
        "size": tuple(row["image_size"]), "sha256": row["encoded_sha256"], "bytes": row["encoded_bytes"],
        "formation": row["formation"], "maps": {"original": row["quad_maps"]},
    } for row in rows] for fold, rows in added.items()}


def load_roles(runtime, meta, original_images, added_images):
    registry, contexts, documents, added = load_design(runtime / "metadata", meta)
    return registry, contexts, original_assets(runtime, documents, original_images), added_assets(added, added_images)


def open_image(asset):
    blob = asset["path"].read_bytes()
    require(len(blob) == asset["bytes"] and sha256(blob).hexdigest() == asset["sha256"],
            f'Image payload differs: {asset["id"]}')
    with Image.open(BytesIO(blob)) as source:
        require(source.size == asset["size"] and source.getexif().get(274, 1) == 1,
                f'Image geometry or orientation differs: {asset["id"]}')
        return source.convert("RGB")
