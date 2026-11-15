"""Stage the authenticated added SVK cohort from the publisher tarballs."""

from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile

from ..r62_svk import DATASETS, digest


METADATA_SHA = "f715d722be45456e5295e0bb17ec6baf9938fe527103ccb63cbbf7aa80cb8f7b"
ARCHIVES = dict(zip(("template", "scanned"), DATASETS, strict=True))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def metadata(path):
    require(digest(path) == METADATA_SHA, "Expected the released added SVK metadata")
    value = json.loads(gzip.decompress(path.read_bytes()))
    rows = value["rows"]
    require([r["id"] for r in rows] == list(range(12116)), "Expected the fixed ordered added population")
    require(len(value["field_order"]) == 11 and all(row["formation"] in ARCHIVES for row in rows),
            "Expected the released SVK field order and formations")
    return value


def stage(meta, archives, output):
    rows = metadata(meta)["rows"]
    files = {p.name: p for p in archives}
    require(files.keys() == DATASETS.keys(), "Expected both immutable SVK publisher archive names")
    for name, expected in DATASETS.items():
        require(digest(files[name]) == expected, f"Publisher archive differs: {name}")
    output.mkdir(parents=True, exist_ok=False)
    present = set()
    for formation, filename in ARCHIVES.items():
        by_size = defaultdict(set)
        pending = defaultdict(list)
        for row in rows:
            if row["formation"] == formation:
                key = row["encoded_bytes"], row["encoded_sha256"]
                by_size[key[0]].add(key[1])
                pending[key].append(row["id"])
        with tarfile.open(files[filename], mode="r|gz") as archive:
            for member in archive:
                if not pending:
                    break
                if member.size not in by_size or not member.isfile():
                    continue
                name = PurePosixPath(member.name)
                require(not name.is_absolute() and ".." not in name.parts and "\\" not in member.name,
                        "Expected a safe regular publisher member")
                with archive.extractfile(member) as stream:
                    raw = stream.read(member.size + 1)
                actual = sha256(raw).hexdigest()
                key = len(raw), actual
                for identity in pending.pop(key, []):
                    with (output / f"{identity}.img").open("xb") as out:
                        out.write(raw)
                    present.add(identity)
        require(not pending, f"Missing {len(pending)} expected payloads in {filename}")
        print(f"Staged {formation}: {len(present)}/12116 rows", flush=True)
    require(len(present) == 12116, "Every added view must be staged")
    write(output / "STAGING.json", {
        "schema": "NRSW_R62_ADDED_SVK_PUBLISHER_IMAGES_V1",
        "metadata_sha256": METADATA_SHA, "rows": len(present),
        "publisher_archives_sha256": DATASETS,
    })


def image_rows(meta):
    return [{
        "id": str(row["id"]), "member": f'{row["id"]}.img',
        "bytes": row["encoded_bytes"], "sha256": row["encoded_sha256"],
        "maps": {"original": row["quad_maps"]},
    } for row in meta["rows"]]
