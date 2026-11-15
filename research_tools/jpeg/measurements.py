"""Re-extract the SVK features after one identical added JPEG operation."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, features, __version__ as pillow_version

from r29_repro import measure, state
from r29_repro.native_encoders import NativeMobile, NativeSD, Offline, ResourceGuard

SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
MOBILE_PIN = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
SEGMENT_ROWS = 640
CODEC = ("12.2.0", "8.0", "3.1.4.1")
JPEG = dict(format="JPEG", quality=85, subsampling=2, optimize=False, progressive=False)
LAYERS = [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".working")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if path.exists():
        raise FileExistsError(path)
    tmp.rename(path)


def ensure(ok, reason):
    if not ok:
        raise ValueError(reason)


def jpeg_bytes(image):
    with io.BytesIO() as stream:
        image.save(stream, **JPEG)
        return stream.getvalue()


def shared_tables():
    with Image.new("RGB", (24, 24)) as fixture:
        payload = jpeg_bytes(fixture)
    with Image.open(io.BytesIO(payload)) as encoded:
        ensure(encoded.layer == LAYERS and len(encoded.quantization) == 2, "Baseline JPEG fixture")
        return {str(key): value for key, value in encoded.quantization.items()}


def render(row, images, tables):
    path = images.joinpath(*measure.safe_member(row["member"]).parts).resolve(strict=True)
    ensure(path.is_relative_to(images) and path.stat().st_size == row["bytes"], "Exact staged source path")
    raw = path.read_bytes()
    ensure(raw.startswith(b"\xff\xd8") and len(raw) == row["bytes"]
           and measure.sha256(raw) == row["sha256"], "Source JPEG checksum")
    with Image.open(io.BytesIO(raw)) as source:
        ensure(source.width * source.height <= 32_000_000 and source.getexif().get(274, 1) == 1,
               "Unrotated bounded source")
        with source.convert("RGB") as canvas:
            canvas.info.clear()
            payload = jpeg_bytes(canvas)
            size = canvas.size
    with Image.open(io.BytesIO(payload)) as encoded:
        ensure(encoded.size == size and encoded.layer == LAYERS and not encoded.info.get("progressive")
               and {str(key): value for key, value in encoded.quantization.items()} == tables,
               "Identical new JPEG profile and original dimensions")
        image = encoded.convert("RGB")
    return image, measure.sha256(payload)


def prepare(root, metadata):
    receipt = read(metadata / "EXPORT_RECEIPT.json")
    for name, pin in receipt["files"].items():
        measure.check(metadata / name, pin)
    measure.plan(SimpleNamespace(metadata=metadata, contexts=["all"], roles=["train", "calibration", "test"],
                                 output=root / "PLAN.json"))
    plan = measure.load_plan(root / "PLAN.json")
    ensure(plan["image_count"] == 16260 and plan["required_recovered_variants"] == 186
           and len(plan["contexts"]) == 384, "Original ordered roster and allowed local recovery")
    save(root / "BINDINGS.json", measurement_bindings(root))


def measurement_bindings(root):
    runtime = (pillow_version, features.version_codec("jpg"), features.version_feature("libjpeg_turbo"))
    ensure(runtime == CODEC, f"Expected the archived Pillow and JPEG codecs {CODEC}, got {runtime}")
    return dict(plan=digest(root / "PLAN.json"), program=digest(__file__),
                sources=state.source_pins(), jpeg=JPEG, jpeg_runtime=list(runtime), tables=shared_tables(),
                mobile=MOBILE_PIN, sd_revision=SD_REVISION)


def checked_plan(root):
    return read(root / "PLAN.json"), checked_bindings(root)


def checked_bindings(root):
    bindings = read(root / "BINDINGS.json")
    ensure(bindings == measurement_bindings(root), "Pinned transform, geometry and measurement code")
    return bindings


def segment_rows(plan, segment):
    start = segment * SEGMENT_ROWS
    rows = plan["rows"][start:start + SEGMENT_ROWS]
    ensure(rows and segment >= 0, f"Out-of-range segment {segment}")
    return rows


def check_transform(root, images, segment):
    plan, binding = checked_plan(root)
    rows = []
    for row in segment_rows(plan, segment):
        image, payload = render(row, images, binding["tables"])
        with image:
            rows.append(dict(id=row["id"], size=list(image.size), payload_sha256=payload,
                             rgb_sha256=measure.sha256(image.tobytes())))
    result = root / f"TRANSFORM_PROBE_{segment:03d}.json"
    save(result, dict(bindings=digest(root / "BINDINGS.json"), rows=rows))
    print(json.dumps(dict(verified=len(rows), result=str(result))))


def worker(root, stage, segment, images, mobile, sd_snapshot, worker_input=None, input_sha256=None):
    bindings = checked_bindings(root)
    binding_hash = digest(root / "BINDINGS.json")
    if worker_input is None:
        rows = segment_rows(read(root / "PLAN.json"), segment)
    else:
        ensure(digest(worker_input) == input_sha256, "Exact bounded worker packet")
        packet = read(worker_input)
        ensure(packet["bindings"] == binding_hash and packet["segment"] == segment
               and 0 < len(packet["rows"]) <= SEGMENT_ROWS, "Worker packet belongs to this measurement plan")
        rows = packet["rows"]
        del packet
    ensure(digest(mobile) == MOBILE_PIN, "Expected the official MobileNet checkpoint")
    stage_root = root / stage / f"{segment:03d}"
    stage_root.mkdir(parents=True, exist_ok=False)
    guard = None
    chunks = []
    encoder = None
    try:
        guard = ResourceGuard(1800)
        with Offline() as offline:
            encoder = NativeMobile(mobile, guard) if stage == "mobile" else NativeSD(sd_snapshot, guard)
            if stage == "sd":
                parent_root = root / "mobile" / f"{segment:03d}"
                parent = read(parent_root / "COMPLETE.json")
                ensure(parent["bindings"] == binding_hash
                       and parent["ids"] == [row["id"] for row in rows], "Exact MobileNet RGB parent")
            for cursor in range(0, len(rows), 8):
                batch = rows[cursor:cursor + 8]
                chunk = cursor // 8
                payloads = []

                def decode(row):
                    image, encoded_sha = render(row, images, bindings["tables"])
                    payloads.append(encoded_sha)
                    return image

                ids = [row["id"] for row in batch]
                if stage == "mobile":
                    arrays, rasters = measure.measure_mobile_rows(batch, encoder, decode, guard)
                else:
                    prior = parent["chunks"][chunk]
                    rasters = prior["rasters"]
                    ensure([row["id"] for row in rasters] == ids, "Paired chunk order")
                    sd = np.empty((len(batch), 512), np.float32)
                    for i, (row, receipt) in enumerate(zip(batch, rasters, strict=True)):
                        guard()
                        with decode(row) as image:
                            ensure(measure.sha256(image.tobytes()) == receipt["rgb_sha256"]
                                   and list(image.size) == receipt["size"], "Equal MobileNet and SD rasters")
                            sd[i] = encoder.one(image)
                    ensure(payloads == prior["payload_sha256"] and np.isfinite(sd).all(),
                           "Equal transformed payloads and finite SD block")
                    arrays = {"sd512": sd}
                output = stage_root / f"{chunk:04d}.npz"
                with output.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                chunks.append(dict(ids=ids, path=output.name, sha256=digest(output),
                                   payload_sha256=payloads, rasters=rasters))
            contract = encoder.contract()
            if stage == "sd":
                ensure(encoder.completed == len(rows), "Exactly one SD forward per document")
            ensure(offline.attempts == 0 and binding_hash == digest(root / "BINDINGS.json"),
                   "No network access and unchanged bindings during extraction")
        save(stage_root / "COMPLETE.json", dict(ids=[row["id"] for row in rows],
             bindings=binding_hash, chunks=chunks, contract=contract, resources=guard.receipt()))
    except BaseException as exc:
        resources = guard.receipt() if guard is not None else None
        save(stage_root / "FAILURE.json", dict(error=repr(exc), chunks=len(chunks), resources=resources))
        raise
    finally:
        if encoder is not None:
            encoder.close()


def controller(root, args):
    ensure(not (root / "MEASUREMENTS_COMPLETE.json").exists(), "Measurements already complete")
    plan, _ = checked_plan(root)
    binding_hash = digest(root / "BINDINGS.json")
    started = time.monotonic()
    row_count = len(plan["rows"])
    segments = (row_count + SEGMENT_ROWS - 1) // SEGMENT_ROWS
    packets = root / "segments"
    packets.mkdir(exist_ok=True)
    descriptors = []
    for segment in range(segments):
        rows = segment_rows(plan, segment)
        packet = packets / f"{segment:03d}.json"
        expected = dict(bindings=binding_hash, segment=segment, rows=rows)
        if packet.exists():
            ensure(read(packet) == expected, f"Existing plan slice differs for segment {segment}")
        else:
            save(packet, expected)
        descriptors.append((packet, digest(packet), [row["id"] for row in rows]))
    del plan, rows, expected
    for stage in ("mobile", "sd"):
        for segment, (packet, packet_hash, ids) in enumerate(descriptors):
            dest = root / stage / f"{segment:03d}"
            if not (dest / "COMPLETE.json").is_file():
                if dest.exists():
                    dest.rename(dest.with_name(f"{dest.name}.incomplete-{time.time_ns()}"))
                with (root / f"{stage}_{segment:03d}.log").open("ab") as log:
                    subprocess.run([sys.executable, "-B", "-m", "research_tools.jpeg",
                                    "--r5-runtime", str(args.r5_runtime.resolve()), "--run", str(root),
                                    "_worker", stage, str(segment), "--images", str(args.images.resolve()),
                                    "--mobile-weights", str(args.mobile_weights.resolve()), "--sd-snapshot",
                                    str(args.sd_snapshot.resolve()), "--worker-input", str(packet),
                                    "--input-sha256", packet_hash], stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1820)
            seal = read(dest / "COMPLETE.json")
            ensure(seal["bindings"] == binding_hash == digest(root / "BINDINGS.json")
                   and seal["ids"] == ids
                   and [item for chunk in seal["chunks"] for item in chunk["ids"]] == ids
                   and all(entry["path"] == f"{index:04d}.npz"
                           and digest(dest / entry["path"]) == entry["sha256"]
                           for index, entry in enumerate(seal["chunks"])),
                   "Segment contents and source seal")
            print(f"{stage}: {segment + 1}/{segments} segments verified", flush=True)
    save(root / "MEASUREMENTS_COMPLETE.json", dict(bindings=digest(root / "BINDINGS.json"),
         rows=row_count, segments=segments, elapsed_seconds=round(time.monotonic() - started, 3),
         receipts={stage: [digest(root / stage / f"{i:03d}" / "COMPLETE.json")
                           for i in range(segments)] for stage in ("mobile", "sd")}))
