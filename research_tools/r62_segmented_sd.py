"""Run the archived R5 SD operator in bounded workers for any released SVK plan."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from r29_repro import measure as m


CHUNKS_PER_WORKER = 80
WORKER_SECONDS = 900


def load_addon(path):
    m.need(m.sha(path) == "9ae1ccbda049772df72a7b587a61aacc11f73287cab2fa6b933e73eb644fe4d5",
           "Use the authenticated R5 segmented-SD operator")
    spec = spec_from_file_location("r29_segmented_sd", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(args):
    plan = m.load_plan(args.plan)
    bindings = input_pins(args)
    mobile = m.check_stage(args.mobile, plan, bindings["plan_sha256"], "R29_SHARED_MOBILE_V1")
    m.need(mobile["admission"] == bindings["admission"], "SD and Mobile must share the exact plan admission")
    return plan, mobile, bindings


def input_pins(args):
    return dict(plan_sha256=m.sha(args.plan), mobile_sha256=m.sha(args.mobile / "COMPLETE.json"),
                addon_sha256=m.sha(args.addon), wrapper_sha256=m.sha(Path(__file__)),
                admission=m.admitted(args, "sd"), sources=m.state.source_pins())


def measure(args, addon, packet, destination):
    from r29_repro.native_encoders import NativeSD, Offline, ResourceGuard

    bindings, span = packet["bindings"], packet["span"]
    start, stop = span
    rows, source_chunks = packet["rows"], packet["mobile_chunks"]
    first = source_chunks[0]["start"]
    local_chunks = [{**entry, "start": entry["start"] - first} for entry in source_chunks]
    plan = {"rows": rows}
    mobile = {"chunks": local_chunks}
    m.need(stop - start == len(source_chunks) > 0, "Expected the worker's full chunk span")
    m.need(input_pins(args) == bindings, "Unchanged segmented SD input")
    guard = ResourceGuard(WORKER_SECONDS)
    try:
        with Offline() as offline:
            encoder = NativeSD(args.sd_snapshot, guard)
            try:
                chunks = addon.measure_range(m, plan, mobile, (0, len(local_chunks)), destination, args.images_root, encoder)
                contract = encoder.contract()
            finally:
                encoder.close()
            row_count = sum(len(chunk["ids"]) for chunk in chunks)
            m.need(encoder.started == encoder.completed == row_count and offline.attempts == 0
                   and contract["leaf_cleanup"]["close_completed"] is True,
                   "Expected one offline SD forward per image")
    except (MemoryError, RuntimeError, OSError, ValueError) as error:
        m.write(destination / "FAILURE.json", dict(error=repr(error), resource=guard.receipt()))
        raise
    for number, chunk in enumerate(chunks, start):
        name = addon.chunk_name(number)
        temporary = destination / f"segment-{name}"
        (destination / chunk["features"]["path"]).rename(temporary)
        chunk["start"] += first
        chunk["features"]["path"] = name
    for number, chunk in enumerate(chunks, start):
        name = addon.chunk_name(number)
        (destination / f"segment-{name}").rename(destination / name)
        m.check(destination / name, chunk["features"])
    m.write(destination / "COMPLETE.json", dict(bindings=bindings, span=list(span), rows=row_count, chunks=chunks,
                                                 contract=contract, resource=guard.receipt()))


def run(args):
    addon = load_addon(args.addon)
    work = args.output.with_name(args.output.name + "_segments")
    work.mkdir(exist_ok=True)
    if args.worker is not None:
        destination = work / addon.segment_name(args.worker)
        packet = m.read(destination / "INPUT.json")
        measure(args, addon, packet, destination)
        return

    plan, mobile, bindings = load_inputs(args)
    spans = addon.span_ranges(len(mobile["chunks"]), CHUNKS_PER_WORKER)
    if args.output.exists():
        raise FileExistsError(f"SD output already exists: {args.output}")
    for index, (start, stop) in enumerate(spans):
        destination = work / addon.segment_name(index)
        if (destination / "COMPLETE.json").exists():
            continue
        if destination.exists():
            destination.rename(work / f"{destination.name}.incomplete-{time.time_ns()}")
        destination.mkdir()
        chunks = mobile["chunks"][start:stop]
        first = chunks[0]["start"]
        last = chunks[-1]["start"] + len(chunks[-1]["ids"])
        m.write(destination / "INPUT.json", dict(bindings=bindings, span=[start, stop],
                                                  mobile_chunks=chunks, rows=plan["rows"][first:last]))
    row_count = len(plan["rows"])
    plan.clear()
    started = time.monotonic()
    parts = []
    for index, span in enumerate(spans):
        destination = work / addon.segment_name(index)
        seal = destination / "COMPLETE.json"
        if not seal.is_file():
            command = [sys.executable, "-B", __file__, *sys.argv[1:], "--worker", str(index)]
            with (work / f"{destination.name}.log").open("ab") as log:
                subprocess.run(command, timeout=WORKER_SECONDS + 60, stdout=log, stderr=subprocess.STDOUT,
                               check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        part = m.read(seal)
        m.need(part["bindings"] == bindings and part["span"] == list(span), "Exact SD worker inputs")
        limits = part["resource"]
        m.need(limits["max_seconds"] == WORKER_SECONDS and 0 <= limits["elapsed_seconds"] <= WORKER_SECONDS
               and limits["failed_branch"] is None and limits["minimum_available_bytes"] >= addon.SYSTEM_RESERVE
               and 0 <= limits["peak_private_bytes"] <= limits["max_private_bytes"] == addon.MAX_PRIVATE
               and 0 <= limits["cuda_peak_allocated"] <= limits["cuda_peak_reserved"] <= limits["max_gpu_bytes"] == addon.MAX_GPU,
               "SD worker time and memory guards")
        m.need(len(part["chunks"]) == span[1] - span[0], "SD worker complete chunk coverage")
        for number, (source, chunk) in enumerate(zip(mobile["chunks"][span[0]:span[1]], part["chunks"], strict=True), span[0]):
            m.need(source["ids"] == chunk["ids"] and source["start"] == chunk["start"], "SD/Mobile chunk order")
            filename = chunk["features"]["path"]
            m.need(filename == addon.chunk_name(number), "Fixed sequential SD chunk filename")
            path = (destination / filename).resolve()
            m.need(path.is_relative_to(destination.resolve()), "Contained SD chunk")
            with np.load(m.check(path, chunk["features"]), allow_pickle=False) as packet:
                values = packet["sd512"]
                m.need(packet.files == ["sd512"] and values.dtype == np.float32
                       and values.shape == (len(chunk["ids"]), 512) and np.isfinite(values).all(),
                       "Finite native SD512 chunk")
        m.need(part["rows"] == sum(len(chunk["ids"]) for chunk in part["chunks"]), "SD worker row count")
        parts.append((destination, m.pin(seal), part))
        print(f"SD workers: {index + 1}/{len(spans)}", flush=True)

    m.need(input_pins(args) == bindings, "Input changed during SD extraction")
    contract = addon.combined_contract(m, parts)
    resource = dict(parts[-1][2]["resource"], elapsed_seconds=time.monotonic() - started)
    for key in ("peak_private_bytes", "peak_rss_bytes", "cuda_peak_allocated", "cuda_peak_reserved"):
        resource[key] = max(part["resource"][key] for _, _, part in parts)
    resource["minimum_available_bytes"] = min(part["resource"]["minimum_available_bytes"] for _, _, part in parts)
    # The R5 validator opens the merged chunk files from the final SD directory.
    staging = work / f"merged-{time.time_ns()}"
    staging.mkdir()
    chunks = []
    for destination, _, part in parts:
        for chunk in part["chunks"]:
            filename = chunk["features"]["path"]
            shutil.copyfile(destination / filename, staging / filename)
            m.check(staging / filename, chunk["features"])
            chunks.append(chunk)
    m.write(staging / "COMPLETE.json", dict(schema="R29_SHARED_SD_V1", status="COMPLETE",
            plan_sha256=bindings["plan_sha256"], mobile_sha256=bindings["mobile_sha256"],
            rows=row_count, sources=bindings["sources"], admission=bindings["admission"],
            contract=contract, resource=resource, chunks=chunks,
            segment_receipts=[dict(path=pin["path"], sha256=pin["sha256"]) for _, pin, _ in parts]))
    m.check_stage(staging, m.load_plan(args.plan), bindings["plan_sha256"], "R29_SHARED_SD_V1")
    staging.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "images-root", "admission", "mobile", "sd-snapshot", "addon", "output"):
        parser.add_argument(f"--{name}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    run(parser.parse_args())
