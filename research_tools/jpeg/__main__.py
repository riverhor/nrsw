"""Rerun the IDSpace experiment with common JPEG settings."""

import argparse
import hashlib
import os
from pathlib import Path
import sys


R5_MANIFEST_SHA256 = "4787d2afd36520cf5483b4ea60d06b1a3de9e4462668b2568de914dcc12f682e"


def verify_runtime(runtime):
    manifest = runtime / "SHA256SUMS"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != R5_MANIFEST_SHA256:
        raise ValueError("The reproduction files do not match the archived checksums")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        item = (runtime / name).resolve(strict=True)
        if not item.is_relative_to(runtime) or hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reproduction file checksum does not match: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-runtime", required=True, type=Path, help="Unpacked reproduction files")
    parser.add_argument("--run", required=True, type=Path, help="New working directory for this JPEG run")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Use the original splits and set the shared JPEG settings")
    stage = commands.add_parser("stage", help="Prepare the original images in the run directory")
    acquisition = stage.add_mutually_exclusive_group(required=True)
    acquisition.add_argument("--archives", nargs=2, type=Path)
    acquisition.add_argument("--local-map", type=Path)
    probe = commands.add_parser("check-transform", help="Decode and re-encode one segment without models")
    probe.add_argument("--images", required=True, type=Path)
    probe.add_argument("--segment", type=int, default=0)
    for name in ("measure", "_worker"):
        command = commands.add_parser(name, help="Run or resume extraction" if name == "measure" else argparse.SUPPRESS)
        command.add_argument("--images", required=True, type=Path)
        command.add_argument("--mobile-weights", required=True, type=Path)
        command.add_argument("--sd-snapshot", required=True, type=Path)
        if name == "_worker":
            command.add_argument("encoder", choices=("mobile", "sd"))
            command.add_argument("segment", type=int)
            command.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
            command.add_argument("--input-sha256", help=argparse.SUPPRESS)
    refit = commands.add_parser("refit", help="Fit and calibrate all five methods before scoring the test set")
    refit.add_argument("--qualification", action="store_true", help="Run one fit per fold")
    frozen = commands.add_parser("freeze", help="Score common JPEG with original fitted models and cutoffs")
    frozen.add_argument("--native-measurements", required=True, type=Path, help="Verified features from images with their original encoding")
    mode = frozen.add_mutually_exclusive_group()
    mode.add_argument("--submission", type=Path, help="Compare the fixed-model results with the paper")
    mode.add_argument("--qualification", action="store_true", help="Run one fit from the first fold")
    intervals = commands.add_parser("intervals", help="Recompute paired JPEG-refit intervals")
    intervals.add_argument("--native-run", type=Path, help="Optional original-encoding raw rerun for paired changes")
    verify = commands.add_parser("verify", help="Compare operating points and intervals with the paper")
    verify.add_argument("--submission", required=True, type=Path)
    args = parser.parse_args()
    runtime = args.r5_runtime.resolve(strict=True)
    root = args.run.resolve()
    verify_runtime(runtime)
    sys.path.insert(0, str(runtime / "package"))
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    from . import measurements
    if args.command == "plan":
        root.mkdir(parents=True, exist_ok=False)
        measurements.prepare(root, runtime / "metadata")
    elif args.command == "stage":
        from r29_repro import stage_images
        measurements.checked_plan(root)
        stage_images.stage(root / "PLAN.json", root / "images",
                           archives=args.archives, local_map=args.local_map)
    elif args.command == "check-transform":
        measurements.check_transform(root, args.images.resolve(strict=True), args.segment)
    elif args.command == "measure":
        measurements.controller(root, args)
    elif args.command == "_worker":
        measurements.worker(root, args.encoder, args.segment, args.images.resolve(strict=True),
                            args.mobile_weights, args.sd_snapshot, args.worker_input, args.input_sha256)
    elif args.command == "refit":
        from .refit import run
        output = "qualification" if args.qualification else "refit"
        run(root, root / output, runtime / "metadata", args.qualification)
    elif args.command == "freeze":
        from .frozen import run
        run(root, runtime / "metadata", args.native_measurements, args.submission, args.qualification)
    elif args.command == "intervals":
        from .evaluation import run
        run(root, runtime / "metadata", args.native_run)
    elif args.command == "verify":
        from .evaluation import verify
        verify(root, args.submission)


if __name__ == "__main__":
    main()
