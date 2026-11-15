"""Compare detectors on the original and expanded sets of Slovak ID images."""

import argparse
from pathlib import Path
import sys

from . import cohort
from ..r62_svk import runtime_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--added-metadata", type=Path, default=Path(__file__).with_name("ADDED_SVK_METADATA.json.gz"))
    action = parser.add_subparsers(dest="action", required=True)
    stage = action.add_parser("stage-added", help="prepare the 12,116 additional images from the published archives")
    stage.add_argument("--archives", nargs=2, type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    measure = action.add_parser("measure-added-segment", help="extract MobileNet or Stable Diffusion features for one segment")
    measure.add_argument("--r5-runtime", type=Path, required=True)
    measure.add_argument("--images", type=Path, required=True)
    measure.add_argument("--measurements", type=Path, required=True)
    measure.add_argument("--encoder", choices=("mobile", "sd"), required=True)
    measure.add_argument("--weights", type=Path, required=True)
    measure.add_argument("--segment", type=int, required=True)
    refit = action.add_parser("fit-shared-encoders", help="fit NRSW and the classifiers using the same encoders")
    refit.add_argument("--r5-runtime", type=Path, required=True)
    refit.add_argument("--original-measurements", type=Path, required=True)
    refit.add_argument("--added-measurements", type=Path)
    refit.add_argument("--shared-parent", type=Path)
    refit.add_argument("--context")
    refit.add_argument("--output", type=Path, required=True)
    for method in ("resnet18", "dinov3", "flid"):
        branch = action.add_parser(f"fit-{method}", help=f"refit and evaluate the SVK {method} comparator")
        branch.add_argument("--r5-runtime", type=Path, required=True)
        branch.add_argument("--original-images", type=Path, required=True)
        branch.add_argument("--added-images", type=Path)
        branch.add_argument("--weights", type=Path, required=True)
        branch.add_argument("--output", type=Path, required=True)
        branch.add_argument("--context")
        if method == "flid":
            branch.add_argument("--flid-source", type=Path, required=True)
    rates = action.add_parser("evaluate", help="compute error rates and paired intervals for both test sets")
    rates.add_argument("--r5-runtime", type=Path, required=True)
    rates.add_argument("--runs", type=Path, nargs="+", required=True)
    rates.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "stage-added":
        cohort.stage(args.added_metadata, args.archives, args.output)
        return
    runtime = args.r5_runtime.resolve(strict=True)
    runtime_sources(runtime)
    sys.path.insert(0, str(runtime / "package"))
    if args.action == "measure-added-segment":
        from .added_measure import measure_segment

        measure_segment(args.added_metadata, args.images, args.measurements, args.encoder, args.segment, args.weights)
        return
    meta = cohort.metadata(args.added_metadata)
    if args.action == "fit-shared-encoders":
        from .refit import METHODS, run

        methods = METHODS if args.shared_parent else METHODS[:2]
        run(runtime, args.original_measurements, args.added_measurements, meta,
            args.output, methods, args.shared_parent, args.context)
    elif args.action == "evaluate":
        from .evaluate import run

        run(runtime, meta, args.runs, args.output)
    elif args.action == "fit-flid":
        from .flid import run

        run(runtime, meta, args.original_images, args.added_images, args.weights, args.flid_source,
            args.output, args.context)
    else:
        if args.action == "fit-resnet18":
            from .resnet import run
        else:
            from .dinov3 import run
        run(runtime, meta, args.original_images, args.added_images, args.weights,
            args.output, args.context)


if __name__ == "__main__":
    main()
