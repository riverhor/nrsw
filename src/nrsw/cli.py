"""Command-line interface for registered-layout document detectors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from .api import DEFAULT_FPRS, Calibration, calibrate, evaluate, fit, load_model, predict, save_model
from .demo import export_demo, run_demo, trace, write_json
from .preparation import prepare_manifest
from .schema import load_feature_dataset


def read_calibration(path):
    return Calibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _prepare(args):
    return prepare_manifest(args.input, args.output, anchors=args.anchors, bank=args.bank,
                            support=args.support, calibration=args.calibration, seed=args.seed)


def _train(args):
    historical = _load_mean(args.historical_mean)
    model = fit(load_feature_dataset(args.features), historical, field_classifiers=args.field_classifiers)
    save_model(model, args.model)
    return {"model": str(args.model), "layout": model.layout.layout_id,
            "training_documents": model.metadata["role_counts"], "classifier_width": model.layout.width}


def _load_mean(path):
    historical = np.load(path, allow_pickle=False) if path else None
    if historical is not None and not isinstance(historical, np.ndarray):
        historical.close()
        raise ValueError("--historical-mean requires a NumPy vector saved as .npy")
    return historical


def _fit_images(args):
    from tempfile import TemporaryDirectory

    from .vision import encoder_options, extract_images

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"training output must be a new folder: {output}")
    historical = _load_mean(args.historical_mean)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="nrsw-fit-", dir=output.parent) as temporary:
        run = Path(temporary) / "run"
        packet = run / "features.npz"
        extract_images(args.layout, packet, manifest_path=args.manifest,
                       field_classifiers=args.field_classifiers, **encoder_options(args))
        data = load_feature_dataset(packet)
        test_labels = data.labels[data.roles == "evaluation"]
        if test_labels.size and set(test_labels) != {0, 1}:
            raise ValueError("evaluation needs labeled genuine and forged documents")
        model = fit(data, historical, field_classifiers=args.field_classifiers)
        save_model(model, run / "model")
        cuts = calibrate(model, data, args.fpr, args.delta)
        write_json(run / "cuts.json", cuts.to_dict())
        summary = {"layout": model.layout.layout_id, "detector": "field classifiers" if args.field_classifiers else "NRSW",
                   "training_documents": model.metadata["role_counts"], "cuts": cuts.to_dict()["cuts"]}
        if test_labels.size:
            metrics = {fpr: evaluate(model, data, cuts, fpr) for fpr, cut in cuts.cuts.items() if cut.available}
            write_json(run / "evaluation.json", metrics)
            summary["evaluation"] = metrics
        run.rename(output)
    return {"output": str(output), **summary}


def _calibrate(args):
    results = calibrate(load_model(args.model), load_feature_dataset(args.features), args.fpr, args.delta)
    write_json(args.output, results.to_dict())
    return {"output": str(args.output), "cuts": results.to_dict()["cuts"]}


def _predict(args):
    model = load_model(args.model)
    features = load_feature_dataset(args.features)
    cuts = read_calibration(args.calibration) if args.calibration else None
    result = predict(model, features, cuts, args.fpr)
    if args.trace:
        details = trace(model, features, cuts)
        if result["threshold"] is not None:
            details["selected_fpr"] = result["threshold"]["eta"]
        write_json(args.trace, details)
    return _output(result, args.output)


def _evaluate(args):
    result = evaluate(load_model(args.model), load_feature_dataset(args.features),
                      read_calibration(args.calibration), args.fpr)
    return _output(result, args.output)


def _reproduce(args):
    from .reproduce import main as reproduce_main

    command = [str(args.archive)]
    if args.svk_scores is not None:
        command += ["--svk-scores", str(args.svk_scores)]
    if args.extract_r5 is not None:
        command += ["--extract-r5", str(args.extract_r5)]
    if args.json:
        command.append("--json")
    if reproduce_main(command):
        raise SystemExit(2)


def _route(args):
    from .registry import LayoutRegistry
    from .vision import encoder_options

    with LayoutRegistry.from_file(args.registry, **encoder_options(args)) as registry:
        result = registry.predict_image(args.layout_id, args.image, fpr=args.fpr,
                                        identifier=args.id, group=args.group)
    return _output(result, args.output)


def _predict_image(args):
    from tempfile import TemporaryDirectory

    from .field_classifiers import FieldClassifierModel
    from .vision import encoder_options, extract_images

    model = load_model(args.model)
    cuts = read_calibration(args.calibration)
    if model.training_identity != cuts.training_identity:
        raise ValueError("calibration belongs to a different fitted model")
    cuts.cut(args.fpr)
    with TemporaryDirectory(prefix="nrsw-predict-") as temporary:
        packet = Path(temporary) / "query.npz"
        extract_images(args.layout, packet, image_path=args.image,
                       field_classifiers=isinstance(model, FieldClassifierModel), **encoder_options(args))
        result = predict(model, load_feature_dataset(packet), cuts, args.fpr)
    return _output(result, args.output)


def _output(result, path):
    if path is None:
        return result
    write_json(path, result)
    return {"output": str(path)}


def _serve(args):
    from .server import serve
    from .demo import example_run

    paths = (args.model, args.features, args.calibration)
    if any(paths):
        if not all(paths):
            raise ValueError("pass --model, --features and --calibration together")
        data = trace(load_model(args.model), load_feature_dataset(args.features), read_calibration(args.calibration))
    else:
        _, _, test, model, cuts = example_run()
        data = trace(model, test, cuts, specimen=True)
    serve(data, args.port)


def build_parser():
    from .vision import add_encoder_options, add_vision_commands

    parser = argparse.ArgumentParser(description="Train and use NRSW document forgery detectors.")
    parser.add_argument("--version", action="version", version=f"nrsw {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    reproduce = commands.add_parser("reproduce", help="recompute the paper's results from saved scores")
    reproduce.add_argument("archive", type=Path, help="paper and results ZIP")
    reproduce.add_argument("--svk-scores", type=Path, help="optional IDSpace score ZIP")
    reproduce.add_argument("--extract-r5", type=Path, help="copy the image reproduction tools to a new ZIP")
    reproduce.add_argument("--json", action="store_true", help="write the verification report as JSON")
    reproduce.set_defaults(handler=_reproduce)

    prepare = commands.add_parser("prepare", help="split images by source-document group")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name, default in (("anchors", 50), ("bank", 225), ("support", 15), ("calibration", 500), ("seed", 2026)):
        prepare.add_argument(f"--{name}", type=int, default=default)
    prepare.set_defaults(handler=_prepare)
    add_vision_commands(commands)

    fit_images = commands.add_parser("fit-images", help="train and calibrate a detector from images")
    fit_images.add_argument("--manifest", type=Path, required=True, help="image list with training and calibration assignments")
    fit_images.add_argument("--layout", type=Path, required=True, help="JSON file with field locations")
    fit_images.add_argument("--output", type=Path, required=True, help="new folder for features, model, thresholds, and any evaluation results")
    fit_images.add_argument("--historical-mean", type=Path, help="mean whole-image features from genuine documents (.npy)")
    fit_images.add_argument("--field-classifiers", action="store_true", help="fit the optional NRSW field-classifier extension")
    fit_images.add_argument("--fpr", nargs="+", default=DEFAULT_FPRS, help="one or more target false-positive rates")
    fit_images.add_argument("--delta", default="0.05", help="allowed probability of exceeding the target false-positive rate")
    add_encoder_options(fit_images)
    fit_images.set_defaults(handler=_fit_images)

    predict_image = commands.add_parser("predict-image", help="score one image with a fitted model and calibrated threshold")
    predict_image.add_argument("--image", type=Path, required=True, help="unlabelled image")
    predict_image.add_argument("--layout", type=Path, required=True, help="layout used for training")
    predict_image.add_argument("--model", type=Path, required=True, help="model directory")
    predict_image.add_argument("--calibration", type=Path, required=True, help="calibration file (.json)")
    predict_image.add_argument("--fpr", default="0.01", help="target false-positive rate used to select the threshold")
    predict_image.add_argument("--output", type=Path)
    add_encoder_options(predict_image)
    predict_image.set_defaults(handler=_predict_image)

    route = commands.add_parser("route", help="score one image with a known layout from a registry")
    route.add_argument("--registry", type=Path, required=True, help="layout registry JSON")
    route.add_argument("--layout-id", required=True, help="known layout ID")
    route.add_argument("--image", type=Path, required=True)
    route.add_argument("--id", help="optional query ID")
    route.add_argument("--group", help="optional source-document group ID")
    route.add_argument("--fpr", default="0.01", help="target false-positive rate used to select the threshold")
    route.add_argument("--output", type=Path)
    add_encoder_options(route)
    route.set_defaults(handler=_route)

    def model_command(name, help_text, handler):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--features", type=Path, required=True, help="document features (.npz)")
        command.add_argument("--model", type=Path, required=True, help="model directory")
        command.set_defaults(handler=handler)
        return command

    train = model_command("train", "fit a detector for one registered layout", _train)
    train.add_argument("--historical-mean", type=Path, help="mean whole-image features from genuine documents (.npy)")
    train.add_argument("--field-classifiers", action="store_true", help="Fit the optional NRSW field-classifier extension")

    calibration = model_command("calibrate", "set thresholds using genuine calibration documents", _calibrate)
    calibration.add_argument("--output", type=Path, required=True, help="calibration file (.json)")
    calibration.add_argument("--fpr", nargs="+", default=DEFAULT_FPRS, help="one or more target false-positive rates")
    calibration.add_argument("--delta", default="0.05", help="allowed probability of exceeding the target false-positive rate")

    for name, help_text, handler in (("predict", "score new documents", _predict),
                                     ("evaluate", "measure errors on test documents at a calibrated threshold", _evaluate)):
        command = model_command(name, help_text, handler)
        command.add_argument("--calibration", type=Path, required=name == "evaluate")
        command.add_argument("--fpr", default="0.01", help="target false-positive rate used to select the threshold")
        command.add_argument("--output", type=Path)
        if name == "predict":
            command.add_argument("--trace", type=Path, help="export scores and field weights for the demo page")

    for name, help_text, action in (("demo", "train and test on simulated features", run_demo),
                                    ("export-demo", "build a static demo for a web host", export_demo)):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", type=Path, required=True)
        command.set_defaults(handler=lambda args, action=action: action(args.output))

    server = commands.add_parser("serve", help="open the demo on your machine")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--model", type=Path)
    server.add_argument("--features", type=Path)
    server.add_argument("--calibration", type=Path)
    server.set_defaults(handler=_serve)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError, ArithmeticError, KeyError) as error:
        print(f"nrsw: {error}", file=sys.stderr)
        return 2
    return 0
