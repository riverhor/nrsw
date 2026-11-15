"""Download the IDSpace data and rerun the Slovak ID experiments."""

import argparse
from hashlib import file_digest, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile


TOOLS = Path(__file__).resolve().parent
SUBMISSION_SHA256 = "c5cbbd107a9774fd604eb311a59c7d87184d8e5b68a70569655ea21313f6b9d1"
RUNTIME_SHA256 = "1052caaed7615a5749e1e257caece9ddc38138cd3be66974039255cfe87a1724"
ADDON_SHA256 = "a34de62fe99a3fb3ddb3b33af24737cb81734e4440e8a77f771f0c7a2fab9fe5"
SVK_REVISION = "a3de23b950d55a39fbd36cfaee16bd260c4d457d"
SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DATASETS = {
    "SVK_template.tar.gz": "2ef411ce08069ed5bf859a2aa294ff951ef8c1fa908ad16d91ba52f0ba9142c7",
    "SVK_scan.tar.gz": "fe12457c293480aafbe82d743872207041fd8c0148ee3dc21ea1e24f2869a99f",
}
MOBILE = "mobilenet_v3_small-047dcff4.pth"
MOBILE_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
SD_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors", "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
)


def digest(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def runtime_sources(root):
    manifest = root / "SHA256SUMS"
    verify(manifest, "4787d2afd36520cf5483b4ea60d06b1a3de9e4462668b2568de914dcc12f682e")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split("  ", 1)
        verify(root / relative_path, expected)


def unpack(submission, work):
    verify(submission, SUBMISSION_SHA256)
    root = work / "r5"
    if root.exists():
        raise FileExistsError(f"runtime already exists: {root}")
    work.mkdir(parents=True, exist_ok=True)
    with ZipFile(submission) as source:
        for name, expected, destination in (
            ("NRSW_R29_portable_reproduction_r5.zip", RUNTIME_SHA256, root),
            ("NRSW_R29_segmented_sd_addon_r1.zip", ADDON_SHA256, work / "sd-addon"),
        ):
            raw = source.read(f"submission_evidence/reproduction/{name}")
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
            with ZipFile(BytesIO(raw)) as archive:
                destination.mkdir(exist_ok=False)
                for entry in archive.infolist():
                    path = PurePosixPath(entry.filename)
                    target = destination.joinpath(*path.parts).resolve()
                    if not target.is_relative_to(destination.resolve()) or "\\" in entry.filename:
                        raise ValueError(f"Unsafe archive member: {entry.filename}")
                    if entry.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(entry) as stream, target.open("xb") as output:
                            shutil.copyfileobj(stream, output)
        manifest = json.loads(source.read("PACKAGE_MANIFEST_R62.json"))["members"]
        for member, filename in (
            ("submission_evidence/svk/RESULTS.json", "svk-complete-estimator.json"),
            ("reviewer_evidence_r52/weight_strength/independent_outcome_audit.json", "svk-weight-strength.json"),
            ("reviewer_evidence_r52/shared_jpeg/refit_aggregate.json", "svk-common-jpeg.json"),
        ):
            raw = source.read(member)
            if sha256(raw).hexdigest() != manifest[member]["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {member}")
            with (work / filename).open("xb") as output:
                output.write(raw)
    return root


def acquire(work, archives, sd_snapshot, mobile, *, encoders_only=False):
    from huggingface_hub import hf_hub_download, snapshot_download
    from urllib.request import urlopen

    work.mkdir(parents=True, exist_ok=True)
    archives = archives or work / "archives"
    if not encoders_only:
        archives.mkdir(parents=True, exist_ok=True)
        for name, expected in DATASETS.items():
            path = archives / name
            if not path.exists():
                hf_hub_download("cactuslab/IDSpace", name, repo_type="dataset", revision=SVK_REVISION,
                                local_dir=archives)
            verify(path, expected)
    if sd_snapshot is None:
        sd_snapshot = work / "sd15"
        snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", revision=SD_REVISION,
                          local_dir=sd_snapshot, allow_patterns=list(SD_FILES))
    if mobile is None:
        mobile = work / MOBILE
        if not mobile.is_file():
            part = mobile.with_suffix(".download")
            with urlopen(f"https://download.pytorch.org/models/{MOBILE}", timeout=60) as source, part.open("xb") as target:
                shutil.copyfileobj(source, target)
            verify(part, MOBILE_SHA256).rename(mobile)
    verify(mobile, MOBILE_SHA256)
    for name in SD_FILES:
        if not (sd_snapshot / name).is_file():
            raise FileNotFoundError(f"Missing Stable Diffusion asset: {sd_snapshot / name}")
    paths = {"archives": str(archives.resolve()), "sd_snapshot": str(sd_snapshot.resolve()), "mobile": str(mobile.resolve())}
    (work / "assets.json").write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")


def run(work, context, stage, local_images=None):
    root = work / "r5"
    runtime_sources(root)
    if context != "all" and not (context.startswith("f") and context.replace("_", "").isalnum()):
        raise ValueError(f"Invalid archived context ID: {context!r}")
    store = work / ("all" if context == "all" else context)
    store.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "package"), PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def call(*argv):
        command = [sys.executable, "-B", *map(str, argv)]
        subprocess.run(command, env=env, cwd=root, check=True)

    metadata, plan = root / "metadata", store / "measurements/PLAN.json"
    admission, images = store / "ADMISSION.json", store / "images"
    if stage in ("prepare", "all"):
        if not plan.exists():
            call("-m", "r29_repro.measure", "plan", "--metadata", metadata, "--contexts", context,
                 "--roles", "train", "calibration", "test", "--output", plan)
        if not admission.exists():
            call(root / "authorize_measurements.py", "--package", root / "package", "--metadata", metadata,
                 "--plan", plan, "--bundle", root / "review/MEASUREMENT_BUNDLE.json", "--output", admission)
        if not (images / "STAGING.json").exists():
            if images.exists():
                images.rename(images.with_name(f"images.incomplete-{time.time_ns()}"))
            if local_images:
                call("-m", "r29_repro.stage_images", "--plan", plan, "--output", images,
                     "--local-map", local_images)
            else:
                assets = json.loads((work / "assets.json").read_text(encoding="utf-8"))
                call("-m", "r29_repro.stage_images", "--plan", plan, "--output", images,
                     "--archives", *(Path(assets["archives"]) / name for name in DATASETS))
        if stage == "prepare":
            return

    measurements = plan.parent
    if stage in ("mobile", "sd", "all"):
        assets = json.loads((work / "assets.json").read_text(encoding="utf-8"))
        base_args = ("--plan", plan, "--images-root", images, "--admission", admission)
        if stage in ("mobile", "all"):
            mobile = measurements / "mobile"
            if not (mobile / "COMPLETE.json").exists():
                if mobile.exists():
                    mobile.rename(mobile.with_name(f"mobile.incomplete-{time.time_ns()}"))
                call("-m", "r29_repro.measure", "mobile", *base_args,
                     "--output", mobile, "--mobile-weights", assets["mobile"])
        if stage in ("sd", "all"):
            if not (measurements / "sd/COMPLETE.json").exists():
                call(TOOLS / "r62_segmented_sd.py", *base_args, "--mobile", measurements / "mobile",
                     "--sd-snapshot", assets["sd_snapshot"], "--addon", work / "sd-addon/run_segmented_sd.py",
                     "--output", measurements / "sd")
            if not (measurements / "COMPLETE.json").exists():
                call("-m", "r29_repro.measure", "seal", "--store", measurements)
        if stage != "all":
            return

    scope = ("--context", context)
    if context == "all":
        scope = ("--all", "--reference-native", work / "svk-complete-estimator.json",
                 "--reference-crossed", work / "svk-weight-strength.json")
    refit = store / "refit"
    report = {"all": "ARCHIVE_PARITY.json", "f0_d3_o0": "WITNESS_PARITY.json"}
    complete = refit / report.get(context, "RESULTS.json")
    if complete.is_file():
        print(f"Verified result already exists: {complete}")
        return
    if refit.exists():
        raise FileExistsError(f"Preserve the partial refit, then move it to another directory before retrying: {refit}")
    call(TOOLS / "reproduce_svk_raw.py", "--r5-runtime", root, "--measurements", measurements,
         "--output", refit, *scope)


def jpeg(work, submission):
    assets = json.loads((work / "assets.json").read_text(encoding="utf-8"))
    submission = verify(submission.resolve(), SUBMISSION_SHA256)
    root, target = work / "r5", work / "common-jpeg"
    images = work / "all/images"

    def call(*options):
        subprocess.run([sys.executable, "-B", "-m", "research_tools.jpeg", "--r5-runtime", str(root),
                        "--run", str(target), *map(str, options)], cwd=TOOLS.parent, check=True)

    if not (target / "PLAN.json").exists():
        call("plan")
    # Both plans select the same 16,260 source files. The JPEG worker rechecks
    # image hashes against its own plan before it decodes each row.
    if not (images / "STAGING.json").is_file():
        images = target / "images"
        if not (images / "STAGING.json").is_file():
            call("stage", "--archives", *(Path(assets["archives"]) / name for name in DATASETS))
    if not (target / "MEASUREMENTS_COMPLETE.json").is_file():
        call("measure", "--images", images, "--sd-snapshot", assets["sd_snapshot"], "--mobile-weights", assets["mobile"])
    if not (target / "refit/RESULTS.json").is_file():
        call("refit")
    if not (target / "PAIRED_RESULTS.json").is_file():
        native = work / "all/refit"
        call("intervals", *(("--native-run", native) if (native / "ARCHIVE_PARITY.json").is_file() else ()))
    if not (target / "VERIFICATION.json").is_file():
        call("verify", "--submission", submission)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=lambda p: Path(p).resolve(), required=True)
    action = parser.add_subparsers(dest="action", required=True)
    init = action.add_parser("unpack", help="unpack the software and references from the paper archive")
    init.add_argument("submission", type=Path)
    get = action.add_parser("download", help="download the original image archives and encoder weights")
    for key in ("archives", "sd-snapshot", "mobile"):
        get.add_argument(f"--{key}", type=Path)
    get.add_argument("--encoders-only", action="store_true", help="for an already verified local image map")
    common = action.add_parser("jpeg", help="extract, refit, and verify the common-JPEG experiment")
    common.add_argument("submission", type=Path)
    for stage, help_text in (
        ("prepare", "check and prepare the original images"),
        ("mobile", "extract MobileNet features"),
        ("sd", "extract Stable Diffusion features and verify both sets of features"),
        ("fit", "fit, calibrate, evaluate, and check against the archived reference"),
        ("all", "stage images, extract features, fit, and evaluate"),
    ):
        command = action.add_parser(stage, help=help_text)
        command.add_argument("--context", default="all", help="all fits, or one saved split such as f0_d3_o0")
        if stage in ("prepare", "all"):
            command.add_argument("--local-images", type=Path, help="reuse an existing raw-file map, with per-image verification")
    args = parser.parse_args()
    if args.action == "unpack":
        unpack(args.submission, args.work)
    elif args.action == "download":
        acquire(args.work, args.archives, args.sd_snapshot, args.mobile, encoders_only=args.encoders_only)
    elif args.action == "jpeg":
        jpeg(args.work, args.submission)
    else:
        run(args.work, args.context, args.action, getattr(args, "local_images", None))


if __name__ == "__main__":
    main()
