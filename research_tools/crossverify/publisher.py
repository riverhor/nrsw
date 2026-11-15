"""Download the published data and rerun the CrossVerify label-budget comparison."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from nrsw.encoders import SD_ASSET_SHA256, SD_MODEL_ID, SD_REVISION
from nrsw.reproduce import RAW_RUNTIME, _r62_evidence
from build_design import PINS
from inputs import HERE, Pixels, design, digest, require
from refit import HISTORICAL_SHA, PARENT_SHA


MOBILE = 'mobilenet_v3_small-047dcff4.pth'


def child(script, *options):
    subprocess.run([sys.executable, str(HERE / script), *(str(v) for v in options)], check=True)


def install_runtime(submission, root):
    if not root.exists():
        with _r62_evidence(submission) as evidence:
            payload = evidence.read(RAW_RUNTIME)
        require(sha256(payload).hexdigest() == PINS['r5'], 'Checksum mismatch in the nested reproduction archive')
        with TemporaryDirectory(dir=root.parent) as temp, ZipFile(BytesIO(payload)) as archive:
            staged = Path(temp) / 'r5'
            archive.extractall(staged)
            staged.rename(root)
    for name, expected in design()['native_r5_sha256'].items():
        require(digest(root / 'package' / 'r29_repro' / name) == expected, f'Changed R5 source {name}')
    mean = root / 'metadata' / 'HISTORICAL_MEAN_FP64.npy'
    require(digest(mean) == HISTORICAL_SHA, 'Changed R5 reference mean')
    return mean


def acquire_snapshot(path, work):
    assets = dict(SD_ASSET_SHA256)
    assets = {name: value for name, value in assets.items()
              if not name.endswith('.safetensors') or name.endswith('.fp16.safetensors')}
    assets['model_index.json'] = 'e2f6f22e274374010aec30c79eb2d6e53fff4a36ac0225523a92cb2d41a83347'
    if path is None:
        from huggingface_hub import snapshot_download

        path = work / 'sd15'
        if not all((path / name).is_file() for name in assets):
            snapshot_download(SD_MODEL_ID, revision=SD_REVISION, local_dir=path, allow_patterns=list(assets))
    for name, expected in assets.items():
        require(digest(path / name) == expected, f'Changed Stable Diffusion asset {name}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', '--r62', dest='r62', metavar='ZIP', type=Path, required=True, help='Paper submission archive')
    parser.add_argument('--work', type=Path, required=True, help='Working directory for assets and results')
    parser.add_argument('--shared-parent', type=Path, help='Optional historical checkpoint from the authors')
    parser.add_argument('--images', type=Path, help='Prepared CrossVerify publisher ZIP or extracted tree')
    parser.add_argument('--sd-snapshot', type=Path, help='Prepared pinned SD1.5 FP16 snapshot')
    parser.add_argument('--r5', metavar='DIR', type=Path, help='Unpacked reproduction directory containing package and metadata')
    parser.add_argument('--prepare-only', action='store_true', help='Acquire and verify inputs without running CUDA')
    parser.add_argument('--all-budgets', action='store_true', help='Include the 60-, 120- and 240-forgery fits')
    args = parser.parse_args()
    with _r62_evidence(args.r62):
        pass
    if args.shared_parent is not None:
        require(digest(args.shared_parent) == PARENT_SHA, 'Historical shared-core parent differs')
    base = args.work.resolve() / 'crossverify'
    base.mkdir(parents=True, exist_ok=True)
    output = base / ('run-all-budgets' if args.all_budgets else 'run')
    if not args.prepare_only:
        require(not output.exists(), f'Output already exists: {output}')
    runtime = args.r5 or args.work.resolve() / 'r5'
    if args.r5 is not None:
        require(runtime.is_dir(), f'Extract the reproduction files first: {runtime}')
    mean = install_runtime(args.r62, runtime)
    images = args.images or base / 'crossverify_dataset_v1.zip'
    if args.images is None and not images.exists():
        child('download_inputs.py', 'parts', base / 'parts')
        child('assemble_dataset.py', base / 'parts', images)
    experiment = design(args.all_budgets)
    with Pixels(images, 'publisher') as source:
        for stage in ('train', 'cal', 'test'):
            for row in experiment[stage]:
                with source.image(row):
                    pass
    weights = base / 'public-weights'
    child('download_inputs.py', 'weights', weights)
    snapshot = acquire_snapshot(args.sd_snapshot, args.work.resolve())
    if args.prepare_only:
        count = sum(len(experiment[role]) for role in ('train', 'cal', 'test'))
        print(f'Verified {count} published images, weights and reproduction files for {output}')
        return
    optional = ('--shared-parent', args.shared_parent) if args.shared_parent is not None else ()
    budget_option = ('--all-budgets',) if args.all_budgets else ()
    child('run.py', '--r62', args.r62, '--images', images, '--r5-runtime', runtime / 'package',
          '--historical-mean', mean, '--sd-snapshot', snapshot, '--mobile-weights', weights / MOBILE,
          '--resnet-weights', weights / 'resnet18-f37072fd.pth', '--dino-weights', weights / 'model.safetensors',
          '--flid-source', weights / 'flid_models.py', '--output', output, *optional, *budget_option)


if __name__ == '__main__':
    main()
