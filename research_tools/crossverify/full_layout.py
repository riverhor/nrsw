"""Train the five detector comparisons on both CrossVerify layouts."""

import argparse
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import time_ns

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np

from inputs import HERE, digest, require, save_arrays
from layout_compare import compare
from layout_features import LayoutPixels, dino, mobile, roster, sd
from layout_fit import model_inventory, stage
from nrsw.reproduce import _r62_evidence
from publisher import acquire_snapshot, child, install_runtime
from refit import write_json


DESIGN_PIN = 'e4da942d221dff398e3a79e3cef35cb48fbf208d45e025572213e9609bdf5bef'
FAMILIES = ('family_a', 'family_b')
PHASES = ('train', 'cal', 'test')
ENCODERS = ('mobile', 'sd', 'dino')


def read_design():
    path = HERE / 'layout_design.json.gz'
    require(digest(path) == DESIGN_PIN, 'Frozen full-layout design differs')
    design = json.loads(gzip.decompress(path.read_bytes()))
    require(tuple(design['layouts']) == FAMILIES and len(design['contexts']) == 72
            and len(design['images']) == 3880, 'Full-layout design is incomplete')
    for family in FAMILIES:
        require(len([c for c in design['contexts'] if c['family'] == family]) == 36, 'Full-layout context count differs')
        for c in (r for r in design['contexts'] if r['family'] == family):
            groups = [{r['component'] for r in c[phase]} for phase in PHASES]
            require([len(c[p]) for p in PHASES] == [115,300,320] and list(map(len,groups)) == [115,300,80]
                    and all(not groups[i] & groups[j] for i,j in ((0,1),(0,2),(1,2))), 'Full-layout role separation differs')
    return design


def verify_stage(path, previous):
    value = json.loads(path.read_bytes())
    require(value['design_sha256'] == DESIGN_PIN and value.get('previous') == previous
            and all(digest(path.parent / name) == pin for name,pin in value['files'].items()),
            f'Changed full-layout stage: {path.name}')
    return digest(path)


def phase_path(root, phase):
    return root / (phase.upper() + '.json')


def previous_stage(root, phase):
    previous = None
    for prior in PHASES[:PHASES.index(phase)]:
        previous = verify_stage(phase_path(root,prior), previous)
    return previous


def verify_inputs(args):
    with _r62_evidence(args.r62):
        pass
    base = args.work / 'crossverify'
    base.mkdir(parents=True, exist_ok=True)
    args.r5 = args.r5 or args.work / 'r5'
    install_runtime(args.r62, args.r5)
    args.images = args.images or base / 'crossverify_dataset_v1.zip'
    if not args.images.exists():
        child('download_inputs.py', 'parts', base / 'parts')
        child('assemble_dataset.py', base / 'parts', args.images)
    args.weights = base / 'public-weights'
    child('download_inputs.py', 'weights', args.weights)
    args.sd_snapshot = acquire_snapshot(args.sd_snapshot, args.work)
    return args


def execute(args, design):
    import torch

    root = args.work / 'crossverify' / 'full-layout'
    features = root / 'measurements'
    root.mkdir(exist_ok=True)
    features.mkdir(exist_ok=True)
    models = root / 'models'
    models.mkdir(exist_ok=True)
    torch.set_num_interop_threads(1)
    bindings = dict(design_sha256=DESIGN_PIN, python=platform.python_version(),
                    packages={k:version(k) for k in ('numpy','Pillow','torch','torchvision','timm','scikit-learn','diffusers','transformers')},
                    gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    sources={name:digest(HERE/name) for name in ('full_layout.py','layout_features.py','layout_fit.py',
                             'layout_flid.py','layout_compare.py','neural.py')})
    environment = root / 'environment.json'
    if environment.exists():
        require(json.loads(environment.read_bytes()) == bindings, 'Full-layout runtime changed during restart')
    else:
        write_json(environment, bindings)
    for phase in PHASES:
        previous = previous_stage(root, phase)
        marker = phase_path(root,phase)
        if marker.exists():
            verify_stage(marker,previous)
            print(f'Verified complete {phase} stage', flush=True)
            continue
        for family in FAMILIES:
            for encoder in ENCODERS:
                path = features / f'{family}-{phase}-{encoder}.npz'
                witness = path.with_suffix('.json')
                if witness.exists():
                    seal = json.loads(witness.read_bytes())
                    require(seal == dict(design_sha256=DESIGN_PIN, previous=previous, sha256=digest(path)),
                            f'Changed feature packet {path.name}')
                    continue
                if path.exists():
                    path.rename(path.with_name(path.name + f'.unsealed-{time_ns()}'))
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--work',str(args.work),
                     '--r62',str(args.r62),'--images',str(args.images),'--sd-snapshot',str(args.sd_snapshot),
                     '--r5',str(args.r5),'--worker',phase,family,encoder],check=True)
        with LayoutPixels(args.images,'publisher') as images:
            for family in FAMILIES:
                ids = roster(design, family, phase)
                stage(args, design, family, phase, ids, images, features, models)
        files = {p.relative_to(root).as_posix(): digest(p) for p in features.glob(f'*-{phase}-*.npz')}
        files.update({p.relative_to(root).as_posix(): digest(p) for p in features.glob(f'*-{phase}-*.json')})
        if phase == 'train':
            files.update({'models/'+name: sha for name,sha in model_inventory(models).items()})
        else:
            files.update({f'models/{family}-{phase}.npz': digest(models/f'{family}-{phase}.npz') for family in FAMILIES})
        write_json(marker,dict(design_sha256=DESIGN_PIN,previous=previous,files=files))
    previous = previous_stage(root, 'test')
    verify_stage(phase_path(root,'test'),previous)
    report = root / 'comparison.json'
    fresh = compare(args.r62,design,models)
    if report.exists():
        require(json.loads(report.read_bytes()) == fresh, 'Changed full-layout comparison receipt')
    else:
        write_json(report, fresh)
    print(root / 'comparison.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', '--r62', dest='r62', metavar='ZIP', type=Path, required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--images',type=Path,help='Existing CrossVerify publisher ZIP or extracted tree')
    parser.add_argument('--sd-snapshot',type=Path,help='Existing pinned Stable Diffusion snapshot')
    parser.add_argument('--r5',metavar='DIR',type=Path,help='Existing unpacked reproduction directory')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare-only',action='store_true',help='Authenticate all 3,880 selected images and pinned assets with no fit')
    actions.add_argument('--check-archived',type=Path,metavar='JSON',help='Recompute results for all five methods from saved scores')
    actions.add_argument('--worker',nargs=3,help=argparse.SUPPRESS)
    args = parser.parse_args()
    design = read_design()
    args.work = args.work.resolve()
    if args.check_archived is not None:
        write_json(args.check_archived,compare(args.r62,design))
        print(args.check_archived)
        return
    if args.worker:
        phase,family,encoder = args.worker
        require(phase in PHASES and family in FAMILIES and encoder in ENCODERS, 'Invalid full-layout measurement stage')
        root = args.work / 'crossverify' / 'full-layout'
        previous = previous_stage(root,phase)
        args.weights = root.parent / 'public-weights'
        ids = roster(design,family,phase)
        with LayoutPixels(args.images,'publisher') as images:
            values = {'mobile':mobile,'sd':sd,'dino':dino}[encoder](args,design,family,ids,images)
        require(all(len(x)==len(ids) and np.isfinite(x).all() for x in values.values()), 'Incomplete measurement stage')
        path = root/'measurements'/f'{family}-{phase}-{encoder}.npz'
        save_arrays(path,ids=ids,**values)
        write_json(path.with_suffix('.json'),dict(design_sha256=DESIGN_PIN,previous=previous,sha256=digest(path)))
        return
    verify_inputs(args)
    if args.prepare_only:
        with LayoutPixels(args.images,'publisher') as images:
            for row in design['images'].values():
                with images.image(row):
                    pass
        print('Verified 3,880 selected CrossVerify images, both layouts and all pinned inputs')
    else:
        execute(args,design)


if __name__ == '__main__':
    main()
