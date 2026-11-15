"""Run the b15 CrossVerify reproduction from authenticated images to paired ROC estimates."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

from nrsw.reproduce import R62_SHA256
from inputs import HERE, design_file, digest, require
from refit import CORE, DINO, FLID, GLOBAL, NRSW, RESNET


def child(script, *options):
    subprocess.run([sys.executable, str(HERE / script), *(str(v) for v in options)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r62', 'images', 'r5-runtime', 'sd-snapshot', 'mobile-weights', 'dino-weights',
                 'resnet-weights', 'flid-source', 'historical-mean', 'shared-parent', 'output'):
        parser.add_argument('--' + name, type=Path, required=name != 'shared-parent')
    parser.add_argument('--images-layout', choices=('publisher', 'census'), default='publisher')
    parser.add_argument('--all-budgets', action='store_true')
    args = parser.parse_args()
    require(digest(args.r62) == R62_SHA256, 'Original R62 submission ZIP differs')
    for name, path in vars(args).items():
        if isinstance(path, Path) and name != 'output':
            require(path.exists(), f'Missing input {name}: {path}')
    output = args.output.resolve()
    output.mkdir(parents=True)
    measurements, models = output / 'measurements', output / 'models'
    picture = ('--images', args.images, '--images-layout', args.images_layout)
    methods = [NRSW, RESNET, DINO, FLID, GLOBAL]
    if args.shared_parent is not None:
        methods.append(CORE)
    import torch
    record = dict(python=platform.python_version(), operating_system=platform.platform(),
                  packages={name: version(name) for name in ('numpy', 'Pillow', 'torch', 'torchvision', 'timm',
                                                            'scikit-learn', 'diffusers', 'transformers')},
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                  methods=methods, r62_sha256=R62_SHA256, design_sha256=digest(design_file(args.all_budgets)),
                  source_sha256={p.name: digest(p) for p in sorted(HERE.glob('*.py'))})
    with (output / 'environment.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    budget_option = ('--all-budgets',) if args.all_budgets else ()
    for stage, phase in (('fit', 'train'), ('cal', 'cal'), ('test', 'test')):
        for name, weights in (('mobile', args.mobile_weights), ('sd', args.sd_snapshot), ('dino', args.dino_weights)):
            child('measure.py', name, phase, *picture, '--weights', weights,
                  '--r5-runtime', args.r5_runtime, '--output', measurements, *budget_option)
        for method in methods:
            options = ('--shared-parent', args.shared_parent) if method == CORE else ()
            child('refit.py', method, stage, *picture, '--features', measurements, '--output', models,
                  '--historical-mean', args.historical_mean, '--flid-source', args.flid_source,
                  '--weights', args.resnet_weights, *options, *budget_option)
    child('compare.py', '--r62', args.r62, '--run', models, '--output', output / 'comparison.json', *budget_option)


if __name__ == '__main__':
    main()
