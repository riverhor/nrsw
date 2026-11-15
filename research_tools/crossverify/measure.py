"""Extract the selected image descriptors in the archived native compute shapes."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np

from inputs import (CANVAS, REGIONS, Pixels, design, digest, feature_file, require, roster, save_arrays)


def native_r5(runtime, experiment):
    root = Path(runtime).resolve(strict=True)
    bindings = experiment['native_r5_sha256']
    for name, expected in bindings.items():
        require(digest(root / 'r29_repro' / name) == expected, f'R5 runtime file differs: {name}')
    sys.path.insert(0, str(root))
    from r29_repro.native_encoders import NativeMobile, NativeSD, ResourceGuard, pixels7
    return NativeMobile, NativeSD, ResourceGuard, pixels7


def regions(image):
    from PIL import Image

    for left, top, right, bottom in REGIONS:
        coefficients = ((right-left)/224., 0., float(left), 0., (bottom-top)/224., float(top), 0., 0.)
        yield np.array(image.transform((224, 224), Image.Transform.PERSPECTIVE, coefficients,
                                       resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)))


def mobile(args, experiment, rows, images):
    from nrsw.encoders import hash_mobile

    NativeMobile, _, ResourceGuard, pixels7 = native_r5(args.r5_runtime, experiment)
    encoder = NativeMobile(args.weights, ResourceGuard())
    selected = {row['id']: row for row in rows}
    found = {}
    try:
        for chunk in experiment['mobile_chunks']:
            if not selected.keys() & set(chunk):
                continue
            blank = np.zeros((224, 224, 3), np.uint8)
            queue = [blank] * (3 * len(chunk))
            global_pixels = [np.zeros((7, 224, 224, 3), np.uint8)] * len(chunk)
            for i, identity in enumerate(chunk):
                if identity in selected:
                    with images.image(selected[identity]) as image:
                        queue[3*i:3*i+3] = regions(image)
                        global_pixels[i] = pixels7(image)
            raw = encoder.encode_raw([queue[i % len(queue)] for i in range(224)])[:len(queue)]
            local = hash_mobile(raw).reshape(len(chunk), 3, 512)
            grid = encoder.global_grid(global_pixels)
            for i, identity in enumerate(chunk):
                if identity in selected:
                    found[identity] = (raw[3*i:3*i+3].copy(), local[i].copy(), grid[i].copy())
    finally:
        encoder.close()
    require(found.keys() == selected.keys(), 'Incomplete original Mobile chunk coverage')
    raw, local, grid = zip(*(found[row['id']] for row in rows), strict=True)
    return {'raw576': np.stack(raw), 'local512': np.stack(local), 'grid512': np.stack(grid)}


def sd(args, experiment, rows, images):
    _, NativeSD, ResourceGuard, _ = native_r5(args.r5_runtime, experiment)
    encoder = NativeSD(args.weights, ResourceGuard())
    values = []
    try:
        for index, row in enumerate(rows, 1):
            with images.image(row) as image:
                values.append(encoder.one(image))
            if index % 50 == 0 or index == len(rows):
                print(f'SD {args.phase}: {index}/{len(rows)}', flush=True)
    finally:
        encoder.close()
    return {'sd512': np.stack(values)}


DINO_SHA = '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'


def dino(args, experiment, rows, images):
    from PIL import Image
    import timm
    import torch
    from safetensors.torch import load_file

    require(timm.__version__ == '1.0.31' and digest(args.weights) == DINO_SHA, 'DINO checkpoint or timm differs')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, num_classes=0,
                              img_size=(256, 576))
    model.load_state_dict(load_file(args.weights, device='cpu'), strict=True)
    model.eval().requires_grad_(False).to(device)
    require(model.num_prefix_tokens == 5 and model.num_features == 384, 'DINO token geometry differs')
    resized = (566, 256)
    offset = (5, 0)
    sx, sy = np.divide(resized, CANVAS)
    y, x = np.mgrid[0:256:16, 0:576:16]
    weights = []
    for l, t, r, b in ((0, 0, *CANVAS), *REGIONS):
        x0, y0, x1, y1 = offset[0]+l*sx, t*sy, offset[0]+r*sx, b*sy
        area = np.maximum(0, np.minimum(x + 16, x1) - np.maximum(x, x0))
        area *= np.maximum(0, np.minimum(y + 16, y1) - np.maximum(y, y0))
        weights.append((area / area.sum()).reshape(-1))
    pool = torch.tensor(np.stack(weights), dtype=torch.float32, device=device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    scale = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), 12):
            pixels = []
            for row in rows[start:start+12]:
                with images.image(row) as image, Image.new('RGB', (576, 256)) as padded:
                    padded.paste(image.resize(resized, Image.Resampling.BICUBIC), offset)
                    pixels.append(np.asarray(padded, dtype=np.uint8).transpose(2, 0, 1))
            batch = torch.from_numpy(np.stack(pixels)).to(device, dtype=torch.float32) / 255
            features = model.forward_features((batch - mean) / scale)[:, 5:]
            values.append((pool @ features).flatten(1).cpu().numpy())
    return {'dense1536': np.concatenate(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('encoder', choices=('mobile', 'sd', 'dino'))
    parser.add_argument('phase', choices=('train', 'cal', 'test'))
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--images-layout', choices=('publisher', 'census'), default='publisher')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--r5-runtime', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-budgets', action='store_true')
    args = parser.parse_args()
    if args.encoder in ('mobile', 'sd') and args.r5_runtime is None:
        parser.error('Mobile and SD need --r5-runtime pointing to the extracted R62 R5 runtime')
    if args.encoder in ('mobile', 'sd') and args.device != 'cuda:0':
        parser.error('The native Mobile and SD route uses cuda:0')
    args.output.mkdir(parents=True, exist_ok=True)
    path = feature_file(args.output, args.encoder, args.phase)
    require(not path.exists(), f'Output already exists: {path}')
    experiment = design(args.all_budgets)
    rows = roster(experiment, args.phase)
    with Pixels(args.images, args.images_layout) as images:
        arrays = {'mobile': mobile, 'sd': sd, 'dino': dino}[args.encoder](args, experiment, rows, images)
    require(all(np.isfinite(array).all() and len(array) == len(rows) for array in arrays.values()),
            'Incomplete/nonfinite descriptor output')
    save_arrays(path, ids=[row['id'] for row in rows], input_kind='native_raw_image', **arrays)
    with path.with_suffix('.json').open('x', encoding='utf-8') as handle:
        json.dump({'file_sha256': digest(path), 'method': args.encoder, 'phase': args.phase,
                   'rows': len(rows), 'torch': version('torch'), 'torchvision': version('torchvision'),
                   'numpy': np.__version__, 'pillow': version('Pillow'), 'device': args.device}, handle, indent=2)
        handle.write('\n')
    print(f'{path} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    main()
