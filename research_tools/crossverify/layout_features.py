"""Measure the two CrossVerify layouts using their frozen raster and batch contracts."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import struct

import numpy as np

from inputs import Pixels, require
from measure import DINO_SHA, native_r5


class LayoutPixels(Pixels):
    def image(self, row):
        from PIL import Image

        if self.archive is not None:
            raw = self.archive.read(row['member'])
        else:
            relative = Path(row['member'])
            path = self.path / relative
            if not path.is_file():
                path = self.path.joinpath(*relative.parts[1:])
            raw = path.read_bytes()
        require(sha256(raw).hexdigest() == row['payload_sha256'], 'Encoded publisher image differs')
        with Image.open(BytesIO(raw)) as source:
            image = source.convert('RGB')
        rgb = sha256(struct.pack('>II', *image.size) + image.tobytes()).hexdigest()
        require(list(image.size) == row['canvas'] and rgb == row['rgb_sha256'], 'Decoded publisher image differs')
        return image


def roster(design, family, phase):
    return sorted({r['id'] for context in design['contexts'] if context['family'] == family for r in context[phase]})


def local_rasters(image, boxes):
    from PIL import Image

    for left, top, right, bottom in boxes:
        coefficients = ((right-left)/224., 0., float(left), 0., (bottom-top)/224., float(top), 0., 0.)
        yield np.array(image.transform((224, 224), Image.Transform.PERSPECTIVE, coefficients,
                                       resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)))


def mobile(args, design, family, ids, images):
    from nrsw.encoders import hash_mobile
    from nrsw.field_classifiers import ink_descriptor

    factory, _, guard, pixels7 = native_r5(args.r5 / 'package', design)
    encoder = factory(args.weights / 'mobilenet_v3_small-047dcff4.pth', guard())
    boxes = design['layouts'][family]['field_boxes_xyxy_edges']
    selected, values = set(ids), {}
    try:
        for chunk in design['mobile_chunks'][family]:
            if not selected.intersection(chunk):
                continue
            queue = [np.zeros((224, 224, 3), np.uint8)] * (3 * len(chunk))
            globals_ = [np.zeros((7, 224, 224, 3), np.uint8)] * len(chunk)
            pixels = {}
            for i, sid in enumerate(chunk):
                if sid in selected:
                    with images.image(design['images'][sid]) as image, image.convert('L') as gray:
                        queue[3*i:3*i+3] = local_rasters(image, boxes)
                        globals_[i] = pixels7(image)
                        pixels[i] = np.stack([ink_descriptor(np.asarray(gray.crop(box))) for box in boxes])
            raw = encoder.encode_raw([queue[i % len(queue)] for i in range(224)])[:len(queue)]
            local, grid = hash_mobile(raw).reshape(len(chunk), 3, 512), encoder.global_grid(globals_)
            for i, pixel in pixels.items():
                values[chunk[i]] = (raw[3*i:3*i+3].copy(), local[i].copy(), grid[i].copy(), pixel)
    finally:
        encoder.close()
    require(values.keys() == selected, 'Incomplete Mobile and pixel coverage')
    return {key: np.stack(matrix) for key, matrix in zip(('raw576', 'local512', 'grid512', 'pixel'),
             zip(*(values[sid] for sid in ids), strict=True), strict=True)}


def sd(args, design, family, ids, images):
    _, factory, guard, _ = native_r5(args.r5 / 'package', design)
    encoder = factory(args.sd_snapshot, guard())
    values = []
    try:
        for sid in ids:
            with images.image(design['images'][sid]) as image:
                values.append(encoder.one(image))
            if len(values) % 100 == 0:
                print(f'{family} SD {len(values)}/{len(ids)}', flush=True)
    finally:
        encoder.close()
    return {'sd512': np.stack(values)}


def dino_geometry(layout):
    width, height = layout['canvas']
    scale = min(576 / width, 256 / height)
    size = round(width * scale), round(height * scale)
    left, top = (576-size[0])//2, (256-size[1])//2
    sx, sy = size[0]/width, size[1]/height
    y, x = np.mgrid[0:256:16, 0:576:16]
    weights = []
    for x0, y0, x1, y1 in [(0, 0, width, height), *layout['field_boxes_xyxy_edges']]:
        a, b, c, d = left + x0*sx, top + y0*sy, left + x1*sx, top + y1*sy
        area = np.maximum(0, np.minimum(x+16, c)-np.maximum(x, a))
        area *= np.maximum(0, np.minimum(y+16, d)-np.maximum(y, b))
        require(np.isclose(area.sum(), (c-a)*(d-b), atol=1e-7, rtol=0), 'DINO patch geometry differs')
        weights.append((area/area.sum()).reshape(-1))
    return size, (left, top), np.asarray(weights, dtype=np.float32)


def dino(args, design, family, ids, images):
    from PIL import Image
    import timm
    import torch
    from safetensors.torch import load_file
    from inputs import digest

    weights = args.weights / 'model.safetensors'
    require(timm.__version__ == '1.0.31' and digest(weights) == DINO_SHA, 'Pinned DINO inputs differ')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, num_classes=0,
                              img_size=(256, 576))
    model.load_state_dict(load_file(weights, device='cpu'), strict=True)
    model.eval().requires_grad_(False).to(device)
    require(model.num_prefix_tokens == 5 and model.num_features == 384, 'DINO patch encoder differs')
    size, offset, weight = dino_geometry(design['layouts'][family])
    pool = torch.from_numpy(weight).to(device)
    center = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    scale = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    values = []
    with torch.inference_mode():
        for start in range(0, len(ids), 12):
            pixels = []
            for sid in ids[start:start+12]:
                with images.image(design['images'][sid]) as image, Image.new('RGB', (576, 256)) as padded:
                    padded.paste(image.resize(size, Image.Resampling.BICUBIC), offset)
                    pixels.append(np.asarray(padded, dtype=np.uint8).transpose(2, 0, 1))
            batch = torch.from_numpy(np.stack(pixels)).to(device, dtype=torch.float32) / 255
            features = model.forward_features((batch-center)/scale)[:, 5:]
            values.append((pool @ features).flatten(1).cpu().numpy())
    return {'dense1536': np.concatenate(values)}
