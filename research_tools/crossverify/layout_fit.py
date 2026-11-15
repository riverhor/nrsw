"""Fit and score the five full-layout CrossVerify methods in separate stages."""

import json
from time import time_ns
from types import SimpleNamespace

import numpy as np

from inputs import digest, require, save_arrays
from layout_flid import fit_flid_layout, score_flid_layout
from neural import fit_dino, fit_resnet, resnet_model, resnet_pixels, score_dino, score_resnet, torch_runtime
from refit import historical_mean, write_json


METHODS = ('NRSW', 'NRSW with field classifiers', 'ResNet-18', 'DINOv3', 'FLiD')
RANKS = np.array([300, 292, 279])


def local_batch(ids, components, values, design, family):
    from nrsw.kernel import FeatureBatch, Layout

    schema = Layout(family, tuple(design['layouts'][family]['field_names']), 512, 512, 'crossverify-r62-full-layout')
    return FeatureBatch(schema, tuple(ids), tuple(components), values['sd512'], values['local512'],
                        values['grid512'], np.ones(len(ids), dtype=bool))


def stage_values(root, phase, family, expected):
    values = {}
    for name in ('mobile', 'sd', 'dino'):
        path = root / f'{family}-{phase}-{name}.npz'
        with np.load(path, allow_pickle=False) as packet:
            require(packet['ids'].tolist() == expected, f'Sealed descriptor roster differs: {path.name}')
            values.update({key: packet[key] for key in packet.files if key != 'ids'})
    return values


def select(values, rows, index):
    ids = [r['id'] for r in rows]
    positions = [index[sid] for sid in ids]
    return ids, [r['component'] for r in rows], positions, {key: block[positions] for key, block in values.items()}


def fit_one(args, context, design, values, rgb, index, reference, directory):
    from nrsw.field_classifiers import fit_field_classifiers
    from nrsw.kernel import TrainBatch, fit_joint
    from nrsw.state import export_primary
    import torch

    rows, family = context['train'], context['family']
    ids, components, positions, data = select(values, rows, index)
    batch = local_batch(ids, components, data, design, family)
    training = TrainBatch(batch, tuple(r['role'] for r in rows), np.array([r['fold'] for r in rows], dtype=np.int64),
                          tuple(tuple(r['candidates']) for r in rows))
    require(len(rows) == 115 and training.labels.sum() == 15, 'Incorrect full-layout TRAIN roster')
    original, _ = fit_joint(training, reference)
    export_primary(original, directory / 'nrsw')
    extended = fit_field_classifiers(training, original, data['pixel'], np.ones((len(rows), 3), dtype=bool))
    export_primary(extended, directory / 'extended')
    save_arrays(directory / 'dino.npz', **fit_dino(data['dense1536'], training.labels))
    save_arrays(directory / 'flid.npz', **fit_flid_layout(data['raw576'], context, args.weights / 'flid_models.py', 'cuda:0'))
    resnet_args = SimpleNamespace(device='cuda:0', weights=args.weights / 'resnet18-f37072fd.pth')
    resnet = fit_resnet(resnet_args, rgb[positions], training.labels)
    torch.save({key: tensor.detach().cpu() for key, tensor in resnet.state_dict().items()}, directory / 'resnet.pt')


def score_one(args, context, design, phase, values, rgb, index, directory):
    from nrsw.state import import_primary

    family = context['family']
    ids, components, positions, data = select(values, context[phase], index)
    batch = local_batch(ids, components, data, design, family)
    original, extended = (import_primary(directory / name) for name in ('nrsw', 'extended'))
    torch, device = torch_runtime('cuda:0', tf32=True)
    resnet = resnet_model(torch).to(device)
    resnet.load_state_dict(torch.load(directory / 'resnet.pt', map_location='cpu', weights_only=True))
    with np.load(directory / 'dino.npz', allow_pickle=False) as dino, np.load(directory / 'flid.npz', allow_pickle=False) as flid:
        scores = np.stack((original.score(batch), extended.score(batch, data['pixel'], np.ones((len(ids), 3), bool)),
                           score_resnet(resnet, rgb[positions], device), score_dino(data['dense1536'], dino),
                           score_flid_layout(data['raw576'], family, args.weights / 'flid_models.py', flid, 'cuda:0')))
    require(scores.shape == (5, len(ids)) and np.isfinite(scores).all(), 'Incomplete full-layout score vector')
    return scores


def stage(args, design, family, phase, ids, images, features, root):
    contexts = [r for r in design['contexts'] if r['family'] == family]
    values = stage_values(features, phase, family, ids)
    rgb = resnet_pixels([design['images'][sid] for sid in ids], images)
    index = {sid: i for i, sid in enumerate(ids)}
    if phase == 'train':
        reference = historical_mean(args.r5 / 'metadata' / 'HISTORICAL_MEAN_FP64.npy')
        for context in contexts:
            folder = root / context['id']
            receipt = folder / 'COMPLETE.json'
            if receipt.exists():
                files = json.loads(receipt.read_bytes())['files']
                require(all(digest(folder/name) == sha for name,sha in files.items()), 'Changed fitted context')
                continue
            if folder.exists():
                folder.rename(folder.with_name(folder.name + f'.unsealed-{time_ns()}'))
            folder.mkdir()
            fit_one(args, context, design, values, rgb, index, reference, folder)
            files = {p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
            write_json(receipt,dict(files=files))
            print(f'{context["id"]} models fitted', flush=True)
        return
    vectors = []
    for context in contexts:
        vectors.append(score_one(args, context, design, phase, values, rgb, index, root / context['id']))
    array = np.stack(vectors)
    scores = {'scores': array, 'context_ids': [c['id'] for c in contexts],
              'ids': [[r['id'] for r in context[phase]] for context in contexts], 'methods': METHODS}
    if phase == 'cal':
        scores['cuts'] = np.sort(array, axis=2)[:, :, RANKS - 1]
    path = root / f'{family}-{phase}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            require(all(np.array_equal(saved[key], value) for key,value in scores.items()), 'Changed full-layout scores')
    else:
        save_arrays(path, **scores)


def model_inventory(root):
    return {p.relative_to(root).as_posix(): digest(p) for folder in sorted(root.glob('family_?_f*_d*'))
            if folder.is_dir() and '.' not in folder.name for p in sorted(folder.rglob('*')) if p.is_file()}
