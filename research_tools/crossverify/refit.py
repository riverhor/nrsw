"""Fit frozen R62 CrossVerify contexts, then seal CAL and TEST separately."""

import argparse
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np

from inputs import FIELDS, Pixels, context_ids, design, design_file, digest, feature_file, require, save_arrays
from neural import (fit_dino, fit_flid, fit_resnet, flid_factory, resnet_model, resnet_pixels,
                    score_dino, score_flid, score_resnet, torch_runtime)


METHODS = ('NRSW', 'CrossVerify_ResNet18', 'DINOv3_frozen_dense', 'FLiD_text_shared_regions',
           'GLOBAL_ONLY', 'FROZEN_SHARED_CORE')
NRSW, RESNET, DINO, FLID, GLOBAL, CORE = METHODS
HISTORICAL_SHA = '1276145aec477bdde03ccff0bb11a4ec6391f04d310eb703b9a96f7caf27fcb1'
PARENT_SHA = '8cb8acc92d4fe1a144622ef500bf8a57a72b7ad37cab1283283547697e353947'


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def features(args, phase, rows):
    ids = [r['id'] for r in rows]
    method = args.method
    by_encoder = {
        NRSW: {'mobile': ('local512', 'grid512'), 'sd': ('sd512',)},
        GLOBAL: {'mobile': ('grid512',), 'sd': ('sd512',)},
        CORE: {'mobile': ('grid512',), 'sd': ('sd512',)},
        FLID: {'mobile': ('raw576',)}, DINO: {'dino': ('dense1536',)},
    }
    if method == RESNET:
        with Pixels(args.images, args.images_layout) as images:
            return resnet_pixels(rows, images), {'kind': 'native_raw_image'}
    values = {}
    binding = {}
    kinds = set()
    for encoder, keys in by_encoder[method].items():
        path = feature_file(args.features, encoder, phase)
        binding[encoder] = digest(path)
        with np.load(path, allow_pickle=False) as packet:
            kinds.add(str(packet['input_kind']))
            require(packet['ids'].tolist() == ids, f'Incorrect descriptor row order: {path}')
            for key in keys:
                array = packet[key]
                expected = dict(dense1536=(1536,), raw576=(3, 576), local512=(3, 512), grid512=(512,), sd512=(512,))
                require(array.dtype == np.float32 and array.shape == (len(rows), *expected[key])
                        and np.isfinite(array).all(), f'Invalid descriptor shape or value: {key}')
                values[key] = array
    require(len(kinds) == 1, 'Mixed archived and newly extracted descriptor sources')
    binding['kind'] = kinds.pop()
    if method == DINO:
        return values['dense1536'], binding
    if method == FLID:
        return values['raw576'], binding
    if method in (GLOBAL, CORE):
        return np.concatenate((values['sd512'], values['grid512']), axis=1), binding
    from nrsw.kernel import FeatureBatch, Layout

    layout = Layout('family_b', FIELDS, 512, 512, 'crossverify-r62-native')
    return FeatureBatch(layout, tuple(ids), tuple(r['component'] for r in rows), values['sd512'],
                        values['local512'], values['grid512'], np.ones(len(ids), dtype=bool)), binding


def historical_mean(path):
    require(digest(path) == HISTORICAL_SHA, 'Historical mean differs')
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float64 and value.shape == (1024,) and np.isfinite(value).all(), 'Invalid historical mean')
    return value


def global_frame(args, data, offset):
    aligned = (data.astype(np.float64) - offset).astype(np.float32)
    if args.method == GLOBAL:
        return aligned.astype(np.float64)
    import torch
    from torch.nn import functional as f

    torch.set_num_threads(2)
    require(digest(args.shared_parent) == PARENT_SHA, 'Historical shared-core parent differs')
    payload = torch.load(args.shared_parent, weights_only=True, map_location='cpu')
    require(payload['representation'] == 'sd_grid' and payload['key'] == 't0_m2203_parent', 'Shared parent metadata differs')
    state = payload['model_state']
    keys = ('core.linear.weight', 'core.linear.bias', 'core.normalization.weight', 'core.normalization.bias')
    tensors = [state[key] for key in keys]
    require(all(t.dtype == torch.float32 and torch.isfinite(t).all() and tuple(t.shape) == shape
                for t, shape in zip(tensors, ((128, 1024), (128,), (128,), (128,)), strict=True)), 'Invalid core state')
    weight, bias, norm_weight, norm_bias = tensors
    with torch.inference_mode():
        hidden = f.gelu(f.linear(torch.from_numpy(aligned), weight, bias), approximate='none')
        return f.layer_norm(hidden, (128,), norm_weight, norm_bias, 1e-5).numpy().astype(np.float64)


def fit(args, experiment, output):
    from nrsw.kernel import TrainBatch, fit_head, fit_joint
    from nrsw.state import export_primary

    rows = experiment['train']
    values, receipt = features(args, 'train', rows)
    positions = {row['id']: i for i, row in enumerate(rows)}
    method = args.method
    if method in (NRSW, GLOBAL, CORE):
        reference = historical_mean(args.historical_mean)
    if method in (GLOBAL, CORE):
        anchors = [positions[r['id']] for r in experiment['contexts'][0]['train'] if r['role'] == 'N']
        offset = values[anchors].astype(np.float64).mean(axis=0) - reference
        values = global_frame(args, values, offset)
        save_arrays(output / 'offset.npz', offset=offset)
    for context in experiment['contexts']:
        train = context['train']
        ids = [positions[r['id']] for r in train]
        labels = np.array([int(r['role'] == 'F') for r in train])
        budget = int(context['id'].split('_')[0][1:])
        require(len(ids) == 100 + budget and labels.sum() == budget, 'Incorrect TRAIN roster')
        path = output / context['id']
        if method == NRSW:
            batch = TrainBatch(values.subset(ids), tuple(r['role'] for r in train),
                               np.array([r['fold'] for r in train], dtype=np.int64),
                               tuple(tuple(r['candidates']) for r in train))
            model, _ = fit_joint(batch, reference)
            export_primary(model, path)
        elif method in (GLOBAL, CORE):
            model = fit_head(values[ids], labels)
            save_arrays(path.with_suffix('.npz'), mean=model.mean, rms=model.rms, w=model.w, b=model.b)
        elif method == DINO:
            save_arrays(path.with_suffix('.npz'), **fit_dino(values[ids], labels))
        elif method == FLID:
            model = fit_flid(args, values[ids], context)
            save_arrays(path.with_suffix('.npz'), **{name: p.detach().cpu().numpy() for name, p in model.state_dict().items()})
        elif method == RESNET:
            import torch
            model = fit_resnet(args, values[ids], labels)
            torch.save({name: v.detach().cpu() for name, v in model.state_dict().items()}, path.with_suffix('.pt'))
        print(f'{method} fitted {context["id"]}', flush=True)
    files = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'FIT.json', {'method': method, 'design_sha256': digest(design_file(args.all_budgets)),
                                     'features': receipt, 'contexts': list(context_ids(experiment)), 'model_files': files})


def score(args, experiment, output, phase):
    from nrsw.kernel import AnalyticHead
    from nrsw.state import import_primary

    fit_path = output / 'FIT.json'
    fit_seal = json.loads(fit_path.read_bytes())
    models = context_ids(experiment)
    require(fit_seal['design_sha256'] == digest(design_file(args.all_budgets)) and fit_seal['method'] == args.method
            and fit_seal['contexts'] == list(models) and all(digest(output / name) == expected
            for name, expected in fit_seal['model_files'].items()), 'Invalid or changed fitted-state receipt')
    path = output / f'{phase.upper()}_SCORES.npz'
    require(not path.exists(), f'Score packet already exists: {path}')
    if phase == 'test':
        previous = json.loads((output / 'CAL.json').read_bytes())
        require(previous['fit_sha256'] == digest(fit_path) and
                previous['score_sha256'] == digest(output / 'CAL_SCORES.npz'), 'Seal CAL before TEST')
    rows = experiment[phase]
    values, receipt = features(args, phase, rows)
    require(receipt['kind'] == fit_seal['features']['kind'], 'Fit and scoring must use the same descriptor source')
    method = args.method
    if method in (GLOBAL, CORE):
        with np.load(output / 'offset.npz', allow_pickle=False) as saved:
            values = global_frame(args, values, saved['offset'])
    if method in (FLID, RESNET):
        torch, device = torch_runtime(args.device, tf32=method == RESNET)
        model = resnet_model(torch).to(device) if method == RESNET else flid_factory(args.flid_source, torch)().to(device)
    scores = []
    for key in models:
        state_path = output / key
        if method == NRSW:
            vector = import_primary(state_path).score(values)
        elif method == RESNET:
            model.load_state_dict(torch.load(state_path.with_suffix('.pt'), weights_only=True, map_location='cpu'))
            vector = score_resnet(model, values, device)
        else:
            with np.load(state_path.with_suffix('.npz'), allow_pickle=False) as saved:
                if method in (GLOBAL, CORE):
                    head = AnalyticHead(saved['mean'], float(saved['rms']), saved['w'], float(saved['b']), {})
                    vector = head.score(values)
                elif method == DINO:
                    vector = score_dino(values, saved)
                else:
                    model.load_state_dict({name: torch.from_numpy(saved[name]) for name in saved.files})
                    vector = score_flid(model, values, device)
        scores.append(vector)
    scores = np.stack(scores)
    require(scores.shape == (len(models), len(rows)) and np.isfinite(scores).all(), 'Incomplete score matrix')
    save_arrays(path, context_ids=models, ids=[r['id'] for r in rows], scores=scores)
    record = {'fit_sha256': digest(fit_path), 'score_sha256': digest(path), 'features': receipt}
    if phase == 'cal':
        # CAL75 admits the one-sided 95% cuts only at 5% and 10%.
        record['cuts'] = np.sort(scores, axis=1)[:, [74, 72]].tolist()
    else:
        record['cal_sha256'] = digest(output / 'CAL.json')
    write_json(output / f'{phase.upper()}.json', record)
    print(f'{method} scored {len(rows)} {phase.upper()} rows', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('phase', choices=('fit', 'cal', 'test'))
    for name in ('output', 'features', 'images', 'weights', 'flid-source', 'historical-mean', 'shared-parent'):
        parser.add_argument('--' + name, type=Path, required=name == 'output')
    parser.add_argument('--images-layout', choices=('publisher', 'census'), default='publisher')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--all-budgets', action='store_true')
    args = parser.parse_args()
    required = ['images'] if args.method == RESNET else ['features']
    if args.method == RESNET and args.phase == 'fit':
        required.append('weights')
    if args.method == FLID:
        required.append('flid_source')
    if args.method in (NRSW, GLOBAL, CORE) and args.phase == 'fit':
        required.append('historical_mean')
    if args.method == CORE:
        required.append('shared_parent')
    for name in required:
        if getattr(args, name) is None:
            parser.error('missing --' + name.replace('_', '-'))
    output = args.output / args.method
    experiment = design(args.all_budgets)
    if args.phase == 'fit':
        output.mkdir(parents=True)
        fit(args, experiment, output)
    else:
        score(args, experiment, output, args.phase)


if __name__ == '__main__':
    main()
