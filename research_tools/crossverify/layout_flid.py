"""FLiD face and text refits for full CrossVerify layouts."""

import ast
import math

import numpy as np

from inputs import digest, require
from neural import FLID_SHA, flid_train, score_flid, torch_runtime


FIELDS = {'face': (0,), 'text': (1, 2)}


def factories(path, torch, family):
    require(digest(path) == FLID_SHA, 'FLiD upstream source differs')
    names = {'face': 'FaceClassifier', 'text': 'TextClassifier'} if family == 'family_a' else {'text': 'TextClassifier'}
    parsed = ast.parse(path.read_bytes(), filename=str(path))
    definitions = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names.values()]
    require({node.name for node in definitions} == set(names.values()), 'FLiD upstream classes differ')
    scope = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    return {branch: scope[name] for branch, name in names.items()}


def positive(row, branch):
    return row['label'] == 1 and row['candidates'][0] in FIELDS[branch]


def selector_folds(rows, family):
    folds = [row['fold'] for row in rows]
    swaps = []
    if family == 'family_a':
        for branch in FIELDS:
            candidates = [i for i, row in enumerate(rows) if positive(row, branch)]
            assigned = {folds[i] for i in candidates}
            require(len(candidates) >= 2, 'Fewer than two modality positives')
            if len(assigned) == 1:
                donor = folds[candidates[0]]
                incoming = min(candidates, key=lambda i: rows[i]['id'])
                outgoing = min((i for i, row in enumerate(rows) if row['label'] and folds[i] == 1-donor
                                and not positive(row, branch)), key=lambda i: rows[i]['id'])
                folds[incoming], folds[outgoing] = 1-donor, donor
                swaps.append((rows[incoming]['id'], rows[outgoing]['id']))
    bank = sorted((i for i, row in enumerate(rows) if row['role'] == 'BANK'), key=lambda i: rows[i]['component'])
    for position, index in enumerate(bank):
        folds[index] = position % 2
    for i, row in enumerate(rows):
        if row['role'] == 'N':
            folds[i] = -1
    return [([i for i, f in enumerate(folds) if f != held], [i for i, f in enumerate(folds) if f == held])
            for held in (0, 1)], swaps


def patches_for(rows, patches, positions, branch):
    admitted = [i for i in positions if not rows[i]['label'] or positive(rows[i], branch)]
    fields = FIELDS[branch]
    values = [patches[i, rows[i]['candidates'] if rows[i]['label'] else list(fields)] for i in admitted]
    targets = np.repeat([rows[i]['label'] for i in admitted], [len(v) for v in values]).astype(np.float32)
    require(np.unique(targets).tolist() == [0, 1], 'Both FLiD patch classes are required')
    return np.concatenate(values), targets, admitted


def branch_scores(torch, model, batch, device, patch_count, *, batch_size=2048):
    x = torch.from_numpy(np.ascontiguousarray(batch.reshape(-1, 576))).to(device)
    model.eval()
    with torch.inference_mode():
        pieces = (x,) if batch_size is None else x.split(batch_size)
        scores = torch.cat([model(chunk).squeeze(-1) for chunk in pieces]).cpu().numpy()
    return scores.reshape(-1, patch_count).max(axis=1).astype(np.float64)


def fit_branch(torch, factory, inputs, targets, seed, device, *, validation=None, rates=None):
    """Same patch BCE and held TRAIN-only schedule used by the full-layout face/text study."""
    torch.manual_seed(seed)
    model = factory().to(device, dtype=torch.float32)
    x, y = torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
    p = int((targets == 1).sum())
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(targets)-p)/p], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5) if validation else None
    loader = torch.utils.data.DataLoader(range(len(y)), batch_size=32, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))
    best, best_epoch, stale = math.inf, None, 0
    trace = []
    for epoch in range(1, len(rates)+1 if rates is not None else 101):
        if rates is not None:
            optimizer.param_groups[0]['lr'] = rates[epoch-1]
        trace.append(optimizer.param_groups[0]['lr'])
        model.train()
        total = torch.zeros((), device=device)
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            value = loss(model(x[batch]).squeeze(-1), y[batch])
            value.backward()
            optimizer.step()
            total += value.detach() * len(batch)
        require(torch.isfinite(total).item(), 'Nonfinite FLiD training loss')
        if validation:
            patches, labels, patch_count = validation
            pred = branch_scores(torch, model, patches, device, patch_count)
            value = float(.5*np.logaddexp(0, pred[labels == 0]).mean() + .5*np.logaddexp(0, -pred[labels == 1]).mean())
            scheduler.step(value)
            if value < best:
                best, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= 15:
                break
    return model, best_epoch, trace


def rates(selectors):
    count = math.ceil(sum(epoch for epoch, _ in selectors)/2)
    a, b = [trace for _, trace in selectors]
    return [math.sqrt(a[min(i,len(a)-1)] * b[min(i,len(b)-1)]) for i in range(count)]


def fit_flid_layout(patches, context, upstream, device_name):
    torch, device = torch_runtime(device_name)
    family, rows = context['family'], context['train']
    seed = 20261111 + 10000*context['fold'] + 1000*context['draw']
    folds, swaps = selector_folds(rows, family)
    require(bool(swaps) == (context['id'] in ('family_a_f4_d1', 'family_a_f4_d3')), 'FLiD selector swaps differ')
    fitted = {}
    for branch, factory in factories(upstream, torch, family).items():
        selectors = []
        branch_seed = seed + (10 if family == 'family_a' and branch == 'text' else 0)
        for stage, (fit, held) in enumerate(folds):
            if family == 'family_b':
                labels = np.array([rows[i]['label'] for i in held])
                _, epoch, trace = flid_train(torch, device, factory, patches, rows, fit, branch_seed+stage,
                                             validation=(held, labels))
            else:
                x, y, _ = patches_for(rows, patches, fit, branch)
                _, _, indices = patches_for(rows, patches, held, branch)
                labels = np.array([rows[i]['label'] for i in indices])
                fields = FIELDS[branch]
                validation = (patches[indices][:, fields], labels, len(fields))
                _, epoch, trace = fit_branch(torch, factory, x, y, branch_seed+stage, device, validation=validation)
            require(epoch is not None, 'No finite FLiD selector objective')
            selectors.append((epoch, trace))
        if family == 'family_b':
            model, _, _ = flid_train(torch, device, factory, patches, rows, range(len(rows)), branch_seed+2,
                                     rates=rates(selectors))
        else:
            x, y, _ = patches_for(rows, patches, range(len(rows)), branch)
            model, _, _ = fit_branch(torch, factory, x, y, branch_seed+2, device, rates=rates(selectors))
        fitted.update({f'{branch}/{name}': p.detach().cpu().numpy().copy() for name,p in model.state_dict().items()})
    return fitted


def score_flid_layout(patches, family, upstream, saved, device_name):
    torch, device = torch_runtime(device_name)
    scores = []
    for branch, factory in factories(upstream, torch, family).items():
        model = factory().to(device)
        model.load_state_dict({name: torch.from_numpy(saved[f'{branch}/{name}']) for name in model.state_dict()})
        if family == 'family_b':
            scores.append(score_flid(model, patches, device))
        else:
            fields = FIELDS[branch]
            scores.append(branch_scores(torch, model, patches[:, fields], device, len(fields), batch_size=None))
    return np.maximum.reduce(scores)
