"""CrossVerify ResNet-18, DINOv3, and FLiD receiving classifiers."""

import ast
import math
import random
import warnings

import numpy as np

if __package__:
    from .inputs import digest, require
else:
    from inputs import digest, require


RESNET_SHA = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
FLID_SHA = 'bc68cfc4c30badd4e3edd25d0b75975dc876d00098a5587259f89eead590f895'


def torch_runtime(device, *, tf32=False):
    import torch

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch, torch.device(device)


def resnet_pixels(rows, source):
    from PIL import Image
    import torch

    data = []
    for row in rows:
        with source.image(row) as image, Image.new('RGB', (224, 224), color=(0, 0, 0)) as canvas:
            w, h = image.size
            ratio = min(224 / w, 224 / h)
            size = (round(w * ratio), round(h * ratio))
            canvas.paste(image.resize(size, Image.Resampling.BILINEAR), ((224-size[0])//2, (224-size[1])//2))
            data.append(np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1))
    x = torch.from_numpy(np.ascontiguousarray(np.stack(data))).to(torch.float32).div_(255)
    mean = torch.tensor((.485, .456, .406), dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor((.229, .224, .225), dtype=x.dtype).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


def resnet_model(torch, weights=None):
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    if weights is not None:
        model.load_state_dict(weights, strict=True)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(512, 1))
    return model


def fit_resnet(args, x, labels):
    torch, device = torch_runtime(args.device, tf32=True)
    require(digest(args.weights) == RESNET_SHA, 'ResNet checkpoint differs')
    weights = torch.load(args.weights, map_location='cpu', weights_only=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    y = torch.tensor(labels, dtype=torch.long)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True)
    model = resnet_model(torch, weights).to(device)
    positive_weight = (y == 0).sum().item() / (y == 1).sum().item()
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    for _ in range(15):
        model.train()
        for images, targets in loader:
            optimizer.zero_grad()
            value = loss(model(images.to(device)), targets.to(device).float().unsqueeze(1))
            require(torch.isfinite(value).item(), 'Nonfinite ResNet training loss')
            value.backward()
            optimizer.step()
        schedule.step()
    return model


def score_resnet(model, x, device):
    import torch

    model.eval()
    with torch.no_grad():
        outputs = [torch.sigmoid(model(batch.to(device))).cpu().numpy().reshape(-1) for batch in x.split(32)]
    return np.concatenate(outputs).astype(np.float64)


def fit_dino(features, labels):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(C=1., class_weight='balanced', fit_intercept=True, max_iter=1000,
                                   tol=1e-8, solver='lbfgs', random_state=42).fit(scaler.transform(features.astype(np.float64)), labels)
    return dict(mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_[0], intercept=model.intercept_[0])


def score_dino(features, state):
    return ((features.astype(np.float64) - state['mean']) / state['scale']) @ state['coef'] + state['intercept']


def flid_factory(source, torch):
    require(digest(source) == FLID_SHA, 'FLiD upstream source differs')
    tree = ast.parse(source.read_bytes(), filename=str(source))
    [classifier] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TextClassifier']
    env = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[classifier], type_ignores=[]), str(source), 'exec'), env)
    return env['TextClassifier']


def score_flid(model, patches, device):
    import torch

    model.eval()
    values = torch.from_numpy(patches.reshape(-1, 576)).to(device)
    with torch.inference_mode():
        scores = torch.cat([model(batch).squeeze(-1) for batch in values.split(2048)]).cpu().numpy()
    return np.maximum.reduceat(scores, np.arange(len(patches), dtype=np.int64) * 3).astype(np.float64)


def flid_train(torch, device, factory, patches, rows, indices, seed, validation=None, rates=None):
    torch.manual_seed(seed)
    model = factory().to(device, dtype=torch.float32)
    features = [patches[i, rows[i]['candidates'] if rows[i]['role'] == 'F' else [0, 1, 2]] for i in indices]
    labels = np.repeat([int(rows[i]['role'] == 'F') for i in indices], [len(p) for p in features]).astype(np.float32)
    x = torch.from_numpy(np.concatenate(features)).to(device)
    y = torch.from_numpy(labels).to(device)
    pos = int((labels == 1).sum())
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(labels)-pos)/pos], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    loader = torch.utils.data.DataLoader(range(len(y)), batch_size=32, shuffle=True,
                                        generator=torch.Generator().manual_seed(seed))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5) if validation is not None else None
    best, best_epoch, stale = math.inf, None, 0
    trace = []
    for epoch in range(1, len(rates) + 1 if rates is not None else 101):
        if rates is not None:
            optimizer.param_groups[0]['lr'] = rates[epoch-1]
        trace.append(optimizer.param_groups[0]['lr'])
        model.train()
        total = torch.zeros((), device=device)
        for subset in loader:
            subset = subset.to(device)
            optimizer.zero_grad()
            step_loss = loss(model(x[subset]).squeeze(-1), y[subset])
            step_loss.backward()
            optimizer.step()
            total += step_loss.detach() * len(subset)
        require(torch.isfinite(total).item(), 'Nonfinite FLiD training loss')
        if validation is not None:
            test, labels = validation
            pred = score_flid(model, patches[test], device)
            value = float(.5 * np.logaddexp(0, pred[labels == 0]).mean() + .5 * np.logaddexp(0, -pred[labels == 1]).mean())
            scheduler.step(value)
            if value < best:
                best, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if stale >= 15:
                break
    return model, best_epoch, trace


def fit_flid(args, patches, context):
    torch, device = torch_runtime(args.device)
    factory = flid_factory(args.flid_source, torch)
    rows = context['train']
    bank = sorted((r for r in rows if r['role'] == 'BANK'), key=lambda r: r['component'])
    bank_folds = {r['id']: i % 2 for i, r in enumerate(bank)}
    folds = [-1 if r['role'] == 'N' else bank_folds.get(r['id'], r['fold']) for r in rows]
    seed = 20261111 + 1000 * context['draw']
    selectors = []
    for fold in range(2):
        fit = [i for i, assigned in enumerate(folds) if assigned != fold]
        held = [i for i, assigned in enumerate(folds) if assigned == fold]
        labels = np.array([int(rows[i]['role'] == 'F') for i in held])
        _, epoch, trace = flid_train(torch, device, factory, patches, rows, fit, seed+fold, validation=(held, labels))
        require(epoch is not None, 'FLiD selector produced no finite validation loss')
        selectors.append((epoch, trace))
    count = math.ceil(sum(epoch for epoch, _ in selectors)/2)
    rates = [math.sqrt(selectors[0][1][min(i, len(selectors[0][1])-1)]
                       * selectors[1][1][min(i, len(selectors[1][1])-1)]) for i in range(count)]
    model, _, _ = flid_train(torch, device, factory, patches, rows, range(len(rows)), seed+2, rates=rates)
    return model
