"""Refit the matched full-image ResNet-18 baseline on the frozen SVK roles."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .cohort import METADATA_SHA, require, write
from .inputs import load_roles, open_image
from .refit import RANKS, write_scores
from ..r62_svk import digest


RESNET_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
METHODS = ("RESNET18",)


def canvas(asset):
    with open_image(asset) as image, Image.new("RGB", (224, 224)) as padded:
        w, h = image.size
        scale = min(224 / w, 224 / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        with image.resize((rw, rh), Image.Resampling.BILINEAR) as resized:
            padded.paste(resized, ((224 - rw) // 2, (224 - rh) // 2))
        return np.ascontiguousarray(np.asarray(padded).transpose(2, 0, 1))


def materialize(rows, destination):
    out = np.lib.format.open_memmap(destination, mode="w+", dtype=np.uint8,
                                    shape=(len(rows), 3, 224, 224))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for first in range(0, len(rows), 24):
            block = rows[first:first + 24]
            for j, values in enumerate(pool.map(canvas, block), start=first):
                out[j] = values
    out.flush()
    return out, {row["id"]: i for i, row in enumerate(rows)}


def normalized(torch, pixels, positions, ids):
    x = torch.from_numpy(pixels[[positions[i] for i in ids]]).to(torch.float32).div_(255)
    mean = torch.tensor((.485, .456, .406), dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor((.229, .224, .225), dtype=x.dtype).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


def run(runtime, meta, original_images, added_images, weights, output, only_context=None):
    from ..crossverify import neural
    import torch

    require(torch.cuda.is_available(), "The archived SVK ResNet recipe requires CUDA")
    require(str(torch.__version__) == "2.13.0+cu130", "Expected archived Torch 2.13.0+cu130")
    require(digest(weights) == RESNET_SHA, "Expected archived ResNet-18 ImageNet initialization")
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    registry, contexts, assets, added = load_roles(runtime, meta, original_images, added_images)
    if only_context:
        contexts = {f: [r for r in c if r["id"] == only_context] for f, c in contexts.items()}
        contexts = {f: c for f, c in contexts.items() if c}
        require(len(contexts) == 1, "Unknown SVK context")
    output.mkdir(parents=True, exist_ok=False)
    states = output / "models"
    states.mkdir()
    train_rows = list(dict.fromkeys(row["id"] for c in contexts.values() for ctx in c for row in ctx["train"]))
    data, positions = materialize([assets[i] for i in train_rows], output / "train_pixels.npy")
    model_args = SimpleNamespace(device=str(device), weights=weights)
    for fold, roster in contexts.items():
        for ctx in roster:
            ids = [row["id"] for row in ctx["train"]]
            labels = np.array([row["role"] == "F" for row in ctx["train"]], np.int64)
            x = normalized(torch, data, positions, ids)
            model = neural.fit_resnet(model_args, x, labels)
            with (states / f'{ctx["id"]}.pt').open("xb") as stream:
                torch.save({key: v.detach().cpu() for key, v in model.state_dict().items()}, stream)
            del model, x
        print(f"FIT ResNet fold {fold}: {len(roster)} models", flush=True)
    fit = output / "FIT_COMPLETE.json"
    fitted = {p.name: digest(p) for p in states.glob("*.pt")}
    write(fit, {"methods": METHODS, "weights_sha256": RESNET_SHA, "epochs": 15, "seed": 42,
                "states": fitted})
    del data

    def infer(keys, x):
        values = []
        for key in keys:
            model = neural.resnet_model(torch)
            path = states / f"{key}.pt"
            require(digest(path) == fitted[path.name], "Fitted ResNet state changed")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.to(device)
            model.eval()
            with torch.inference_mode():
                scores = torch.cat([torch.sigmoid(model(block)).flatten() for block in x.split(32)])
            values.append(scores.cpu().numpy().astype(np.float64))
            del model
        return np.stack(values)[:, None, :]

    packets, limits = {}, {}
    for fold, roster in contexts.items():
        selected = registry["outer_folds"][str(fold)]["calibration_ids"]
        data, positions = materialize([assets[i] for i in selected], output / f"cal{fold}_pixels.npy")
        x = normalized(torch, data, positions, selected).to(device)
        keys = [ctx["id"] for ctx in roster]
        scores = infer(keys, x)
        limits[fold] = np.sort(scores, axis=-1)[..., np.asarray(RANKS) - 1]
        packets[fold] = write_scores(output, fold, "cal", selected, keys, METHODS, scores, cuts=limits[fold])
        del x, data
    cal = output / "CALIBRATION_COMPLETE.json"
    write(cal, {"fit_sha256": digest(fit), "packets": packets, "ascending_ranks": RANKS, "decision": "score > cut"})
    packets = {}
    for fold, roster in contexts.items():
        original_ids = registry["outer_folds"][str(fold)]["test_ids"]
        test_rows = [assets[i] for i in original_ids] + added[fold]
        selected = [row["id"] for row in test_rows]
        data, positions = materialize(test_rows, output / f"test{fold}_pixels.npy")
        x = normalized(torch, data, positions, selected).to(device)
        keys = [ctx["id"] for ctx in roster]
        scores = infer(keys, x)
        ids = [f"original:{i}" for i in original_ids] + [row["id"] for row in added[fold]]
        packets[fold] = write_scores(output, fold, "test", ids, keys, METHODS, scores)
        del x, data
    write(output / "COMPLETE.json", {"schema": "R62_SVK_MATCHED_SYSTEM_SCORES_V1", "methods": METHODS,
          "cal_seal_sha256": digest(cal), "test_packets": packets, "added_metadata_sha256": METADATA_SHA})
