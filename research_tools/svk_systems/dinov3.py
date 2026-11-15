"""Frozen DINOv3 pooled-region comparison for original and extended SVK."""

from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import math
import os

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from .cohort import METADATA_SHA, require, write
from .inputs import SIZES, load_roles, open_image
from .refit import RANKS, write_scores
from ..r62_svk import digest


MODEL = "vit_small_patch16_dinov3.lvd1689m"
WEIGHTS_SHA = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
FIELDS = ("birthday", "card_num", "country_code", "expire_date", "face", "gender",
          "given_name", "issue_authority", "issue_date", "personal_num", "surname")
PATCH, DIM, BATCH = 16, 384, 12
BLOCKS = len(FIELDS) + 1
WIDTH = BLOCKS * DIM
CORNER = np.array(((0, 0, 1), (224, 0, 1), (224, 224, 1), (0, 224, 1)), np.float64)
METHODS = ("DINOV3",)


@dataclass(frozen=True)
class Job:
    asset: dict
    maps: object

    @property
    def id(self):
        return self.asset["id"]

    @property
    def size(self):
        return self.asset["size"]

    @property
    def key(self):
        return self.asset["sha256"], self.size, (
            None if self.maps is None else sha256(np.asarray(self.maps, dtype=">f8").tobytes()).hexdigest()
        )


def layout(size):
    scale = 256 / min(size)
    resized = tuple(round(x * scale) for x in size)
    canvas = tuple(math.ceil(x / PATCH) * PATCH for x in resized)
    offset = tuple((p - x) // 2 for p, x in zip(canvas, resized, strict=True))
    return resized, canvas, offset


def area(polygon):
    if len(polygon) < 3:
        return 0.0
    x0, y0 = polygon[0]
    return abs(math.fsum((x - x0) * (v - y0) - (y - y0) * (u - x0)
                         for (x, y), (u, v) in zip(polygon[1:], polygon[2:]))) / 2


def clip(polygon, axis, boundary, direction):
    if not polygon:
        return []
    output = []
    a = polygon[-1]
    a_inside = direction * (a[axis] - boundary) >= 0.0
    for b in polygon:
        b_inside = direction * (b[axis] - boundary) >= 0.0
        if a_inside != b_inside:
            t = (boundary - a[axis]) / (b[axis] - a[axis])
            transverse = a[1 - axis] + t * (b[1 - axis] - a[1 - axis])
            output.append((boundary, transverse) if axis == 0 else (transverse, boundary))
        if b_inside:
            output.append(b)
        a, a_inside = b, b_inside
    return output


def normalized_patch_areas(polygon, canvas):
    """Exact geometric clipping in double precision, with no pixel rasterization."""
    width, height = canvas
    bounds = np.asarray(polygon, dtype=np.float64)
    require(bounds.shape == (4, 2) and np.isfinite(bounds).all(), "invalid finite quadrilateral")
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    require(np.all(lo >= 0) and np.all(hi <= canvas), "quadrilateral outside canvas")
    edges = np.roll(bounds, -1, axis=0) - bounds
    turn = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    require(np.all(turn > 0) or np.all(turn < 0), "registered polygon must be strictly convex")
    weights = np.zeros((height // PATCH, width // PATCH), dtype=np.float64)
    source = [tuple(point) for point in bounds.tolist()]
    expected = area(source)
    first_x, first_y = (math.floor(x / PATCH) for x in lo)
    stop_x, stop_y = (math.ceil(x / PATCH) for x in hi)
    if np.all((edges[:, 0] == 0) | (edges[:, 1] == 0)):
        x = PATCH * np.arange(first_x, stop_x)
        y = PATCH * np.arange(first_y, stop_y)
        weights[first_y:stop_y, first_x:stop_x] = np.outer(
            np.minimum(y + PATCH, hi[1]) - np.maximum(y, lo[1]),
            np.minimum(x + PATCH, hi[0]) - np.maximum(x, lo[0]))
    else:
        for row in range(first_y, stop_y):
            y0 = PATCH * row
            strip = clip(clip(source, 1, y0, 1), 1, y0 + PATCH, -1)
            for col in range(first_x, stop_x):
                x0 = PATCH * col
                weights[row, col] = area(clip(clip(strip, 0, x0, 1), 0, x0 + PATCH, -1))
    total = weights.sum()
    require(total > 0 and abs(total - expected) <= 1e-11 + 1e-10 * expected, "patch area conservation failed")
    return (weights / total).astype(np.float32).reshape(-1)


def rectangle(lo, hi):
    x0, y0 = lo
    x1, y1 = hi
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def pooling_weights(job):
    resized, canvas, offset = layout(job.size)
    end = tuple(x + p for x, p in zip(resized, offset, strict=True))
    whole = normalized_patch_areas(rectangle(offset, end), canvas)
    weights = np.zeros((BLOCKS, whole.size), dtype=np.float32)
    weights[0] = whole
    if job.maps is not None:
        maps = np.asarray(job.maps, dtype=np.float64)
        require(maps.shape == (len(FIELDS), 3, 3) and np.isfinite(maps).all(), f"invalid field maps {job.id}")
        homogeneous = CORNER @ maps.transpose(0, 2, 1)
        require(np.all(homogeneous[:, :, 2] > 0), f"invalid homogeneous denominator {job.id}")
        polygons = homogeneous[:, :, :2] / homogeneous[:, :, 2:]
        require(np.all(polygons >= 0) and np.all(polygons <= job.size), f"field polygon outside source {job.id}")
        mapped = polygons * (np.asarray(resized, dtype=np.float64) / job.size) + offset
        for index, polygon in enumerate(mapped, 1):
            weights[index] = normalized_patch_areas(polygon, canvas)
    return weights


def transform_image(image, resized, canvas, offset):
    require(image.mode == "RGB", "RGB input required")
    padded = Image.new("RGB", canvas)
    padded.paste(image.resize(resized, Image.Resampling.BICUBIC), offset)
    return np.asarray(padded, dtype=np.uint8).transpose(2, 0, 1)


class Encoder:
    def __init__(self, path):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import timm
        import torch
        from safetensors.torch import load_file

        require(torch.cuda.is_available() and timm.__version__ == "1.0.31", "Expected archived timm 1.0.31 and CUDA")
        require(digest(path) == WEIGHTS_SHA, "Expected pinned DINOv3 ViT-S/16 weights")
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch, self.timm, self.device = torch, timm, torch.device("cuda:0")
        self.state = load_file(path, device="cpu")
        self.models = {}
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def pool(self, canvas, pixels, weights, rows):
        torch = self.torch
        if canvas not in self.models:
            model = self.timm.create_model(MODEL, pretrained=False, num_classes=0, img_size=canvas[::-1])
            model.load_state_dict(self.state, strict=True)
            require(model.num_prefix_tokens == 5 and model.num_features == DIM, "Unexpected DINO patch contract")
            self.models[canvas] = model.requires_grad_(False).eval().to(self.device, dtype=torch.float32)
        model = self.models[canvas]
        pixels += [pixels[-1]] * (BATCH - len(pixels))
        x = torch.from_numpy(np.stack(pixels)).to(self.device, dtype=torch.float32) / 255
        regions = torch.from_numpy(np.stack(weights)).to(self.device, dtype=torch.float32)
        index = torch.tensor(rows, device=self.device, dtype=torch.long)
        with torch.inference_mode():
            tokens = model.forward_features((x - self.mean) / self.std)[:, 5:]
            return torch.bmm(regions, tokens.index_select(0, index)).flatten(1).cpu().numpy()


def measure(jobs, encoder, cache):
    result, pending = {}, defaultdict(dict)
    validated = set()
    representatives = {}
    for job in jobs:
        key = job.key
        locator = job.asset["path"]
        if key in cache:
            if locator not in validated:
                with open_image(job.asset):
                    pass
                validated.add(locator)
            result[job.id] = cache[key]
        else:
            payload = key[:2]
            pending[job.size].setdefault(payload, {})
            pending[job.size][payload][key] = job
            representatives.setdefault(payload, {})[locator] = job

    def prepare(entry):
        payload, unique = entry
        row = next(iter(unique.values()))
        with open_image(row.asset) as image:
            pixels = transform_image(image, *layout(row.size))
        for locator, linked in representatives[payload].items():
            if locator != row.asset["path"]:
                with open_image(linked.asset):
                    pass
        return pixels, [pooling_weights(job) for job in unique.values()]

    with ThreadPoolExecutor(max_workers=6) as workers:
        for size in SIZES.values():
            selected = list(pending[size].items())
            for start in range(0, len(selected), BATCH):
                block = selected[start:start + BATCH]
                pixels, weights, lanes, keys = [], [], [], []
                for lane, ((_, unique), (image, areas)) in enumerate(zip(block, workers.map(prepare, block), strict=True)):
                    pixels.append(image)
                    weights.extend(areas)
                    keys.extend(unique)
                    lanes.extend([lane] * len(areas))
                vectors = encoder.pool(layout(size)[1], pixels, weights, lanes)
                require(vectors.dtype == np.float32 and vectors.shape == (len(keys), WIDTH)
                        and np.isfinite(vectors).all(), "Expected finite DINOv3 feature rows")
                cache.update(zip(keys, vectors, strict=True))
    for job in jobs:
        result[job.id] = cache[job.key]
    return result


def select(assets, owners):
    selected = {}
    for ids, recovered, residual in owners:
        for identity in ids:
            asset = assets[identity]
            maps = asset["maps"]
            if identity in recovered:
                region = maps["recovered"]
            elif identity in residual:
                region = None
            else:
                region = maps["original"]
            job = Job(asset, region)
            require((job.maps is not None) == (identity not in residual), "Exact owning-role DINO geometry")
            require(selected.setdefault(identity, job).key == job.key, "Conflicting geometry in this evaluation phase")
    return [selected[i] for i in sorted(selected, key=int)]


def matrix(table, ids):
    return np.stack([table[key] for key in ids])


def score(features, state):
    output = np.empty(len(features), np.float64)
    for first in range(0, len(features), 512):
        values = features[first:first + 512]
        x = (values.astype(np.float64) - state["mean"]) / state["scale"]
        missing = np.all(values[:, DIM:] == 0, axis=1)
        x[missing, DIM:] = 0
        y = x @ state["coef"] + state["intercept"]
        y[missing] = x[missing, :DIM] @ state["coef"][:DIM] + state["intercept"]
        output[first:first + len(values)] = y
    return output


def run(runtime, meta, original_images, added_images, weights, output, only_context=None):
    import sklearn
    from ..crossverify.neural import fit_dino

    require(sklearn.__version__ == "1.9.0", "Expected archived scikit-learn 1.9.0")
    registry, contexts, assets, added = load_roles(runtime, meta, original_images, added_images)
    require(tuple(registry["field_order"]) == FIELDS, "Unexpected SVK DINO field order")
    if only_context:
        contexts = {f: [r for r in c if r["id"] == only_context] for f, c in contexts.items()}
        contexts = {f: c for f, c in contexts.items() if c}
        require(len(contexts) == 1, "Unknown SVK context")
    outer = {f: registry["outer_folds"][str(f)] for f in contexts}
    output.mkdir(parents=True, exist_ok=False)
    encoder = Encoder(weights)
    cache = {}
    train_jobs = select(assets, [
        ([row["id"] for row in ctx["train"]], ctx["recovered_train_ids"], [])
        for roster in contexts.values() for ctx in roster
    ])
    train = measure(train_jobs, encoder, cache)
    heads = {}
    with threadpool_limits(limits=1):
        for fold, roster in contexts.items():
            for ctx in roster:
                rows = ctx["train"]
                heads[ctx["id"]] = fit_dino(matrix(train, [row["id"] for row in rows]),
                                           np.array([row["role"] == "F" for row in rows], np.int64))
            print(f"FIT DINOv3 fold {fold}: {len(roster)} models", flush=True)
        model_path = output / "heads.npz"
        with model_path.open("xb") as stream:
            np.savez_compressed(stream, contexts=list(heads),
                                **{key: np.stack([head[key] for head in heads.values()])
                                   for key in ("mean", "scale", "coef", "intercept")})
        fit = output / "FIT_COMPLETE.json"
        write(fit, {"methods": METHODS, "weights_sha256": WEIGHTS_SHA, "heads_sha256": digest(model_path)})
        calibration = measure(select(assets, [(row["calibration_ids"], row["calibration_recovered_ids"], [])
                                               for row in outer.values()]), encoder, cache)
        packets = {}
        for fold, roster in contexts.items():
            keys = [row["id"] for row in roster]
            ids = outer[fold]["calibration_ids"]
            values = matrix(calibration, ids)
            scores = np.stack([score(values, heads[key]) for key in keys])[:, None, :]
            cuts = np.sort(scores, axis=-1)[..., np.asarray(RANKS) - 1]
            packets[fold] = write_scores(output, fold, "cal", ids, keys, METHODS, scores, cuts=cuts)
        cal = output / "CALIBRATION_COMPLETE.json"
        write(cal, {"fit_sha256": digest(fit), "packets": packets, "ascending_ranks": RANKS, "decision": "score > cut"})
        jobs = select(assets, [(row["test_ids"], row["test_recovered_ids"], row["test_residual_ids"])
                               for row in outer.values()])
        extras = sorted([row for rows in added.values() for row in rows], key=lambda row: int(row["id"].split(":")[1]))
        jobs.extend(Job(row, row["maps"]["original"]) for row in extras)
        test = measure(jobs, encoder, cache)
        packets = {}
        for fold, roster in contexts.items():
            keys = [row["id"] for row in roster]
            old = outer[fold]["test_ids"]
            extra = [row["id"] for row in added[fold]]
            values = matrix(test, old + extra)
            scores = np.stack([score(values, heads[key]) for key in keys])[:, None, :]
            ids = [f"original:{i}" for i in old] + extra
            packets[fold] = write_scores(output, fold, "test", ids, keys, METHODS, scores)
    write(output / "COMPLETE.json", {"schema": "R62_SVK_MATCHED_SYSTEM_SCORES_V1", "methods": METHODS,
          "cal_seal_sha256": digest(cal), "test_packets": packets, "added_metadata_sha256": METADATA_SHA})
