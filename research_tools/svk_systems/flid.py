"""Reproduce the matched SVK FLiD text arm from the released images."""
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cohort import METADATA_SHA, require as need, write
from .inputs import load_roles, open_image
from .refit import RANKS, write_scores
from ..r62_svk import digest


FLID_SOURCE_SHA256 = "bc68cfc4c30badd4e3edd25d0b75975dc876d00098a5587259f89eead590f895"
METHODS = ("FLID",)
TEXT = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10)


@dataclass(frozen=True)
class TrainingRow:
    id: str
    group: str
    role: str
    fold: int
    candidates: tuple

    @property
    def label(self):
        return int(self.role == "F")


@dataclass(frozen=True)
class TrainingContext:
    key: tuple
    train: tuple


def inner_roles(rows):
    """Retain anchors in both selector fits and split BANK by sorted component."""
    banks = sorted((row for row in rows if row.role == "BANK"), key=lambda row: row.group)
    bank_folds = {row.id: i % 2 for i, row in enumerate(banks)}
    assignments = {row.id: bank_folds[row.id] if row.role == "BANK" else row.fold
                   for row in rows if row.role != "N"}
    return tuple((tuple(i for i, row in enumerate(rows) if assignments.get(row.id) != held),
                  tuple(i for i, row in enumerate(rows) if assignments.get(row.id) == held))
                 for held in (0, 1))


BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCORE_BATCH = 2048


def author_components(source):
    from torchvision import transforms

    code = source.read_bytes()
    need(hashlib.sha256(code).hexdigest() == FLID_SOURCE_SHA256, "Expected pinned upstream FLiD source")
    tree = ast.parse(code, filename=str(source))
    text = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TextClassifier"]
    transform = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "TRANSFORM" for t in node.targets)]
    need(len(text) == len(transform) == 1, "Expected upstream FLiD classifier and image transform")
    scope = {"torch": torch, "nn": nn, "transforms": transforms}
    exec(compile(ast.Module(body=text + transform, type_ignores=[]), str(source), "exec"), scope)
    return scope["TextClassifier"], scope["TRANSFORM"]

def validate_patches(patches, counts):
    need(patches.dtype == np.float32 and patches.shape == (len(counts), 10, 576)
         and counts.dtype == np.int64 and counts.shape == (len(patches),)
         and np.isin(counts, (1, 10)).all(), "Invalid raw576 patch packet")


@dataclass(frozen=True)
class ScorePatches:
    patches: torch.Tensor
    starts: np.ndarray

    @classmethod
    def build(cls, patches, counts, device):
        validate_patches(patches, counts)
        need(len(counts) > 0, "Cannot score an empty document set")
        valid = np.arange(10)[None, :] < counts[:, None]
        compact = patches[valid]
        need(np.isfinite(compact).all(), "Nonfinite active patch descriptor")
        return cls(torch.from_numpy(compact).to(device), np.r_[0, np.cumsum(counts[:-1])])


def score_documents(model, batch):
    """Return the maximum forged logit over each document's available patches."""
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(batch.patches), SCORE_BATCH):
            patch_batch = batch.patches[start:start + SCORE_BATCH]
            values.append(model(patch_batch).squeeze(-1))
        raw = torch.cat(values).cpu().numpy()
    scores = np.maximum.reduceat(raw, batch.starts).astype(np.float64)
    need(np.isfinite(scores).all(), "Nonfinite document scores")
    return scores


def fitting_patches(patches, counts, rows, indices, text_indices):
    """Use all genuine text patches and only tagged support text patches."""
    field_slots = {index: slot for slot, index in enumerate(text_indices)}
    chunks, labels = [], []
    for i in indices:
        row = rows[i]
        if row.label:
            need(counts[i] == 10, "Support forgery is missing a tagged text patch")
            subset = patches[i, [field_slots[index] for index in row.candidates]]
        else:
            need(counts[i] == 10, "Training genuine is missing text patches")
            subset = patches[i]
        chunks.append(subset)
        labels.extend([row.label] * len(subset))
    values = np.concatenate(chunks)
    targets = np.asarray(labels, dtype=np.float32)
    need(np.isfinite(values).all() and np.unique(targets).tolist() == [0.0, 1.0],
         "Expected finite fitting patches from both classes")
    return values, targets


def balanced_validation_loss(scores, labels):
    return float(0.5 * np.logaddexp(0, scores[labels == 0]).mean()
                 + 0.5 * np.logaddexp(0, -scores[labels == 1]).mean())


def _train(factory, values, targets, seed, device, *, validation=None, learning_rates=None, max_epochs=MAX_EPOCHS):
    torch.manual_seed(seed)
    model = factory().to(device, dtype=torch.float32)
    features = torch.from_numpy(values).to(device)
    labels = torch.from_numpy(targets).to(device)
    positive = int((targets == 1).sum())
    negative = len(targets) - positive
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
                 if validation is not None else None)
    loader = DataLoader(range(len(targets)), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)

    best_loss, best_epoch, stale = math.inf, None, 0
    trace, updates = [], 0
    epochs = len(learning_rates) if learning_rates is not None else max_epochs
    for epoch in range(1, epochs + 1):
        if learning_rates is not None:
            optimizer.param_groups[0]["lr"] = learning_rates[epoch - 1]
        rate = optimizer.param_groups[0]["lr"]
        model.train()
        total = torch.zeros((), device=device)
        for indices in loader:
            indices = indices.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(features[indices]).squeeze(-1), labels[indices])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(indices)
            updates += 1
        training_loss = float(total.item() / len(labels))
        need(math.isfinite(training_loss), "Nonfinite training loss")
        row = {"epoch": epoch, "lr": rate, "training_bce": training_loss}
        if validation is not None:
            held_patches, held_labels = validation
            validation_loss = balanced_validation_loss(score_documents(model, held_patches), held_labels)
            need(math.isfinite(validation_loss), "Nonfinite held validation loss")
            scheduler.step(validation_loss)
            row["balanced_validation_bce"] = validation_loss
            if validation_loss < best_loss:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
            else:
                stale += 1
        trace.append(row)
        if stale >= PATIENCE:
            break
    need(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
         "Nonfinite fitted FLiD parameters")
    return model, {"seed": seed, "positive_patches": positive, "negative_patches": negative,
                   "positive_weight": negative / positive, "best_epoch": best_epoch,
                   "optimizer_updates": updates, "epochs": trace}


def final_rates(selectors):
    epochs = math.ceil(sum(selector["best_epoch"] for selector in selectors) / len(selectors))
    traces = [selector["epochs"] for selector in selectors]
    return [math.sqrt(traces[0][min(i, len(traces[0]) - 1)]["lr"]
                      * traces[1][min(i, len(traces[1]) - 1)]["lr"]) for i in range(epochs)]


def fit_context(factory, context, patches, counts, text_indices, device, *, max_epochs=MAX_EPOCHS):
    validate_patches(patches, counts)
    rows = context.train
    need(len(patches) == len(rows) and (counts == 10).all(), "Every training row needs its complete frozen patch bag")
    fold, draw, omit = context.key
    seed_base = 20261111 + 10000 * fold + 1000 * draw + 10 * omit
    selectors = []
    for stage, (fitting, held) in enumerate(inner_roles(rows)):
        values, targets = fitting_patches(patches, counts, rows, fitting, text_indices)
        held_labels = np.asarray([rows[i].label for i in held], dtype=np.int64)
        held_indices = list(held)
        validation = (ScorePatches.build(patches[held_indices], counts[held_indices], device), held_labels)
        model, stats = _train(factory, values, targets, seed_base + stage, device,
                              validation=validation, max_epochs=max_epochs)
        del model
        stats.update(fit_ids=[rows[i].id for i in fitting], held_ids=[rows[i].id for i in held])
        selectors.append(stats)
    values, targets = fitting_patches(patches, counts, rows, range(len(rows)), text_indices)
    model, final = _train(factory, values, targets, seed_base + 2, device, learning_rates=final_rates(selectors))
    return model, {"selectors": selectors, "final": final}



def local_packs(rows, maps, columns, documents_per_pack, encode, keep=None):
    """Crop in archived order and flush the physical 224-view queue inside each pack."""
    from nrsw.registration import rectify_map

    keep = tuple(range(len(columns))) if keep is None else keep
    output = np.empty((len(rows), len(keep), 576), dtype=np.float32)
    for first in range(0, len(rows), documents_per_pack):
        block = rows[first:first + documents_per_pack]
        packed = np.empty((len(block), len(columns), 576), dtype=np.float32)
        flat = packed.reshape(-1, 576)
        crops = []
        position = 0

        def flush():
            nonlocal position
            descriptors = encode(crops)
            need(descriptors.shape == (len(crops), 576) and descriptors.dtype == np.float32
                 and np.isfinite(descriptors).all(), "Invalid MobileNet raw576 field descriptors")
            flat[position:position + len(crops)] = descriptors
            position += len(crops)
            crops.clear()

        for asset in block:
            field_maps = asset["maps"][maps]
            need(len(field_maps) == 11, "Expected 11 registered SVK maps")
            with open_image(asset) as image:
                for col in columns:
                    crops.append(rectify_map(image, field_maps[col]))
                    if len(crops) == 224:
                        flush()
        if crops:
            flush()
        need(position == len(block) * len(columns), "Every archived crop measured once")
        output[first:first + len(block)] = packed[:, keep, :]
    return output


def whole_packs(rows, transform, mobile, guard, *, batch_size):
    output = np.empty((len(rows), 576), dtype=np.float32)
    for first in range(0, len(rows), batch_size):
        images = []
        for asset in rows[first:first + batch_size]:
            with open_image(asset) as image:
                images.append(transform(image))
        batch = torch.stack(images).to(mobile.device, dtype=torch.float32)
        guard()
        with torch.inference_mode():
            vector = mobile.model.avgpool(mobile.model.features(batch)).flatten(1).cpu().numpy()
        torch.cuda.synchronize()
        need(vector.dtype == np.float32 and vector.shape == (len(images), 576) and np.isfinite(vector).all(),
             "Invalid whole-image FLiD descriptor")
        output[first:first + len(images)] = vector
    return output


@dataclass(frozen=True)
class Features:
    original: np.ndarray
    original_counts: np.ndarray
    recoveries: dict
    added: np.ndarray
    added_counts: np.ndarray

    def rows(self, ids, recovered=()):
        indices = [int(key) for key in ids]
        patches = self.original[indices]
        counts = self.original_counts[indices]
        positions = {key: index for index, key in enumerate(ids)}
        for key in recovered:
            index = positions[key]
            need(counts[index] == 1 and key in self.recoveries, "Expected admitted role-specific recovery")
            patches[index] = self.recoveries[key]
            counts[index] = 10
        return patches, counts

    def added_rows(self, ids):
        positions = [int(identity.split(":")[-1]) for identity in ids]
        return self.added[positions], self.added_counts[positions]


def extract(original_rows, added_rows, weights, transform):
    """Use the exact physical batching of the original shared QUAD and FLiD measurements."""
    from r29_repro.native_encoders import NativeMobile, Offline, ResourceGuard, cyclic_pad

    need([int(row["id"]) for row in original_rows] == list(range(16260)),
         "Original FLiD documents must remain in archive order")
    need([int(row["id"].split(":")[-1]) for row in added_rows] in ([], list(range(12116))),
         "Added FLiD documents must remain in archive order")
    anchors = [row for row in original_rows if int(row["id"]) < 50]
    original_local = [row for row in original_rows[50:] if row["maps"]["original"] is not None]
    residual = [row for row in original_rows if row["maps"]["original"] is None]
    recoverable = [row for row in original_rows if row["maps"]["recovered"] is not None]
    added_local = [row for row in added_rows if row["maps"]["original"] is not None]
    added_residual = [row for row in added_rows if row["maps"]["original"] is None]
    need(len(anchors) == 50 and len(original_local) == 16023 and len(residual) == 187 and len(recoverable) == 186,
         "Unexpected original-encoding FLiD geometry roster")
    need(not added_rows or (len(added_local), len(added_residual)) == (12115, 1),
         "Unexpected expanded FLiD geometry roster")
    original = np.zeros((len(original_rows), 10, 576), dtype=np.float32)
    added = np.zeros((len(added_rows), 10, 576), dtype=np.float32)
    original_counts = np.full(len(original_rows), 10, dtype=np.int64)
    added_counts = np.full(len(added_rows), 10, dtype=np.int64)
    guard = ResourceGuard(7200)
    with Offline() as offline:
        mobile = NativeMobile(weights, guard)
        try:
            def encode(crops):
                padded = cyclic_pad(crops)
                guard(sum(crop.nbytes for crop in crops) + sum(crop.nbytes for crop in padded))
                result = mobile.encode_raw(padded)[:len(crops)]
                torch.cuda.synchronize()
                guard()
                return result

            for population, destination in ((anchors, original), (original_local, original), (added_local, added)):
                if not population:
                    continue
                vectors = local_packs(population, "original", tuple(range(11)), 32, encode, keep=TEXT)
                indices = [int(asset["id"].split(":")[-1]) for asset in population]
                # The physical packing includes the face view. The classifier uses ten text columns.
                destination[indices] = vectors
                del vectors
                print(f"FLiD measured {len(indices)} original-map bags", flush=True)
            recovered = local_packs(recoverable, "recovered", TEXT, 8, encode)
            recovery_by_id = {asset["id"]: vector for asset, vector in zip(recoverable, recovered, strict=True)}
            if added_residual:
                residual_indices = [int(asset["id"].split(":")[-1]) for asset in added_residual]
                added[residual_indices, 0] = whole_packs(added_residual, transform, mobile, guard, batch_size=1)
                added_counts[residual_indices] = 1

            # Original missing-field whole-image vectors used the author's FP32
            # transform with cuDNN TF32 disabled in physical batches 64, 64, 59.
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(False)
            residual_indices = [int(asset["id"]) for asset in residual]
            original[residual_indices, 0] = whole_packs(residual, transform, mobile, guard, batch_size=64)
            original_counts[residual_indices] = 1
        finally:
            mobile.close()
    need(offline.attempts == 0, "FLiD image extraction must stay offline")
    return Features(original, original_counts, recovery_by_id, added, added_counts)


def patch_batch(features, fold, role, device):
    ids = fold[role + "_ids"]
    patches, counts = features.rows(ids, fold[role + "_recovered_ids"])
    need({key for key, count in zip(ids, counts, strict=True) if count == 1} == set(fold[role + "_residual_ids"]),
         "Unexpected FLiD residual field eligibility")
    return ids, ScorePatches.build(patches, counts, device)


def run(runtime, meta, original_images, added_images, weights, flid_source, output, only_context=None,
        *, device_name="cuda:0"):
    registry, contexts, original, added = load_roles(runtime, meta, original_images, added_images)
    need(len(registry["field_order"]) == 11 and registry["field_order"][4] == "face", "Unexpected SVK field order")
    if only_context is not None:
        contexts = {fold: [row for row in roster if row["id"] == only_context]
                    for fold, roster in contexts.items()}
        contexts = {fold: roster for fold, roster in contexts.items() if roster}
        need(len(contexts) == 1, "Unknown SVK context")
    factory, transform = author_components(flid_source)
    output.mkdir(parents=True, exist_ok=False)
    original_rows = sorted(original.values(), key=lambda row: int(row["id"]))
    added_rows = sorted((row for group in added.values() for row in group), key=lambda row: int(row["id"].split(":")[-1]))
    features = extract(original_rows, added_rows, weights, transform)
    device = torch.device(device_name)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 1.25 * 2**30 / gpu_memory), device)
    fits, cal_packets, test_packets = {}, {}, {}
    histories = output / "history"
    histories.mkdir()
    for fold, roster in contexts.items():
        fold_models, scores = [], []
        held = registry["outer_folds"][str(fold)]
        cal_ids, calibration = patch_batch(features, held, "calibration", device)
        for row in roster:
            training = tuple(TrainingRow(t["id"], t["group"], t["role"], t["fold"], tuple(t["candidates"]))
                             for t in row["train"])
            need(Counter(t.role for t in training) == {"N": 50, "BANK": 225, "F": 15}
                 and len({t.group for t in training}) == 290
                 and [int(t.id) for t in training] == row["input_order"],
                 "Expected exact archived and component-distinct FLiD TRAIN290")
            need(all((t.fold in (0, 1) and len(t.candidates) in (1, 2)
                      and len(set(t.candidates)) == len(t.candidates) and set(t.candidates) <= set(TEXT))
                     if t.role == "F" else not t.candidates for t in training), "Expected archived field tags")
            context = TrainingContext(tuple(row["context"]), training)
            train_ids = [item.id for item in context.train]
            values, counts = features.rows(train_ids, row["recovered_train_ids"])
            model, history = fit_context(factory, context, values, counts, TEXT, device)
            scores.append(score_documents(model, calibration))
            fold_models.append(model.cpu())
            write(histories / (row["id"] + ".json"), history)
            print(f"FLiD fitted and calibrated {row['id']}", flush=True)
        fits[fold] = fold_models
        score_array = np.stack(scores)[:, None, :]
        cuts = np.sort(score_array, axis=-1)[..., np.asarray(RANKS) - 1]
        cal_packets[fold] = write_scores(output, fold, "cal", cal_ids, [row["id"] for row in roster],
                                         METHODS, score_array, cuts=cuts)
    cal_seal = output / "CALIBRATION_COMPLETE.json"
    write(cal_seal, {"methods": METHODS, "packets": cal_packets, "ascending_ranks": RANKS, "decision": "score > cut"})
    for fold, roster in contexts.items():
        held = registry["outer_folds"][str(fold)]
        test_ids, test = patch_batch(features, held, "test", device)
        extra_ids = [row["id"] for row in added[fold]]
        extra = ScorePatches.build(*features.added_rows(extra_ids), device) if extra_ids else None
        scores = []
        for model in fits[fold]:
            model = model.to(device)
            score = score_documents(model, test)
            if extra is not None:
                score = np.concatenate((score, score_documents(model, extra)))
            scores.append(score)
            model.cpu()
        ids = [f"original:{key}" for key in test_ids] + extra_ids
        test_packets[fold] = write_scores(output, fold, "test", ids, [row["id"] for row in roster],
                                          METHODS, np.stack(scores)[:, None, :])
        print(f"FLiD scored fold {fold}: {len(ids)} images", flush=True)
    write(output / "COMPLETE.json", {"schema": "R62_SVK_MATCHED_SYSTEM_SCORES_V1", "methods": METHODS,
          "cal_seal_sha256": digest(cal_seal), "test_packets": test_packets, "added_metadata_sha256": METADATA_SHA,
          "author_source_sha256": FLID_SOURCE_SHA256, "mobile_weights_sha256": digest(weights)})
