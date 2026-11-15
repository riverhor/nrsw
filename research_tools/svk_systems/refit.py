"""Use original R1-scaled NRSW, global-only and frozen-core heads for both SVK panels."""

import numpy as np
from threadpoolctl import threadpool_limits

from .added_measure import load_features
from .cohort import METADATA_SHA, require, write
from ..r62_svk import digest


RANKS = (499, 484, 462)
SEED_2203_SHA = "8cb8acc92d4fe1a144622ef500bf8a57a72b7ad37cab1283283547697e353947"
METHODS = ("NRSW", "GLOBAL_ONLY", "SHARED_CORE")


def load_design(source, meta):
    from r29_repro.cli import context, read as read_r5

    registry, _ = context(source, "f0_d3_o0")
    require(registry["field_order"] == meta["field_order"], "Added and original field order differs")
    documents = {r["id"]: r for r in read_r5(source / "documents.json")["rows"]}
    assigned = meta["r5_component_order"]
    require(assigned.keys() == documents.keys() and len(meta["rows"]) == 12116, "Incomplete roles")
    added = {f: [row for row in meta["rows"] if row["fold"] == f] for f in range(4)}
    require([len(added[f]) for f in range(4)] == [3020, 3120, 3062, 2914], "Unexpected added fold order")
    contexts = {f: [row for row in registry["contexts"] if row["context"][0] == f] for f in range(4)}
    require(all(len(c) == 96 for c in contexts.values()), "Expected 96 original contexts per outer fold")
    groups = {row["component_order"] for row in meta["rows"]}
    require(len(groups) == 1024 and not groups.intersection(assigned.values()), "Original and added components overlap")
    for fold, roster in contexts.items():
        outer = registry["outer_folds"][str(fold)]
        calibration = {documents[i]["group"] for i in outer["calibration_ids"]}
        test = {documents[i]["group"] for i in outer["test_ids"]}
        require(len(calibration) == len(outer["calibration_ids"]) == 500 and calibration.isdisjoint(test),
                "Expected independent CAL500 and TEST groups")
        for ctx in roster:
            train = ctx["train"]
            training = {row["group"] for row in train}
            require(len(training) == 290 and training.isdisjoint(calibration | test)
                    and all(documents[row["id"]]["group"] == row["group"]
                            and documents[row["id"]]["label"] == (row["role"] == "F") for row in train),
                    "Expected matched and component-disjoint TRAIN, CAL and TEST")
    return registry, contexts, documents, added


def project(parent):
    import torch
    from torch.nn import functional as f

    require(digest(parent) == SEED_2203_SHA, "Expected the seed-2203 historical parent checkpoint")
    torch.set_num_threads(2)
    payload = torch.load(parent, weights_only=True, map_location="cpu")
    require(payload["key"] == "t0_m2203_parent" and payload["representation"] == "sd_grid",
            "Expected the native historical SD/grid parent")
    weights = payload["model_state"]
    linear, bias, norm_w, norm_b = (weights[name] for name in (
        "core.linear.weight", "core.linear.bias", "core.normalization.weight", "core.normalization.bias",
    ))

    def transform(aligned):
        with torch.inference_mode():
            hidden = f.gelu(f.linear(torch.from_numpy(aligned), linear, bias), approximate="none")
            return f.layer_norm(hidden, (128,), norm_w, norm_b, 1e-5).numpy().astype(np.float64)

    return transform


def aligned(batch, offset):
    raw = np.concatenate((batch.global_features, batch.global_mobile), axis=1)
    return (raw.astype(np.float64) - offset).astype(np.float32)


def add_batch(model_layout, rows, values):
    from r29_repro.kernel import FeatureBatch

    ix = [row["id"] for row in rows]
    return FeatureBatch(model_layout, tuple(f"test_scale_20261107:{i}" for i in ix),
                        tuple(f'added:{row["component_order"]}' for row in rows),
                        values["sd512"][ix], values["original_local512"][ix], values["grid512"][ix],
                        values["original_valid"][ix])


def write_scores(output, fold, phase, ids, contexts, methods, scores, **extra):
    require(np.isfinite(scores).all(), "Expected finite SVK scores")
    name = f"{phase}{fold}.npz"
    path = output / name
    with path.open("xb") as stream:
        np.savez_compressed(stream, ids=ids, contexts=contexts, methods=methods, scores=scores, **extra)
    return {"file": name, "sha256": digest(path)}


def run(runtime, original, added_features, meta, output, methods, parent=None, only_context=None):
    from r29_repro import kernel
    from ..svk_measurements import Measurements

    source = runtime / "metadata"
    reference = np.load(source / "HISTORICAL_MEAN_FP64.npy", allow_pickle=False)
    registry, contexts, documents, added = load_design(source, meta)
    if only_context is not None:
        contexts = {f: [r for r in c if r["id"] == only_context] for f, c in contexts.items()}
        contexts = {f: c for f, c in contexts.items() if c}
        require(len(contexts) == 1, "Unknown SVK context")
    data = Measurements(source, original)
    require(all(row["id"] in data.contexts for c in contexts.values() for row in c),
            "Native measurement store lacks a requested context")
    core = project(parent) if "SHARED_CORE" in methods else None
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    limits, cal_packets, test_packets = {}, {}, {}
    with threadpool_limits(limits=2, user_api="blas"):
        if core:
            scope = registry["contexts"] if only_context is None else [c for roster in contexts.values() for c in roster]
            train_ids = sorted({row["id"] for c in scope for row in c["train"]}, key=int)
            projected = core(global_rows(data, train_ids))
            train_pos = {key: index for index, key in enumerate(train_ids)}
        for fold, roster in contexts.items():
            fits = []
            for ctx in roster:
                training = data.training(ctx["id"])
                frame = aligned(training.features, data.offset)
                fits.append(tuple(
                    kernel.fit_joint(training, reference, frozen_offset=data.offset)[0] if method == "NRSW"
                    else kernel.fit_head(projected[[train_pos[i] for i in training.features.ids]]
                                         if method == "SHARED_CORE" else frame, training.labels)
                    for method in methods
                ))
            models[fold] = fits
            print(f"FIT fold {fold}: {len(fits)} contexts", flush=True)
        model_root = output / "models"
        model_root.mkdir()
        from r29_repro import state

        for fold, roster in contexts.items():
            for ctx, fitted in zip(roster, models[fold], strict=True):
                location = model_root / ctx["id"]
                location.mkdir()
                analytic = {}
                for method, model in zip(methods, fitted, strict=True):
                    if method == "NRSW":
                        state.export_primary(model, location / "NRSW")
                    else:
                        for key in ("mean", "rms", "w", "b"):
                            analytic[f"{method}_{key}"] = getattr(model, key)
                if analytic:
                    with (location / "analytic.npz").open("xb") as stream:
                        np.savez_compressed(stream, **analytic)
        fit = output / "FIT_COMPLETE.json"
        write(fit, {"methods": methods, "contexts": {f: [r["id"] for r in c] for f, c in contexts.items()},
                    "original_measurements_sha256": data.input_sha256,
                    "model_files_sha256": {p.relative_to(model_root).as_posix(): digest(p)
                                           for p in model_root.rglob("*") if p.is_file()},
                    "shared_checkpoint_sha256": SEED_2203_SHA if core else None})

        def score(batch, fold, hidden=None):
            frame = aligned(batch, data.offset)
            scores = []
            for fit_models in models[fold]:
                result = []
                for method, fitted in zip(methods, fit_models, strict=True):
                    if method == "NRSW":
                        chunks = (batch.subset(np.arange(i, min(i + 64, len(batch))))
                                  for i in range(0, len(batch), 64))
                        result.append(np.concatenate([fitted.score(block) for block in chunks]))
                    else:
                        result.append(fitted.score(hidden if method == "SHARED_CORE" else frame))
                scores.append(np.stack(result))
            return np.stack(scores)

        if core:
            cal_ids = sorted({i for fold in contexts for i in registry["outer_folds"][str(fold)]["calibration_ids"]}, key=int)
            cal_pos = {key: index for index, key in enumerate(cal_ids)}
            cal_hidden = core(global_rows(data, cal_ids))
        for fold, roster in contexts.items():
            keys = [row["id"] for row in roster]
            cal = data.batch(keys[0], "calibration")
            require(all(documents[x]["label"] == 0 for x in cal.ids) and len(set(cal.groups)) == len(cal) == 500,
                    "Expected independent genuine CAL500 components")
            scores = score(cal, fold, cal_hidden[[cal_pos[i] for i in cal.ids]] if core else None)
            limits[fold] = np.sort(scores, axis=-1)[..., np.asarray(RANKS) - 1]
            cal_packets[fold] = write_scores(output, fold, "cal", cal.ids, keys, methods, scores, cuts=limits[fold])
            print(f"CAL fold {fold}: {len(cal)} images", flush=True)
        cal_seal = output / "CALIBRATION_COMPLETE.json"
        write(cal_seal, {"fit_sha256": digest(fit), "packets": cal_packets, "ascending_ranks": RANKS,
                         "decision": "score > cut"})

        if added_features is not None:
            extra, measurement_receipts = load_features(added_features, meta)
        else:
            extra, measurement_receipts = None, None
        for fold, roster in contexts.items():
            keys = [row["id"] for row in roster]
            test = data.batch(keys[0], "test")
            ids = [f"original:{i}" for i in test.ids]
            extended = add_batch(test.layout, added[fold], extra) if extra is not None else None
            hidden = None
            if core:
                frame = aligned(test, data.offset)
                if extended is not None:
                    frame = np.concatenate((frame, aligned(extended, data.offset)))
                hidden = core(frame)
            scores = score(test, fold, hidden[:len(test)] if core else None)
            if extra is not None:
                extra_scores = score(extended, fold, hidden[len(test):] if core else None)
                ids.extend(extended.ids)
                scores = np.concatenate((scores, extra_scores), axis=-1)
            test_packets[fold] = write_scores(output, fold, "test", ids, keys, methods, scores)
            print(f"TEST fold {fold}: {len(ids)} images", flush=True)
    write(output / "COMPLETE.json", {"schema": "R62_SVK_MATCHED_SYSTEM_SCORES_V1",
          "methods": methods, "cal_seal_sha256": digest(cal_seal), "test_packets": test_packets,
          "added_metadata_sha256": METADATA_SHA, "added_measurement_receipts": measurement_receipts})


def global_rows(data, ids):
    # The archived core evaluated the entire phase union together. Its
    # CPU linear operator can depend on matrix shape. A one-context witness
    # lacking the other contexts' inputs instead measures its own available union.
    positions = [data.positions[key] for key in ids if key in data.positions]
    require(len(positions) == len(ids), "Historical core requires the exact TRAIN and CAL global unions")
    raw = np.column_stack((data.sd[positions], data.mobile[positions]))
    return (raw.astype(np.float64) - data.offset).astype(np.float32)
