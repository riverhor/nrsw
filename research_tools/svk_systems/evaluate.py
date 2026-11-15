"""Evaluate detector scores with equal weight for source groups."""

from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from .cohort import read, require, write
from .refit import RANKS, load_design
from ..jpeg.statistics import accumulate, class_weighted_parts, merge_metrics, metrics
from ..r62_svk import digest


PANELS = ("original", "added", "combined")
ENDPOINTS = ("roc_FNR", "own_cut_FPR", "own_cut_FNR")
PUBLISHED_TABLE = {
    "NRSW": ([11.24, 6.69, 4.92], [11.50, 6.92, 5.11]),
    "RESNET18": ([98.34, 92.82, 86.28], [98.40, 93.05, 86.54]),
    "DINOV3": ([96.49, 87.80, 79.72], [96.56, 88.08, 80.08]),
    "FLID": ([64.37, 54.81, 48.51], [64.71, 55.14, 48.85]),
    "GLOBAL_ONLY": ([67.81, 56.51, 50.08], [67.98, 56.60, 50.27]),
    "SHARED_CORE": ([81.11, 66.10, 57.73], [81.15, 66.25, 58.08]),
}


def bootstrap(groups, cohorts, folds):
    owners = {}
    for group, cohort, fold in zip(groups, cohorts, folds, strict=True):
        key = cohort, fold
        require(owners.setdefault(group, key) == key, "Component crosses cohort or fold")
    by_stratum = defaultdict(list)
    member_index = {group: i for i, group in enumerate(sorted(owners))}
    for key in member_index:
        by_stratum[owners[key]].append(member_index[key])
    draws = np.empty((5000, len(owners)), dtype=np.int32)
    rng = np.random.Generator(np.random.PCG64(20261107))
    for _, group_indices in sorted(by_stratum.items()):
        n = len(group_indices)
        draws[:, group_indices] = rng.multinomial(n, np.full(n, 1.0 / n), size=5000)
    return member_index, draws


def truth(runtime, meta):
    registry, contexts, docs, added = load_design(runtime / "metadata", meta)
    result = {}
    for fold in range(4):
        old = registry["outer_folds"][str(fold)]
        ids = old["test_ids"]
        extra = added[fold]
        components = [meta["r5_component_order"][i] for i in ids]
        components.extend(row["component_order"] for row in extra)
        result[fold] = {
            "cal_ids": old["calibration_ids"],
            "contexts": [row["id"] for row in contexts[fold]],
            "ids": [f"original:{i}" for i in ids] + [f'test_scale_20261107:{row["id"]}' for row in extra],
            "labels": np.array([docs[i]["label"] for i in ids] + [row["label"] for row in extra], np.uint8),
            "groups": np.asarray(components, np.int32),
            "cohorts": np.asarray(["original"] * len(ids) + [row["cohort"] for row in extra]),
        }
    return result


def load_fold(run, fold, info):
    seal = read(run / "COMPLETE.json")
    cal_seal = run / "CALIBRATION_COMPLETE.json"
    require(seal["cal_seal_sha256"] == digest(cal_seal), "Changed CAL receipt")
    cal = read(cal_seal)
    arms = seal["methods"]
    arrays = []
    for role, ids, desc in (("cal", info["cal_ids"], cal["packets"][str(fold)]),
                            ("test", info["ids"], seal["test_packets"][str(fold)])):
        path = run / f"{role}{fold}.npz"
        require(desc["file"] == path.name and desc["sha256"] == digest(path), f"Changed {role} score packet")
        with np.load(path, allow_pickle=False) as packet:
            scores = packet["scores"]
            require(packet["ids"].tolist() == ids and packet["contexts"].tolist() == info["contexts"]
                    and packet["methods"].tolist() == arms and scores.shape == (96, len(arms), len(ids))
                    and scores.dtype == np.float64 and np.isfinite(scores).all(), f"Unexpected {role} score ABI")
            if role == "cal":
                cuts = packet["cuts"]
                require(np.array_equal(cuts, np.sort(scores, axis=-1)[..., np.array(RANKS) - 1]),
                        "Changed own CAL500 cutoff")
            arrays.append(scores)
    return arms, arrays[-1], cuts


def run(runtime, meta, runs, output):
    rows = truth(runtime, meta)
    arrays = {fold: [load_fold(run, fold, info) for run in runs] for fold, info in rows.items()}
    methods = [name for names, _, _ in arrays[0] for name in names]
    require(len(methods) == len(set(methods)) and set(methods) <= PUBLISHED_TABLE.keys() and "NRSW" in methods,
            "One run per known arm and original NRSW are required")
    coordinates = lambda name: np.concatenate([value[name] for value in rows.values()])
    groups, cohorts = coordinates("groups"), coordinates("cohorts")
    folds = np.concatenate([np.full(len(info["labels"]), f) for f, info in rows.items()])
    member_index, counts = bootstrap(groups, cohorts, folds)
    require(len(member_index) == 2435, "Unchanged combined component universe")
    outcome = {}
    with threadpool_limits(limits=2, user_api="blas"):
        for panel in PANELS:
            points, masses = [], []
            sums = np.zeros((len(counts), len(methods), 9))
            den = np.zeros((2, len(counts)))
            for fold, info in rows.items():
                require([name for names, _, _ in arrays[fold] for name in names] == methods,
                        "All folds must have identical method order")
                mask = np.ones(len(info["labels"]), bool)
                if panel != "combined":
                    mask = (info["cohorts"] == "original") == (panel == "original")
                y, components = info["labels"][mask], info["groups"][mask]
                scores = np.concatenate([s for _, s, _ in arrays[fold]], axis=1)[..., mask]
                cuts = np.concatenate([t for _, _, t in arrays[fold]], axis=1)
                parts = class_weighted_parts(y, components, member_index, counts)
                den += parts[-1]
                p, mass = [], None
                for s, t in zip(scores, cuts, strict=True):
                    point, mass = metrics(y, components, s, t)
                    p.append(point)
                    sums += accumulate(y, *parts, s, t) / 96
                points.append(np.mean(p, axis=0))
                masses.append(mass)
                print(f"{panel} fold {fold}: 96 contexts", flush=True)
            values = merge_metrics(points, masses)[:, 1:].reshape(-1, 3, 3)
            replicates = (sums / np.repeat(den[[1, 0, 1]].T, 3, axis=1)[:, None, :]).reshape(-1, len(methods), 3, 3)
            index = methods.index("NRSW")
            result = {}
            for i, name in enumerate(methods):
                point = values[i]
                difference = values[index] - point
                ci = np.quantile(replicates[:, index] - replicates[:, i], (.025, .975), axis=0).transpose(1, 2, 0)
                result[name] = dict(zip(ENDPOINTS, point.tolist(), strict=True))
                if i != index:
                    result[name]["nrsw_minus_method"] = dict(zip(ENDPOINTS, difference.tolist(), strict=True))
                    result[name]["pointwise95"] = dict(zip(ENDPOINTS, ci.tolist(), strict=True))
            outcome[panel] = result
    published_checks = {}
    for method in methods:
        published_checks[method] = {}
        for name, expected in zip(("original", "combined"), PUBLISHED_TABLE[method], strict=True):
            generated = np.round(np.asarray(outcome[name][method]["roc_FNR"]) * 100, 2).tolist()
            published_checks[method][name] = {"published_fnr_percent": expected,
                                               "regenerated_fnr_percent": generated,
                                               "matches_at_two_decimals": expected == generated}
    write(output, {"schema": "R62_SVK_MATCHED_SYSTEM_RATES_V1", "methods": methods, "targets": [0.01, 0.05, 0.10],
                   "bootstrap_replicates": 5000, "results": outcome, "main_table_comparison": published_checks})
