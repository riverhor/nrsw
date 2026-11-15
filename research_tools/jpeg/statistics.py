"""Grouped point and paired resampling arithmetic from the R62 common-JPEG study."""

import numpy as np


TARGETS = np.array([.01, .05, .10])


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def draw_counts(groups):
    ordered = sorted(set(groups))
    ensure(len(ordered) == 1411, "Expected the original 1,411 held components")
    generator = np.random.Generator(np.random.PCG64(20261102))
    counts = generator.multinomial(len(ordered), np.full(len(ordered), 1 / len(ordered)), size=5000)
    return ordered, counts.astype(np.int32)


def point_roc(score, label, weight):
    order = np.argsort(-score, kind="stable")
    score, label, weight = score[order], label[order], weight[order]
    end = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    fp = np.r_[0, (np.cumsum(weight * (label == 0)) / weight[label == 0].sum())[end]]
    tp = np.r_[0, (np.cumsum(weight * (label == 1)) / weight[label == 1].sum())[end]]
    fp[-1], tp[-1] = 1., 1.
    return np.trapezoid(tp, fp), 1 - np.interp(TARGETS, np.round(fp, 12), tp)


def metrics(labels, groups, scores, cuts):
    _, group = np.unique(groups, return_inverse=True)
    counts = np.stack([np.bincount(group[labels == c], minlength=group.max() + 1) for c in (0, 1)])
    weights = 1. / counts[labels, group]
    wg, wf = weights[labels == 0], weights[labels == 1]
    points = []
    for s, cut in zip(scores, cuts, strict=True):
        auc, roc_fnr = point_roc(s, labels, weights)
        fixed_fpr = np.average(s[labels == 0, None] > cut, weights=wg, axis=0)
        fixed_fnr = np.average(s[labels == 1, None] <= cut, weights=wf, axis=0)
        points.append(np.r_[auc, roc_fnr, fixed_fpr, fixed_fnr])
    return np.stack(points), np.array([wg.sum(), wf.sum()])


def merge_metrics(folds, masses):
    values, masses = np.stack(folds), np.stack(masses)
    return np.column_stack((np.average(values[:, :, 0], weights=masses.prod(axis=1), axis=0),
                            np.average(values[:, :, 1:4], weights=masses[:, 1], axis=0),
                            np.average(values[:, :, 4:7], weights=masses[:, 0], axis=0),
                            np.average(values[:, :, 7:], weights=masses[:, 1], axis=0)))


def missed_boot(score, y, weight, component, multiplicities, genuine_mass, forged_mass, targets):
    """Original genuine-block brackets with direct suffix counts of missed forgeries."""
    genuine = np.flatnonzero(y == 0)
    genuine = genuine[np.argsort(-score[genuine], kind='stable')]
    forged = np.flatnonzero(y == 1)
    g_score = score[genuine]
    ends = np.r_[np.flatnonzero(g_score[1:] != g_score[:-1]), len(genuine) - 1]
    levels = g_score[ends]
    k = len(levels)
    first_lower = np.searchsorted(-levels, -score[forged], side='right')
    at_level = np.searchsorted(-levels, -score[forged], side='left')
    tied = (at_level < k) & (levels[np.minimum(at_level, k - 1)] == score[forged])
    increments = np.zeros((k + 1, multiplicities.shape[1]))
    np.add.at(increments, (first_lower, component[forged]), weight[forged])
    # At level b, count forged scores <= level b before that tie block is added.
    missed = np.cumsum(increments[:0:-1], axis=0)[::-1]
    equal = np.zeros_like(missed)
    np.add.at(equal, (at_level[tied], component[forged[tied]]), weight[forged[tied]])
    out = np.empty((len(multiplicities), len(targets)))
    initial = max(1, int(np.searchsorted(ends, len(genuine) // 5)) + 1)
    for first in range(0, len(multiplicities), 128):
        last = min(first + 128, len(multiplicities))
        mult, gm = multiplicities[first:last], genuine_mass[first:last]
        num_blocks = min(k, initial)
        while True:
            selected = genuine[:ends[num_blocks - 1] + 1]
            cum = np.cumsum(mult[:, component[selected]] * weight[selected], axis=1)
            fp = np.round(cum[:, ends[:num_blocks]] / gm[:, None], 12)
            if np.all(fp[:, -1] > max(targets)):
                break
            ensure(num_blocks < k, 'positive upper ROC bracket in every replicate')
            num_blocks = min(k, 2 * num_blocks)
        ix = np.arange(len(mult))
        for j, target in enumerate(targets):
            hi = (fp <= target).sum(axis=1)
            lo_fp = np.where(hi > 0, fp[ix, np.maximum(hi - 1, 0)], 0.)
            alpha = (target - lo_fp) / (fp[ix, hi] - lo_fp)
            at_or_below = np.einsum('ij,ij->i', mult, missed[hi], optimize=False)
            tied_mass = np.einsum('ij,ij->i', mult, equal[hi], optimize=False)
            out[first:last, j] = at_or_below - alpha * tied_mass
    ensure(np.isfinite(out).all() and (out >= 0).all(), 'finite directly counted missed mass')
    return out


def class_weighted_parts(y, groups, member_index, counts):
    _, group = np.unique(groups, return_inverse=True)
    weight = np.empty(len(y))
    for c in (0, 1):
        sel = y == c
        weight[sel] = 1. / np.bincount(group[sel], minlength=group.max() + 1)[group[sel]]
    columns = [member_index[g] for g in np.unique(groups)]
    mult = np.ascontiguousarray(counts[:, columns], dtype=np.float64)
    mass = np.stack([mult[:, np.unique(group[y == c])].sum(axis=1) for c in (0, 1)])
    ensure((mass > 0).all(), "Both class masses remain positive in each held fold")
    return group, weight, mult, mass


def accumulate(y, group, weight, mult, mass, scores, cuts):
    contributions = np.empty((len(mult), len(scores), 9))
    for arm, (s, cut) in enumerate(zip(scores, cuts, strict=True)):
        contributions[:, arm, :3] = missed_boot(s, y, weight, group, mult, *mass, TARGETS)
        flagged = s[:, None] > cut
        for cls, wrong, start in ((0, flagged, 3), (1, ~flagged, 6)):
            indices = y == cls
            err = np.zeros((mult.shape[1], 3))
            np.add.at(err, group[indices], wrong[indices] * weight[indices, None])
            contributions[:, arm, start:start + 3] = mult @ err
    return contributions


def summarize(point, draws):
    return dict(point=point.tolist(), percentile95=np.quantile(draws, [.025, .975], axis=0).T.tolist())
