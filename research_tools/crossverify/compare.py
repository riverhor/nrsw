"""Compare CrossVerify scores with the paper's label-budget results."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from nrsw import reproduce as released
from inputs import BUDGETS, context_ids, design, design_file, digest, require


def numerical_inputs(original):
    held = original['held_canonical_order']
    position = {sid: i for i, sid in enumerate(held)}
    rows = {row['id']: row for fold in original['test_folds'].values() for row in fold}
    labels = np.array([rows[sid]['label'] for sid in held])
    _, membership = np.unique([rows[sid]['component'] for sid in held], return_inverse=True)
    rng = np.random.Generator(np.random.PCG64(5980716172743851900))
    pooled = np.zeros((5001, 71), dtype=int)
    folds = []
    for fold in original['test_folds'].values():
        indices = np.array([position[r['id']] for r in fold])
        global_ids, local_ids = np.unique(membership[indices], return_inverse=True)
        counts = released._counts(rng, len(global_ids), 5000)
        pooled[:, global_ids] = counts
        folds.append((indices, labels[indices], local_ids, counts))
    return labels, membership, pooled, folds


def evaluate(vectors, labels, membership, pooled_counts, folds):
    pooled = np.stack([released._roc_fnr(s, labels, membership, pooled_counts) for s in vectors])
    weighted = np.zeros_like(pooled)
    for indices, fold_labels, records, counts in folds:
        weight = counts.shape[1] / pooled_counts.shape[1]
        for draw, vector in enumerate(vectors):
            weighted[draw] += weight * released._roc_fnr(vector[indices], fold_labels, records, counts)
    return dict(zip(released.MODES, [pooled, weighted], strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', '--r62', dest='r62', metavar='ZIP', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-budgets', action='store_true')
    parser.add_argument('--check-archived-budgets', action='store_true',
                        help='Validate every reported budget point and interval from released scores, with no new fit')
    args = parser.parse_args()
    if not args.check_archived_budgets and args.run is None:
        parser.error('--run is required unless checking archived budgets')
    require(digest(args.r62) == released.R62_SHA256, 'The submission archive checksum does not match')
    experiment = design(args.all_budgets or args.check_archived_budgets)
    models = context_ids(experiment)
    with ZipFile(args.r62) as archive:
        evidence = released._Evidence(archive)
        original = evidence.json(f'{released.BUDGET_DIR}/DESIGN.json')
        if args.check_archived_budgets:
            validation = validate_archived_budgets(evidence, original)
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(validation, stream, indent=2, allow_nan=False)
                stream.write('\n')
            print('Verified archived FNR points and paired intervals at all four CrossVerify budgets')
            return
        metrics = numerical_inputs(original)
        held = original['held_canonical_order']
        require(held == [r['id'] for r in experiment['test']], 'Portable and original TEST order differ')
        output = {'method_scores': {}, 'analyses': {key: {} for key in released.MODES}}
        evaluated = {}
        indices = [i for i, ctx in enumerate(original['contexts']) if ctx['id'] in models]
        for method, folder, _, _ in released.METHODS:
            path = args.run / method
            if not path.is_dir():
                continue
            fit, cal, test = [path / f'{x}.json' for x in ('FIT', 'CAL', 'TEST')]
            test_record = json.loads(test.read_bytes())
            packet_path = path / 'TEST_SCORES.npz'
            require(test_record['fit_sha256'] == digest(fit) and test_record['cal_sha256'] == digest(cal)
                    and test_record['score_sha256'] == digest(packet_path), f'{method} new score receipt differs')
            with np.load(packet_path, allow_pickle=False) as packet:
                vectors = packet['scores']
                require(packet['ids'].tolist() == held and packet['context_ids'].tolist() == list(models)
                        and vectors.shape == (len(models), 284) and np.isfinite(vectors).all(), f'{method} TEST packet differs')
            with np.load(BytesIO(evidence.read(f'{folder}/TEST_SCORES.npz')), allow_pickle=False) as packet:
                previous = packet['scores'][indices]
                require(packet['ids'].tolist() == held and [packet['context_ids'][i] for i in indices] == list(models),
                        'Original score roster differs')
            evaluated[method] = (evaluate(vectors, *metrics), evaluate(previous, *metrics))
            cuts = np.array(json.loads(cal.read_bytes())['cuts'])
            decisions = vectors[:, :, None] > cuts[:, None, :]
            labels = metrics[0]
            output['method_scores'][method] = {
                'input_kind': test_record['features']['kind'], 'held_scores_exact': bool(np.array_equal(vectors, previous)),
                'maximum_absolute_score_difference': float(np.max(np.abs(vectors - previous))),
                'cal_targets_fpr': [5, 10],
                'observed_calibrated_fpr_percent_by_draw': (100 * decisions[:, labels == 0].mean(1)).tolist(),
                'observed_calibrated_fnr_percent_by_draw': (100 * (~decisions[:, labels == 1]).mean(1)).tolist(),
            }
        budgets = BUDGETS if args.all_budgets else (15,)
        output['analyses'] = summarize(evaluated, models, budgets)
        output['eligible_methods'] = [row[0] for row in released.METHODS]
        output['targets_fpr_percent'] = list(released.TARGETS)
        output['held_images'], output['held_records'] = 284, 71
        output['forged_training_images_per_draw'] = list(budgets) if args.all_budgets else 15
        output['design_sha256'] = digest(design_file(args.all_budgets))
        if args.all_budgets:
            output['context_ids'] = list(models)
    require(output['method_scores'], 'No freshly fitted methods found')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    pooled = output['analyses'][released.MODES[0]]
    for budget, rows in (pooled.items() if args.all_budgets else [(15, pooled)]):
        for method, row in rows.items():
            print(f'{budget} {method:26s} {row["fnr_percent"]} (change in pp {row["difference_pp"]})')


def interval(differences):
    return np.percentile(differences[1:], (2.5, 97.5), axis=0).T


def summarize(evaluated, models, budgets):
    analyses = {name: {} for name in released.MODES}
    for name in analyses:
        for budget in budgets:
            selected = [models.index(f'b{budget}_d{draw}') for draw in range(3)]
            group = analyses[name].setdefault(str(budget), {}) if len(budgets) > 1 else analyses[name]
            for method, (current, past) in evaluated.items():
                fresh, baseline = current[name][selected], past[name][selected]
                mean = fresh[:, 0].mean(0)
                archived_mean = baseline[:, 0].mean(0)
                row = dict(fnr_percent=mean.tolist(), archived_fnr_percent=archived_mean.tolist(),
                           difference_pp=(mean - archived_mean).tolist(), fnr_by_draw_percent=fresh[:, 0].tolist())
                if method != 'NRSW' and 'NRSW' in evaluated:
                    reference, old_reference = evaluated['NRSW']
                    paired = (reference[name][selected] - fresh).mean(0)
                    archived_paired = (old_reference[name][selected] - baseline).mean(0)
                    bounds, old_bounds = interval(paired), interval(archived_paired)
                    row.update(nrsw_minus_method_pp=paired[0].tolist(), pointwise_95_percent_pp=bounds.tolist(),
                               archived_pointwise_95_percent_pp=old_bounds.tolist(),
                               max_absolute_interval_endpoint_difference_pp=float(np.max(np.abs(bounds - old_bounds))))
                group[method] = row
    return analyses


def validate_archived_budgets(evidence, original):
    """Validate the full-budget comparison engine against authenticated result summaries."""
    metrics = numerical_inputs(original)
    models = tuple(row['id'] for row in original['contexts'])
    reference = None
    errors = {}
    for method, folder, report, contrast_key in released.METHODS:
        with np.load(BytesIO(evidence.read(f'{folder}/TEST_SCORES.npz')), allow_pickle=False) as packet:
            require(packet['context_ids'].tolist() == list(models)
                    and packet['ids'].tolist() == original['held_canonical_order'], 'Archived score roster differs')
            now = evaluate(packet['scores'], *metrics)
        if method == 'NRSW':
            reference = now
        evidence_rows = evidence.json(report)['achieved_fpr_analyses']
        maximum = 0.0
        for name, values in now.items():
            for budget in BUDGETS:
                selected = [models.index(f'b{budget}_d{draw}') for draw in range(3)]
                row = evidence_rows[name]['budgets'][str(budget)]
                observed = values[selected, 0]
                pairs = [(observed, row['draw_fnr_pct'][method]), (observed.mean(0), row['mean_fnr_pct'][method])]
                if contrast_key is not None:
                    difference = (reference[name][selected] - values[selected]).mean(0)
                    pairs += [(difference[0], row[contrast_key]), (interval(difference), row['pointwise_95pct_interval_pp'])]
                for actual, expected in pairs:
                    released._check_close(actual, expected, f'{name} b{budget} {method}')
                    maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        errors[method] = maximum
    return dict(scope='Archived score validation. No fresh higher-budget GPU fits.', budgets=list(BUDGETS),
                contexts=list(models), modes=list(released.MODES), paired_replicates=5000,
                maximum_absolute_error_by_method_pp=errors)


if __name__ == '__main__':
    main()
