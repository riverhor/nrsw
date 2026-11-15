"""Compare full-layout fresh predictions with the five released R62 figure arms."""

from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np

from nrsw.reproduce import _r62_evidence, _roc_fnr
from inputs import require
from layout_fit import METHODS, RANKS


def packet(raw):
    with np.load(BytesIO(raw), allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def released_scores(evidence, roster):
    base = 'new_evidence/crossverify_nrsw/test/PAIRED_SCORES.npz'
    nrsw = packet(evidence.read(base))
    candidate = {p: packet(evidence.read(f'field_classifier_evidence/replay/{p.upper()}.npz')) for p in ('cal','test')}
    candidate_roles = json.loads(evidence.read('field_classifier_evidence/replay/roles.json'))['contexts']
    require(nrsw['context_ids'].tolist() == [c['id'] for c in roster], 'R62 NRSW full-layout context order')
    c = nrsw['arms'].tolist().index('NRSW_NATIVE')
    with ZipFile(BytesIO(evidence.read('score_replay/NRSW_R39_CrossVerify_score_replay.zip'))) as nested:
        prefix = 'crossverify_score_replay/full_inputs/'
        authority = json.loads(nested.read(prefix+'MANIFEST.json'))['files']
        witness = nested.read(prefix+'NRSW_BOOTSTRAP_AUTHORITY.npz')
        require(sha256(witness).hexdigest() == authority['NRSW_BOOTSTRAP_AUTHORITY.npz'], 'Archive bootstrap witness differs')
        bootstrap = packet(witness)['multiplicities']
        by_phase = {p: [] for p in candidate}
        for i, context in enumerate(roster):
            key = context['id']
            dino = packet(evidence.read(f'audit_evidence/full_layout_dinov3/scores/{key}.npz'))
            flid_dir = 'layout_a_evidence/flid_family_a' if context['family'] == 'family_a' else 'new_evidence/flid_registry'
            flid = packet(evidence.read(f'{flid_dir}/scores/{key}.npz'))
            relative = f'resnet_scores/{key}.npz'
            raw = nested.read(prefix+relative)
            require(sha256(raw).hexdigest() == authority[relative], 'R62 ResNet archived score hash differs')
            resnet = packet(raw)
            for phase, array in candidate.items():
                ids = [r['id'] for r in context[phase]]
                require(all(x[f'{phase}_ids'].tolist() == ids for x in (resnet, dino, flid)) and
                        nrsw[f'{phase}_ids'][i].tolist() == ids, f'{key}: archived score roster differs')
                require(array['context_ids'][i] == key and candidate_roles[i]['id'] == key and
                        array['ids'][i].tolist() == [row['id'] for row in candidate_roles[i][phase]] and
                        all(a['label'] == b['label'] for a,b in zip(context[phase],candidate_roles[i][phase],strict=True)),
                        f'{key}: candidate sequential score roster differs')
                score = nrsw['cal_scores' if phase == 'cal' else 'scores'][i,c]
                by_phase[phase].append(np.stack([score, array['scores'][i,3],
                                                *(p[f'{phase}_scores'] for p in (resnet,dino,flid))]))
    return {p: np.stack(rows) for p,rows in by_phase.items()}, bootstrap


def counts(family):
    domain = 'NRSW_CROSSVERIFY_COMPLETE_ESTIMATORS_20261109_R1'
    seed = int.from_bytes(sha256(f'{domain}|BOOTSTRAP|{family}|R1'.encode()).digest()[:8], 'big')
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.stack([rng.multinomial(80, np.full(80, 1/80), 5000) for _ in range(6)])


def evaluate(cal, test, contexts, draws):
    curves = np.zeros((len(METHODS), 5001, 3))
    fixed = np.zeros((len(METHODS), 2, 3))
    for calibrate, scores, context in zip(cal, test, contexts, strict=True):
        rows = context['test']
        labels = np.array([r['label'] for r in rows])
        roots, membership = np.unique([r['component'] for r in rows], return_inverse=True)
        require(len(roots) == 80 and np.array_equal(np.bincount(membership[labels == 0]), np.ones(80)) and
                np.array_equal(np.bincount(membership[labels == 1]), np.full(80,3)), 'Invalid test record balance')
        count = np.vstack((np.ones(80,dtype=int), draws[context['fold']]))
        cuts = np.sort(calibrate,axis=1)[:,RANKS-1]
        for i, vector in enumerate(scores):
            curves[i] += _roc_fnr(vector, labels, membership, count) / len(contexts)
            flags = vector[:,None] > cuts[i,None]
            fixed[i] += 100 * np.stack(((~flags[labels==1]).mean(0),flags[labels==0].mean(0))) / len(contexts)
    return curves, fixed


def summarize(curves, fixed, past, past_fixed):
    result = {}
    for i, method in enumerate(METHODS):
        values = curves[i]
        old = past[i]
        diff, old_diff = values-curves[0], old-past[0]
        interval = np.percentile(diff[1:], (2.5,97.5),axis=0).T
        old_interval = np.percentile(old_diff[1:], (2.5,97.5),axis=0).T
        result[method] = dict(fnr_percent=values[0].tolist(), archived_fnr_percent=old[0].tolist(),
                              point_difference_pp=(values[0]-old[0]).tolist(),
                              method_minus_nrsw_pp=diff[0].tolist(), pointwise_95_percent_pp=interval.tolist(),
                              archived_pointwise_95_percent_pp=old_interval.tolist(),
                              max_absolute_interval_difference_pp=float(np.max(np.abs(interval-old_interval))),
                              fixed_fnr_percent=fixed[i,0].tolist(), fixed_fpr_percent=fixed[i,1].tolist(),
                              archived_fixed_fnr_percent=past_fixed[i,0].tolist(), archived_fixed_fpr_percent=past_fixed[i,1].tolist())
    return result


def compare(r62, design, root=None):
    roster = design['contexts']
    with _r62_evidence(r62) as evidence:
        old, witness = released_scores(evidence, roster)
        figure = json.loads(evidence.read('field_classifier_evidence/r55_figure_stage/PIXEL_PAIRED_ENDPOINTS.json'))
        original_methods = json.loads(evidence.read('scripts/FAMILY_A_FIGURES.json'))['layouts']
    results = {}
    for order, family in enumerate(design['layouts']):
        indices = [i for i,c in enumerate(roster) if c['family'] == family]
        contexts = [roster[i] for i in indices]
        draws = counts(family)
        require(np.array_equal(draws,witness[order]), f'{family} bootstrap differs from R62')
        archived = evaluate(old['cal'][indices], old['test'][indices], contexts, draws)
        rv = figure['layout_pairs'][family]
        plot_names = ('NRSW', None, 'CrossVerify_ResNet18', 'DINOv3_frozen_dense', 'FLiD_text_shared_regions')
        plot = original_methods[family]
        for i,name in enumerate(plot_names):
            point = rv['candidate_fnr_pct'] if name is None else plot['rates_pct'][name]
            np.testing.assert_allclose(archived[0][i,0], point, rtol=0, atol=2e-10)
            if i == 0:
                continue
            paired = archived[0][i]-archived[0][0]
            expected = rv['candidate_minus_original_pp'] if name is None else plot['reduction_pp'][name]
            bounds = np.percentile(paired[1:], (2.5,97.5),axis=0).T
            plot_values = [[r['point'],r['lower95'],r['upper95']] for r in expected]
            np.testing.assert_allclose(np.column_stack((paired[0],bounds)), plot_values, rtol=0, atol=2e-10)
        current = {p: old[p][indices] for p in ('cal','test')}
        if root is not None:
            for phase in current:
                with np.load(root/f'{family}-{phase}.npz',allow_pickle=False) as arr:
                    require(arr['context_ids'].tolist() == [c['id'] for c in contexts] and arr['methods'].tolist() == list(METHODS)
                            and arr['ids'].tolist() == [[r['id'] for r in c[phase]] for c in contexts],
                            f'{family} fresh {phase} roster differs')
                    current[phase] = arr['scores']
        results[family] = summarize(*evaluate(current['cal'],current['test'],contexts,draws),*archived)
        results[family]['maximum_test_score_difference'] = dict(zip(METHODS,
             np.abs(current['test']-old['test'][indices]).max(axis=(0,2)).tolist(),strict=True))
    return dict(analysis='Archived score comparison only' if root is None else 'Fresh full-layout predictions compared with R62',
                roc_targets_percent=[1,5,10], paired_replicates=5000, ascending_cal_ranks=RANKS.tolist(), layouts=results)
