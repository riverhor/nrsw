"""Export the verified image list for the 15-label CrossVerify comparison."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from inputs import FULL_MODEL_IDS, MODEL_IDS, design_file, digest, require


PINS = dict(design='0b53382533121df4cbd1c8d8525b4519cece2e28ad89007ded4c7ad66059c19e',
            census='1915eea3c00f3a531cbce29f97bc40417317911fd34e2d1bb8b719afa237b199',
            prepare='c73a31bf4ee20ccad47c0a191b1d1fbedd4882c1c054bb23717ea04a252cb468',
            mobile='9bb3ab4ddd07244509747908f1a2f16a3b8a9799c121af75768f415dde2115bf',
            r5='1052caaed7615a5749e1e257caece9ddc38138cd3be66974039255cfe87a1724')


def main():
    from hashlib import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    for key in PINS:
        parser.add_argument(f'--{key}', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--all-budgets', action='store_true')
    args = parser.parse_args()
    for name, expected in PINS.items():
        require(digest(getattr(args, name)) == expected, f'Archived {name} differs')
    old = json.loads(args.design.read_bytes())
    census = {r['sample_id']: r for r in json.loads(args.census.read_bytes())['rows']}
    prepared = json.loads(args.prepare.read_bytes())
    train = {}
    contexts = []
    for context in old['contexts']:
        key = context['id']
        if key not in (FULL_MODEL_IDS if args.all_budgets else MODEL_IDS):
            continue
        training = []
        for r in context['train']:
            sid = r['id']
            candidate = prepared['tags'].get(sid)
            train[sid] = r['component']
            training.append(dict(id=sid, component=r['component'], role=r['role'],
                                 fold=prepared['nrsw_forged_folds'][key].get(sid, r['inner_fold']),
                                 candidates=candidate['candidates'] if candidate is not None else []))
        contexts.append(dict(id=key, draw=context['draw'], train=training))
    held = {r['id']: r for group in old['test_folds'].values() for r in group}

    def image_row(identity, component):
        row = census[identity]
        require(row['record_id'] == component and row['family'] == 'family_b', f'Bad census join: {identity}')
        return dict(id=identity, component=component, member=row['member'], payload_sha256=row['payload_sha256'],
                    rgb_sha256=row['rgb_sha256'], archive_name=Path(row['path']).name)

    mobile = json.loads(args.mobile.read_bytes())
    selected = set(train) | set(held) | {r['id'] for r in old['cal']}
    chunks = []
    for desc in mobile['chunks']:
        path = Path(desc['path'])
        require(digest(path) == desc['sha256'], f'Bad chunk metadata: {path}')
        ids = json.loads(path.read_bytes())['ids']
        if set(ids) & selected:
            chunks.append([sid if sid in selected else None for sid in ids])
    require({sid for ids in chunks for sid in ids if sid is not None} == selected, 'Incomplete archived chunk map')
    with ZipFile(args.r5) as archive:
        files = ('__init__.py', 'native_encoders.py', 'mapped_sd.py', 'sd_bundle.py', 'readonly_leaf_loader.py')
        runtime = {name: sha256(archive.read('package/r29_repro/' + name)).hexdigest() for name in files}
    value = dict(sources=PINS, native_r5_sha256=runtime, contexts=contexts, mobile_chunks=chunks,
                 train=[image_row(sid, train[sid]) for sid in sorted(train)],
                 cal=[image_row(r['id'], r['component']) for r in old['cal']],
                 test=[image_row(sid, held[sid]['component']) for sid in old['held_canonical_order']])
    with (args.output or design_file(args.all_budgets)).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, separators=(',', ':'))
        handle.write('\n')


if __name__ == '__main__':
    main()
