"""Export the authenticated CrossVerify full-layout roles and image provenance."""

import argparse
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

from build_design import PINS
from inputs import HERE, digest, require


LAYOUT_PINS = {**{k: PINS[k] for k in ('census', 'mobile', 'r5')},
               'roles': 'c4838142f4406491e67939a5359a8a077334182b3a2af1eba5d857c85bcf7f44',
               'tags': '067281da021cb4fa2c82bcf7e2e17b3e304054411413575e79cf444dc899a180',
               'geometry': '61cdba6ad9f2c5f9f8540f13ded2e3e30b361732d2b6e0d45d1aec6ed6af062d'}


def main():
    from hashlib import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    for name in LAYOUT_PINS:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'layout_design.json.gz')
    args = parser.parse_args()
    for name, expected in LAYOUT_PINS.items():
        require(digest(getattr(args, name)) == expected, f'Archived {name} differs')
    census = {r['sample_id']: r for r in json.loads(args.census.read_bytes())['rows']}
    layouts = {r['layout_id']: r for r in json.loads(args.geometry.read_bytes())['layouts']}
    role_file = json.loads(args.roles.read_bytes())
    tags = json.loads(args.tags.read_bytes())['tags']
    contexts = []
    selected = set()
    for old in role_file['contexts']:
        record = {key: old[key] for key in ('id', 'family', 'fold', 'draw')}
        for phase in ('train', 'cal', 'test'):
            rows = []
            for row in old[phase]:
                sid = row['id']
                source = census[sid]
                require(source['family'] == old['family'] and row['component'] == source['record_id'],
                        'Census and role identity differ')
                label = int(row['role'] == 'F') if phase == 'train' else row.get('label', 0)
                require(source['tamper_type'] == ('visual_splice' if label else 'genuine'), 'Census label differs')
                r = dict(id=sid, component=row['component'], label=label)
                if phase == 'train':
                    r.update(role=row['role'], fold=row['inner_fold'], candidates=tags[sid]['candidates'] if label else [])
                selected.add(sid)
                rows.append(r)
            record[phase] = rows
        contexts.append(record)
    images = {sid: {k: census[sid][k] for k in ('member', 'payload_sha256', 'rgb_sha256', 'canvas', 'family')}
              for sid in sorted(selected)}
    chunks = {family: [] for family in layouts}
    for item in json.loads(args.mobile.read_bytes())['chunks']:
        path = Path(item['path'])
        require(digest(path) == item['sha256'], 'Changed original Mobile chunk')
        chunk = json.loads(path.read_bytes())
        ids = chunk['ids']
        if selected.intersection(ids):
            chunks[chunk['layout_id']].append([sid if sid in images else None for sid in ids])
    require({sid for family in chunks.values() for chunk in family for sid in chunk if sid is not None} == selected,
            'Incomplete selected Mobile chunks')
    with ZipFile(args.r5) as archive:
        files = ('__init__.py', 'native_encoders.py', 'mapped_sd.py', 'sd_bundle.py', 'readonly_leaf_loader.py')
        runtime = {name: sha256(archive.read('package/r29_repro/' + name)).hexdigest() for name in files}
    design = dict(source_digests=LAYOUT_PINS, layouts=layouts, contexts=contexts, images=images,
                  mobile_chunks=chunks, native_r5_sha256=runtime)
    payload = (json.dumps(design, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with args.output.open('xb') as output, gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as handle:
        handle.write(payload)


if __name__ == '__main__':
    main()
