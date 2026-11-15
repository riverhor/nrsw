"""Save and load fitted parameters as JSON and numeric arrays."""
from dataclasses import fields, is_dataclass
import hashlib
import json
import numbers
from pathlib import Path

import numpy as np
from . import kernel
from .field_classifiers import METHOD, PIXEL_DIMENSION, FieldClassifierModel, FieldHead, TailReference

SCHEMA = 'nrsw.joint-model.v1'
FIELD_SCHEMA = 'nrsw.field-classifier-model.v1'
RECORDS = {c.__name__: c for c in (kernel.Layout, kernel.Teacher, kernel.JointModel,
                                  kernel.EvidenceModel, kernel.StrengthReference,
                                  kernel.AnalyticHead, FieldClassifierModel, FieldHead, TailReference)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def _validate(model, *, field_only=False):
    if type(model) is FieldClassifierModel and not field_only:
        _validate(model.document, field_only=True)
        if (not isinstance(model.training_identity, str) or len(model.training_identity) != 64
                or any(c not in '0123456789abcdef' for c in model.training_identity)
                or type(model.metadata) is not dict or model.metadata.get('method') != METHOD):
            raise ValueError('Invalid field-classifier training identity or metadata')

        def check_reference(reference):
            if (type(reference) is not TailReference or not np.isfinite([reference.upper, reference.spread]).all()
                    or reference.spread <= 0):
                raise ValueError('Invalid genuine field score reference')

        if model.document_reference is not None:
            check_reference(model.document_reference)
        for heads, width in ((model.signed_heads, model.layout.d), (model.pixel_heads, PIXEL_DIMENSION)):
            if type(heads) is not tuple or len(heads) != model.layout.m:
                raise ValueError('Field heads must follow the fitted layout')
            for branch in heads:
                if branch is None:
                    continue
                if type(branch) is not FieldHead or type(branch.head) is not kernel.AnalyticHead:
                    raise ValueError('Invalid field head')
                head = branch.head
                if (any(type(a) is not np.ndarray or a.shape != (width,) or a.dtype != np.float64
                        or not np.isfinite(a).all() for a in (head.mean, head.w))
                        or not np.isfinite([head.rms, head.b]).all() or head.rms <= 0):
                    raise ValueError('Invalid field classifier coefficients')
                check_reference(branch.reference)
        return
    if type(model) is not kernel.JointModel:
        raise TypeError('Expected the explicit current JointModel type')
    if type(model.layout) is not kernel.Layout or type(model.head) is not kernel.AnalyticHead:
        raise TypeError('Explicit layout and classifier records required')
    model.layout.validate()
    s = model.layout
    if (type(s.layout_id) is not str or type(s.extractor_id) is not str
            or type(s.fields) is not tuple or any(type(v) is not str for v in s.fields)):
        raise TypeError('Invalid layout record types')
    if type(model.metadata) is not dict or type(model.head.stats) is not dict:
        raise TypeError('Fitted metadata must be dictionaries')

    def array(value, shape):
        if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid fitted numeric state shape, dtype or value')

    def real(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
            raise ValueError('Finite real scalar required')

    expected = ((model.delta, (s.m, s.d)), (model.offset, (s.dg+s.d,)),
                (model.historical_mean, (s.dg+s.d,)),
                (model.local_mean, (s.m*s.d,)),
                (model.head.mean, (s.width,)), (model.head.w, (s.width,)))
    for value, shape in expected:
        array(value, shape)
    for value in (model.strength_unit, model.head.rms, model.head.b):
        real(value)
    if min(model.strength_unit, model.head.rms) <= 0:
        raise ValueError('Invalid fitted classifier scale')
    if type(model.teachers) is not tuple or len(model.teachers) != 3 or any(type(t) is not kernel.Teacher for t in model.teachers):
        raise TypeError('Explicit tuple of three teacher records required')
    if (tuple(t.served_fold for t in model.teachers) != (0, 1, None)
            or any(type(t.served_fold) is not int for t in model.teachers[:2])):
        raise ValueError('Two complementary teachers plus one serving teacher required')
    if (type(model.training_identity) is not str or len(model.training_identity) != 64
            or any(c not in '0123456789abcdef' for c in model.training_identity)):
        raise ValueError('Missing training identity')
    for t in model.teachers:
        e, r = t.evidence, t.strength
        if type(e) is not kernel.EvidenceModel or type(r) is not kernel.StrengthReference or type(e.stats) is not dict:
            raise TypeError('Explicit evidence and strength records required')
        if e.stats.get('residual_mode') != ('field_only' if field_only else None):
            raise ValueError('Unexpected evidence residual mode')
        if type(e.active) is not bool or type(r.active) is not bool:
            raise TypeError('Teacher activity flags must be boolean')
        for ids, minimum in ((t.fit_ids, 3), (t.bank_ids, 2)):
            if (type(ids) is not tuple or len(ids) < minimum or len(set(ids)) != len(ids)
                    or any(type(v) is not str or not v for v in ids)):
                raise ValueError('Teacher ownership must be unique nonempty ID tuples')
        if set(t.fit_ids).intersection(t.bank_ids):
            raise ValueError('Teacher support and genuine bank must be disjoint')
        array(e.means, (s.m, s.d))
        if type(e.factor) is not np.ndarray or e.factor.ndim != 2:
            raise ValueError('Evidence factor must be a matrix')
        rank = e.factor.shape[1]
        array(e.factor, (s.d, rank))
        scales = (s.m,) if e.active else (0,)
        vector = (s.d,) if e.active else (0,)
        array(e.studentizers, scales)
        array(e.leading_studentizers, scales)
        array(e.leading_vector, vector)
        if type(e.eigenvalues) is not np.ndarray or e.eigenvalues.shape not in ((s.d,), (0,)):
            raise ValueError('Invalid teacher eigenvalue vector')
        array(e.eigenvalues, e.eigenvalues.shape)
        if (np.any(np.diff(e.eigenvalues) < 0) or rank != int(np.count_nonzero(e.eigenvalues > 0))
                or (e.active and e.eigenvalues.shape != (s.d,))):
            raise ValueError('Evidence eigenvalues and factor rank disagree')
        if (e.active and rank == 0) or (not e.active and rank != 0):
            raise ValueError('Evidence factor does not match active status')
        if e.active and (np.any(e.studentizers <= 0) or np.any(e.leading_studentizers <= 0)):
            raise ValueError('Active evidence requires positive field studentizers')
        if (type(e.stats.get('positive_modes')) is not int or e.stats['positive_modes'] != rank
                or e.stats.get('fields') != s.m or e.stats.get('dimension') != s.d
                or type(e.stats.get('normal_count')) is not int or e.stats['normal_count'] < 2
                or type(e.stats.get('fake_count')) is not int or e.stats['fake_count'] < 1
                or e.stats['normal_count'] + e.stats['fake_count'] != len(t.fit_ids)):
            raise ValueError('Evidence fit counts and dimensions disagree')
        real(r.mean)
        real(r.std)
        if (r.mean < 0 or r.std < 0 or type(r.rows) is not int or r.rows < 2
                or r.rows != len(t.bank_ids) or r.active != (e.active and r.std > 0)
                or (not e.active and (r.mean != 0 or r.std != 0))):
            raise ValueError('Invalid genuine strength reference')


def export_primary(model, directory):
    """Write a new directory; returns its create-only completion record."""
    _validate(model)
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                raise ValueError('Only finite numeric arrays are admissible')
            name = f'a{len(arrays):04d}'
            arrays[name] = value.copy()
            return {'kind': 'array', 'name': name}
        if isinstance(value, np.generic):
            return encode(value.item())
        if is_dataclass(value):
            name = type(value).__name__
            if RECORDS.get(name) is not type(value):
                raise TypeError('Unregistered state type')
            return {'kind': 'record', 'name': name,
                    'fields': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
        if isinstance(value, dict):
            if not all(type(k) is str for k in value):
                raise TypeError('State dictionary keys must be strings')
            return {'kind': 'dict', 'items': [[k, encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                    'items': [encode(v) for v in value]}
        if value is None or type(value) in (str, bool, int, float):
            if type(value) is float and not np.isfinite(value):
                raise ValueError('Nonfinite scalar')
            return {'kind': 'scalar', 'value': value}
        raise TypeError('Unsupported state member: '+type(value).__name__)

    state = encode(model)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'arrays.npz', **arrays)
    _write(out / 'state.json', state)
    receipt = {'schema': FIELD_SCHEMA if type(model) is FieldClassifierModel else SCHEMA,
               'training_identity': model.training_identity,
               'arrays': {k: {'shape': list(v.shape), 'dtype': v.dtype.str,
                              'raw_sha256': hashlib.sha256(v.tobytes()).hexdigest()}
                          for k, v in arrays.items()},
               'files': {n: digest(out/n) for n in ('state.json', 'arrays.npz')}}
    _write(out / 'COMPLETION.json', receipt)
    return receipt


def import_primary(directory):
    """Read only completed, numeric state; never enables pickle."""
    folder = Path(directory)
    receipt = json.loads((folder / 'COMPLETION.json').read_text(encoding='utf-8'))
    if receipt.get('schema') not in (SCHEMA, FIELD_SCHEMA):
        raise ValueError('Unsupported model format')
    if set(receipt.get('files', {})) != {'state.json', 'arrays.npz'}:
        raise ValueError('Incomplete state inventory')
    for name, h in receipt['files'].items():
        if digest(folder/name) != h:
            raise ValueError('Changed fitted state file: '+name)
    with np.load(folder/'arrays.npz', allow_pickle=False) as archive:
        if set(archive.files) != set(receipt['arrays']):
            raise ValueError('Array inventory mismatch')
        arrays = {}
        for name, spec in receipt['arrays'].items():
            value = archive[name]
            if (value.dtype.kind not in 'biuf' or not np.isfinite(value).all()
                    or list(value.shape) != spec['shape'] or value.dtype.str != spec['dtype']
                    or hashlib.sha256(value.tobytes()).hexdigest() != spec['raw_sha256']):
                raise ValueError('Changed fitted numeric array: '+name)
            arrays[name] = value.copy()
    used = set()

    def decode(value):
        kind = value['kind']
        if kind == 'array':
            name = value['name']
            if name in used:
                raise ValueError('Unexpected shared state-array reference')
            used.add(name)
            return arrays[name]
        if kind == 'record':
            cls = RECORDS[value['name']]
            if set(value['fields']) != {f.name for f in fields(cls)}:
                raise ValueError('Dataclass field schema mismatch')
            return cls(**{k: decode(v) for k, v in value['fields'].items()})
        if kind == 'dict':
            keys = [k for k, _ in value['items']]
            if not all(type(k) is str for k in keys) or len(set(keys)) != len(keys):
                raise ValueError('Invalid dictionary keys')
            return {k: decode(v) for k, v in value['items']}
        if kind in ('tuple', 'list'):
            vals = [decode(v) for v in value['items']]
            return tuple(vals) if kind == 'tuple' else vals
        if kind == 'scalar':
            item = value['value']
            if item is not None and type(item) not in (str, bool, int, float):
                raise ValueError('Invalid scalar')
            if type(item) is float and not np.isfinite(item):
                raise ValueError('Nonfinite scalar')
            return item
        raise ValueError('Unknown state encoding')

    model = decode(json.loads((folder/'state.json').read_text(encoding='utf-8')))
    expected = FieldClassifierModel if receipt['schema'] == FIELD_SCHEMA else kernel.JointModel
    if type(model) is not expected:
        raise ValueError('Model record does not match the saved format')
    if used != set(arrays) or model.training_identity != receipt['training_identity']:
        raise ValueError('Unbound array or training identity')
    _validate(model)
    return model
