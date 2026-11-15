"""Read the registry-card splits and verify the CrossVerify v1 images."""

from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
DESIGN = HERE / 'design_b15.json'
DESIGN_SHA = 'ac7c1d48902f02e82b709de314909cbc8a83854a2ea3334fe69a485350314a7a'
FULL_DESIGN = HERE / 'design_all_budgets.json'
FULL_DESIGN_SHA = '560818d103d92ef87085b5b58b9f15e3a053f1538cb4d5884e899e91feddd6dc'
CANVAS = (1150, 520)
FIELDS = ('upper_table_content_region', 'middle_table_content_region', 'lower_table_content_region')
REGIONS = ((30, 236, 700, 294), (30, 294, 700, 352), (30, 352, 700, 410))
MODEL_IDS = tuple(f'b15_d{i}' for i in range(3))
BUDGETS = (15, 60, 120, 240)
FULL_MODEL_IDS = tuple(f'b{budget}_d{draw}' for draw in range(3) for budget in BUDGETS)


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def require(valid, message):
    if not valid:
        raise ValueError(message)


def design_file(all_budgets=False):
    return FULL_DESIGN if all_budgets else DESIGN


def context_ids(value):
    return tuple(c['id'] for c in value['contexts'])


def design(all_budgets=False):
    raw = design_file(all_budgets).read_bytes().replace(b'\r\n', b'\n')
    expected_sha, expected_ids = (FULL_DESIGN_SHA, FULL_MODEL_IDS) if all_budgets else (DESIGN_SHA, MODEL_IDS)
    require(sha256(raw).hexdigest() == expected_sha, 'Frozen portable design differs')
    value = json.loads(raw)
    require(context_ids(value) == expected_ids, 'Invalid fitted context order')
    return value


def roster(value, phase):
    return value[phase]


def feature_file(output, name, phase):
    return Path(output) / f'{name}-{phase}.npz'


def save_arrays(path, **arrays):
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


class Pixels:
    """Accept a publisher ZIP or its extracted tree. Local census staging is optional."""

    def __init__(self, path, layout):
        self.path = Path(path)
        self.archive = ZipFile(self.path) if self.path.is_file() else None
        self.layout = layout

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        if self.archive is not None:
            self.archive.close()

    def image(self, row):
        from PIL import Image

        if self.archive is not None:
            raw = self.archive.read(row['member'])
        elif self.layout == 'census':
            raw = (self.path / row['archive_name']).read_bytes()
        else:
            relative = Path(row['member'])
            path = self.path / relative
            if not path.is_file():
                path = self.path.joinpath(*relative.parts[1:])
            raw = path.read_bytes()
        identity = row['id']
        require(sha256(raw).hexdigest() == row['payload_sha256'], f'Encoded image differs: {identity}')
        with Image.open(BytesIO(raw)) as source:
            image = source.convert('RGB')
        rgb = sha256(struct.pack('>II', *image.size) + image.tobytes()).hexdigest()
        require(image.size == CANVAS and rgb == row['rgb_sha256'], f'Decoded image differs: {identity}')
        return image
