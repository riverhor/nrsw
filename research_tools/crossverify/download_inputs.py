"""Download the pinned public CrossVerify inputs, verifying each completed file."""

import argparse
from hashlib import file_digest
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from assemble_dataset import read_manifest
from inputs import require
from measure import DINO_SHA
from neural import FLID_SHA, RESNET_SHA


def download(url, output, expected, algorithm, size=None):
    def valid(path):
        if size is not None and path.stat().st_size != size:
            return False
        with path.open('rb') as stream:
            return file_digest(stream, algorithm).hexdigest() == expected

    if output.exists():
        require(valid(output), f'Existing file differs from published bytes: {output}')
        return
    print(f'Downloading {output.name}', flush=True)
    temporary = None
    try:
        request = Request(url, headers={'User-Agent': 'NRSW-R62-reproduction/1'})
        with urlopen(request, timeout=120) as response, NamedTemporaryFile(
                mode='wb', dir=output.parent, suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
        require(valid(temporary), f'Publisher download hash or size differs: {output}')
        temporary.rename(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=('parts', 'weights'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.target == 'parts':
        for item in read_manifest():
            key = item['name']
            url = f'https://zenodo.org/api/records/23136723/files/{key}/content'
            download(url, args.output / key, item['publisher_md5'], 'md5', item['bytes'])
    else:
        weights = (
            ('https://download.pytorch.org/models/resnet18-f37072fd.pth', 'resnet18-f37072fd.pth', RESNET_SHA),
            ('https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
             'mobilenet_v3_small-047dcff4.pth', '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'),
            ('https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/resolve/'
             '3bf4720a82ec2066db88137180ff1f83a675cef0/model.safetensors', 'model.safetensors', DINO_SHA),
            ('https://raw.githubusercontent.com/merealone2516/FLiD/e432dbb401da50bfeb176ae2a081ab4e4f613dda/'
             'flid/models.py', 'flid_models.py', FLID_SHA),
        )
        for url, name, digest in weights:
            download(url, args.output / name, digest, 'sha256')


if __name__ == '__main__':
    main()
