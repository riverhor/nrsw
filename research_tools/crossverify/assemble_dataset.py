"""Verify locally downloaded CrossVerify v1 parts and join the publisher's ZIP."""

import argparse
from hashlib import md5, sha256
import json
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from inputs import HERE, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', type=Path, help='Directory containing the 46 publisher files')
    parser.add_argument('output', type=Path, help='New output ZIP')
    args = parser.parse_args()
    require(not args.output.exists(), f'Output already exists: {args.output}')
    manifest = read_manifest()
    partial = None
    try:
        with NamedTemporaryFile(mode='w+b', dir=args.output.parent, suffix='.partial', delete=False) as handle:
            partial = Path(handle.name)
            for item in manifest:
                source = args.parts / item['name']
                require(source.stat().st_size == item['bytes'], f'Part size differs: {source}')
                checksum = md5(usedforsecurity=False)
                with source.open('rb') as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        checksum.update(block)
                        handle.write(block)
                require(checksum.hexdigest() == item['publisher_md5'], f'Part checksum differs: {source}')
        with ZipFile(partial) as archive:
            require('dataset_crossverify/manifests/manifest.json' in archive.namelist(), 'Publisher archive missing manifest')
        # Publish the verified file on the same volume without copying or overwriting.
        if os.name == 'nt':
            partial.rename(args.output)
        else:
            os.link(partial, args.output)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    print(args.output)


def read_manifest():
    pin = '1fd394f17bd7b987e8c7607055cc2f5b0e9bac11409cbebcaff027692ceede64'
    raw = (HERE / 'publisher_parts.json').read_bytes().replace(b'\r\n', b'\n')
    require(sha256(raw).hexdigest() == pin, 'Pinned CrossVerify part manifest differs')
    return json.loads(raw)


if __name__ == '__main__':
    main()
