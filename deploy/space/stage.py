"""Assemble the public Space from source and a fitted synthetic-ID detector."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile


SPACE = Path(__file__).resolve().parent
REPOSITORY = SPACE.parents[1]
DETECTOR_FILES = (
    "layouts.json",
    "layout.json",
    "cuts.json",
    "model/state.json",
    "model/arrays.npz",
    "model/COMPLETION.json",
    "references/reference-1.jpg",
    "references/reference-2.jpg",
    "ATTRIBUTION.md",
    "EXTRACTOR.json",
)


def stage(detector, output):
    detector = detector.resolve(strict=True)
    output = output.resolve()
    sources = {name: SPACE / name for name in ("app.py", "requirements.txt", "README.md")}
    sources["LICENSE"] = REPOSITORY / "LICENSE"
    for source in sorted((REPOSITORY / "src" / "nrsw").glob("*.py")):
        sources[f"nrsw/{source.name}"] = source
    manifest = (detector / "FILES.json").resolve(strict=True)
    if not manifest.is_relative_to(detector):
        raise ValueError("Detector file inventory points outside the fitted model directory")
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    if set(files) != set(DETECTOR_FILES):
        raise ValueError("Detector file inventory does not match the public deployment files")
    sources["detector/FILES.json"] = manifest
    for name in DETECTOR_FILES:
        source = (detector / name).resolve(strict=True)
        if not source.is_relative_to(detector):
            raise ValueError(f"Detector file points outside the fitted model directory: {name}")
        data = source.read_bytes()
        if files[name] != {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}:
            raise ValueError(f"Detector file differs from the audited copy: {name}")
        sources[f"detector/{name}"] = source
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    output.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, type=Path, help="Audited fitted IDSpace detector directory")
    parser.add_argument("--output", required=True, type=Path, help="New directory to upload as a Space")
    args = parser.parse_args()
    stage(args.detector, args.output)


if __name__ == "__main__":
    main()
