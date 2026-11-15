# NRSW

Normal-Referenced Spectral Weighting trains a document forgery detector for a known layout. It uses genuine examples and a small number of confirmed forgeries with coarse field tags, such as `name` or `document_number`. To check a new document, supply its image and layout.

[繁體中文](README.zh-TW.md) · [Train on your images](docs/training.md) · [Data formats](docs/data.md) · [Dataset downloads and citations](docs/datasets.md) · [Reproduce the paper](docs/reproduction.md)

## Try the demo

Install from this directory in a Python 3.11+ virtual environment.

```console
python -m pip install .
nrsw serve
```

Open [localhost:8000](http://127.0.0.1:8000) to see how field evidence and the calibrated threshold affect decisions. The demo runs on CPU with generated descriptors. To save its model, data, and metrics, run `nrsw demo --output runs/example`.

## Reproduce the paper

Use the [raw-image guide](docs/raw-reproduction.md) to download the exact datasets, extract features, and train and evaluate the main R62 experiments. For a CPU check of the reported numbers from saved predictions, place the two evidence archives in the current directory and run:

```console
nrsw reproduce NRSW_R62_submission_package.zip --svk-scores NRSW_R62_SVK_rank_score_replay.zip
```

The [reproduction guide](docs/reproduction.md) lists each supported comparison, the files it requires, and which results can be recomputed from saved predictions or from a new fit.

## Fit a detector

The [training guide](docs/training.md) shows how to annotate candidate fields, define layout coordinates, prepare independent fitting and calibration groups, and obtain the encoder weights. After preparing `data/roles.jsonl` and `data/layout.json`, run:

```console
python -m pip install ".[vision]"
nrsw fit-images --manifest data/roles.jsonl --layout data/layout.json --output runs/card --device cuda --allow-download --fpr 0.05
nrsw predict-image --image data/submission.png --layout data/layout.json --model runs/card/model --calibration runs/card/cuts.json --device cuda --fpr 0.05
```

The first run downloads the pinned Stable Diffusion 1.5 and MobileNetV3-Small weights. Use `--device cpu` on both commands for CPU extraction. For offline use, pass `--sd-snapshot` and `--mobile-weights` to both commands. The fit saves cached features, a model, a calibration file and, when the manifest contains held-out documents, an evaluation report. A score above the threshold flags a document for review.

The default detector is the paper's NRSW. It estimates local evidence relative to genuine variation and uses both field weights and peak evidence in the document classifier. Add `--field-classifiers` to `fit-images` to try the [section 3.7 extension](docs/section-3-7.md), which also scores individual fields. Use validation data at your target false-positive rate to choose between the two.

## Use the Python API

```python
from nrsw import calibrate, fit, load_feature_dataset, predict, save_model

data = load_feature_dataset("runs/card/features.npz")
model = fit(data)
cuts = calibrate(model, data, fprs=(0.05,))
save_model(model, "runs/another-fit")
queries = load_feature_dataset("runs/queries.npz")
results = predict(model, queries, calibration=cuts, fpr="0.05")
```

For several layouts, use [`LayoutRegistry`](docs/layouts.md). Each layout has its own geometry, model and threshold. One registry instance shares the visual encoders across layouts.

## License and citation

NRSW is released under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) to cite the software and [third-party notices](THIRD_PARTY_NOTICES.md) for dataset and weight terms.
