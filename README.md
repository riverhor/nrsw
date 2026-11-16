# NRSW · Document forgery detector

NRSW (Normal-Referenced Spectral Weighting) detects signs of tampering in documents with known layouts. To train a detector for a layout, provide genuine examples and a few confirmed forgeries with the suspected fields marked, such as the name or document number. To check a new document, supply its image and layout.

[繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [Train on your images](docs/training.md) · [Architecture](docs/architecture.md) · [Data formats](docs/data.md) · [Datasets](docs/datasets.md) · [Reproduce the paper](docs/reproduction.md)

## Try the demo

Open the [browser demo](https://riverhor.github.io/nrsw/#upload) and choose **Try a random sample** to run the hosted detector on a synthetic Slovak ID image from [IDSpace](docs/datasets.md). You can also upload an image of the same layout. The results show field weights, the forgery score, and the review decision. The example gallery has saved predictions for the unaltered and edited pairs.

To open the demo locally, install from this directory in a Python 3.11+ virtual environment.

```console
python -m pip install .
nrsw serve
```

Open [localhost:8000](http://127.0.0.1:8000). You can also train a small example model on simulated features with `nrsw demo --output runs/example`.

To enable uploads in your local copy, [start the demo with a registered detector](docs/upload.md).

## Reproduce the paper

The [raw-image guide](docs/raw-reproduction.md) shows how to download the datasets, extract features, train the detectors and evaluate them. The [reproduction guide](docs/reproduction.md) lists the files for each experiment and shows how to check the reported numbers on a CPU using saved predictions.

## Fit a detector

The [training guide](docs/training.md) shows how to mark fields, define a layout, and split images into training and calibration groups. The commands below use CUDA. Install the [CUDA builds of PyTorch and torchvision](https://pytorch.org/get-started/locally/) and prepare `data/roles.jsonl` and `data/layout.json` before running them.

```console
python -m pip install ".[vision]"
nrsw fit-images --manifest data/roles.jsonl --layout data/layout.json --output runs/card --device cuda --allow-download --fpr 0.05
nrsw predict-image --image data/submission.png --layout data/layout.json --model runs/card/model --calibration runs/card/cuts.json --device cuda --fpr 0.05
```

The first run downloads the pinned Stable Diffusion 1.5 and MobileNetV3-Small weights. Use `--device cpu` on both commands for CPU extraction. For offline use, pass `--sd-snapshot` and `--mobile-weights` to both commands. The fit saves cached features, a model, a calibration file and, when the manifest contains held-out documents, an evaluation report. A score above the threshold flags a document for review.

NRSW measures how unusual a change is for each field using variation in genuine documents. The document classifier uses these measurements for field weighting and to retain the strongest evidence. Add `--field-classifiers` to `fit-images` to [score individual fields as well](docs/field-classifiers.md). Compare the two detectors on validation data at your target false-positive rate.

## Use the Python API

Extract features for a new document using the same layout and encoder settings as above.

```console
nrsw extract --image data/submission.png --layout data/layout.json --output runs/queries.npz --device cuda
```

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
