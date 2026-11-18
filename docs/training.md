# Train and use a layout detector

Use Python 3.11 or newer in a virtual environment. Install the [PyTorch and torchvision builds for your device](https://pytorch.org/get-started/locally/), then run these commands from the repository root.

```console
python -m pip install ".[vision]"
nrsw --version
```

Download the [Stable Diffusion and MobileNet weights](reproduction.md#encoder-weights). Use the same weights, device, precision, batch settings and geometry for training and prediction. The commands below use CUDA and FP16 Stable Diffusion weights. For CPU, use `--device cpu --precision fp32` with the full-precision weights. Run `nrsw demo --output runs/install-check` for a quick installation check with generated features.

## Prepare your images

Prepare one layout at a time. In the [image manifest](data.md#image-manifest), use `label: 0` for genuine images and `label: 1` for forgeries. Give related images the same `group` and merge groups linked by a donor document. For each forgery eligible for training, mark one or two fields that may have been edited, such as `surname` or `given_name`. Test images need genuine or forged labels. To score a new image, supply the image and its layout.

If the manifest has no role assignments, use `prepare` to split it by group. This example uses the IDSpace training counts and 64 independent genuine calibration groups for a 5% false-positive rate target. Reserve groups with genuine images and forgeries to measure both false positives and misses.

```console
nrsw prepare --input data/all.jsonl --output data/roles.jsonl --anchors 50 --bank 225 --support 15 --calibration 64 --seed 2026
```

Check the [role assignments](data.md#image-manifest) before extracting features. Each training fold needs at least two genuine anchors and one annotated forgery with all fields available. Training also needs at least two `bank` images with all fields available. Error rates can vary substantially with datasets this small.

Create `data/layout.json` with the field names in order. Mark the field corners on at least two distinct genuine training references after correcting their orientation. If the images already align at the pixel level, use `aligned` mode with fixed coordinates. See the [layout examples](data.md#layout-and-genuine-references).

## Fit, evaluate and predict

Choose a new output directory. `fit-images` extracts features, fits NRSW and sets thresholds from the genuine calibration images. If the manifest includes held-out genuine and forged images and calibration supplies a finite threshold, it also reports test errors.

```console
nrsw fit-images --manifest data/roles.jsonl --layout data/layout.json --output runs/card --fpr 0.05 --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
nrsw predict-image --image data/query.png --layout data/layout.json --model runs/card/model --calibration runs/card/cuts.json --fpr 0.05 --output runs/card/prediction.json --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
```

- `features.npz` and `features.metadata.json` record the extracted features, extraction settings and field availability.
- Keep the entire `model/` directory and `cuts.json` together. Scores strictly above the selected threshold flag the document for review.
- `evaluation.json` gives each source group equal weight within each class when computing the false-positive rate (FPR) and false-negative rate (FNR). `FP`, `FN`, `TP` and `TN` count images.
- At the default `--delta 0.05`, calibrating thresholds for 1%, 5% and 10% FPR requires at least 299, 59 and 29 independent genuine groups, respectively. `UNAVAILABLE_NO_FINITE_RANK` means the chosen target needs more genuine calibration groups. To inspect scores alone, run `predict` on a feature file without `--calibration`.

To reuse features, run `nrsw extract`, `nrsw train`, `nrsw calibrate`, `nrsw evaluate` and `nrsw predict` separately. Each command accepts `--help`. To score several queries at once, pass an `inference` manifest to `extract --manifest`, then score the resulting file with `predict --features`. The [feature file format](data.md#feature-files) also accepts features you compute yourself.

To try NRSW with field classifiers, add `--field-classifiers` to `fit-images` and choose a new output directory. It fits classifiers for fields with enough annotated training forgeries. The pixel features require crops at least 48 pixels wide and high. `predict-image` reads the model settings and extracts the required features. See the [field-classifier case study](reproduction.md#field-classifier-case-study) for results across layouts.

## Worked IDSpace image example

The [IDSpace image guide](archived-image-workflows.md#slovak-idspace) shows how to unpack the paper's reproduction files and download the two SVK image archives. After its setup and download steps, run the following commands from the repository root. The first command prepares one split, and the tutorial uses that split's images and annotations.

```console
python -m research_tools.svk --work runs/paper prepare --context f0_d3_o0
python examples/tutorial_idspace.py --metadata runs/paper/r5/metadata --images runs/paper/f0_d3_o0/images --output data/idspace-tutorial
nrsw fit-images --manifest data/idspace-tutorial/roles.jsonl --layout data/idspace-tutorial/layout.json --output runs/idspace-tutorial --fpr 0.05 0.10 --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
nrsw extract --manifest data/idspace-tutorial/query.jsonl --layout data/idspace-tutorial/layout.json --output runs/idspace-tutorial/query.npz --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
nrsw predict --features runs/idspace-tutorial/query.npz --model runs/idspace-tutorial/model --calibration runs/idspace-tutorial/cuts.json --fpr 0.05 --output runs/idspace-tutorial/prediction.json
```

You can also pass the image path from `query.jsonl` directly to `predict-image`.

The example uses 290 training images, including 15 annotated forgeries, and 64 genuine calibration images. The test set contains 48 genuine images and 48 forgeries. Each image comes from a separate source group. The query adds a forgery from another group. The script selects images by a fixed hash of their IDs and uses the package's alignment procedure.

In a CUDA check, 447 of the 450 training, calibration and test images had all 11 fields. One `bank` image, one calibration image and one test forgery were scored with the missing-field rule. The detector flagged the additional forged query at the threshold targeting 5% FPR. After setup and image staging, the workflow took about five minutes on Windows with a 16 GB RTX 5070 Ti and FP16 Stable Diffusion weights. These rates are from the small retrospective test set.

| Nominal FPR | Genuine calibration rank (of 64) | Held-out false positives | Held-out misses |
| ---: | ---: | ---: | ---: |
| 5% | 64 | 2/48 (4.17%) | 6/48 (12.50%) |
| 10% | 62 | 5/48 (10.42%) | 6/48 (12.50%) |

## Train from archived IDSpace features

The [original IDSpace workflow](archived-image-workflows.md#slovak-idspace) uses the full test set with field recovery applied by role. After generating its `measurements/` directory, use the script below to convert the archived features while keeping the original roles, folds and annotations. Pass the unpacked `package/` directory to the script. The NRSW CLI computes the feature alignment offset from the training anchors and the supplied historical genuine mean. The archived software also supports loading the saved offset directly.

```console
python examples/tutorial_r29_features.py --companion path/to/companion/package --metadata path/to/companion/metadata --measurements path/to/companion/measurements --output runs/idspace-archived/features.npz
nrsw train --features runs/idspace-archived/features.npz --model runs/idspace-archived/model --historical-mean path/to/companion/metadata/HISTORICAL_MEAN_FP64.npy
nrsw calibrate --features runs/idspace-archived/features.npz --model runs/idspace-archived/model --output runs/idspace-archived/cuts.json
nrsw evaluate --features runs/idspace-archived/features.npz --model runs/idspace-archived/model --calibration runs/idspace-archived/cuts.json --fpr 0.05 --output runs/idspace-archived/evaluation-5pct.json
```

Repeat `evaluate` with `--fpr 0.01` and `--fpr 0.10`, using a new output path each time.

Using the 4,896 archived `f0_d3_o0` feature rows, the installed package reproduced the three thresholds fitted on 500 genuine documents and the group-balanced error rates on 4,106 held-out images. The table reports the observed error rates at those thresholds for this fit.

| Nominal FPR | Observed FPR | Observed FNR |
| ---: | ---: | ---: |
| 1% | 0.543% | 14.020% |
| 5% | 3.227% | 8.289% |
| 10% | 9.159% | 5.062% |
