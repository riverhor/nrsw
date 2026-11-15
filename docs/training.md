# Train and use a layout detector

Install Python 3.11 or newer. From a virtual environment, install the [PyTorch and torchvision builds for your device](https://pytorch.org/get-started/locally/) and then install the package and its image dependencies from the repository root:

```console
python -m pip install ".[vision]"
nrsw --version
```

Download the [pinned Stable Diffusion and MobileNet weights](reproduction.md#encoder-weights). Keep the same encoder weights, device, precision, batch settings, and geometry when fitting and predicting. The commands below use CUDA and FP16 Stable Diffusion weights. For CPU use `--device cpu --precision fp32` with the FP32 weights. For a quick installation check that uses generated descriptors and skips image extraction, run `nrsw demo --output runs/install-check`.

## Prepare your images

Use one layout ID. In [the image manifest](data.md#image-manifest), set `label` to `0` for a genuine image or `1` for a forgery. Assign the same `group` to every view tied to the same source document, and merge groups linked by a donor document. Give each forged training candidate one or two plausible field names from your layout, such as `surname` and `given_name`. Test forgeries need a class label only. A new query needs the image and layout.

If your starting manifest has no roles, choose how many distinct groups to allocate to each purpose. This example uses the IDSpace fitting budget and 64 genuine calibration groups for a nominal 5% threshold. Keep enough remaining groups with genuine examples and forgeries to evaluate both error rates.

```console
nrsw prepare --input data/all.jsonl --output data/roles.jsonl --anchors 50 --bank 225 --support 15 --calibration 64 --seed 2026
```

Review the resulting roles before extraction. `anchor` learns the genuine field reference, `bank` supplies additional genuine classifier training, `support` supplies the tagged training forgeries, `calibration` sets the decision threshold, and `evaluation` measures held-out errors. `prepare` keeps whole groups together. The smallest supported fit needs four locally observable anchors split across the two inner folds, two complete bank images, and at least one complete tagged forgery per inner fold. Using these minimums can produce unstable error rates.

Create `data/layout.json` with the ordered field names and mark their corners on at least two distinct genuine fitting reference images. Measure coordinates on the EXIF-corrected image. Choose the `aligned` mode with fixed coordinates if every image already shares the same pixel frame. See [both geometry options](data.md#layout-and-genuine-references).

## Fit, evaluate and predict

The output directory must be new. The fitting command extracts the descriptors, fits NRSW, and sets thresholds using genuine calibration rows. When evaluation rows and a finite cut are present, it also computes held-out error rates.

```console
nrsw fit-images --manifest data/roles.jsonl --layout data/layout.json --output runs/card --fpr 0.05 --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
nrsw predict-image --image data/query.png --layout data/layout.json --model runs/card/model --calibration runs/card/cuts.json --fpr 0.05 --output runs/card/prediction.json --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
```

- `features.npz` and `features.metadata.json` record the descriptors, feature configuration, and whether each document had complete local fields.
- Keep the entire `model/` directory and `cuts.json` together. A score above the selected threshold flags the document. A score equal to the threshold is accepted.
- `evaluation.json` reports observed FPR and FNR with equal mass per dependency group within each class. `FP`, `FN`, `TP`, and `TN` count images. Evaluation requires both classes.
- At the default calibration setting (`--delta 0.05`), a finite cut needs at least 299, 59, or 29 independent genuine calibration groups for nominal FPRs of 1%, 5%, or 10%, respectively. A cut with status `UNAVAILABLE_NO_FINITE_RANK` requires more genuine calibration groups at the chosen budget. Use feature-based `predict` without `--calibration` to inspect scores alone.

For reusable features or separate role files, use the individual commands `nrsw extract`, `nrsw train`, `nrsw calibrate`, `nrsw evaluate`, and `nrsw predict`. Each accepts `--help`. For multiple queries, use `extract --manifest` with an `inference` manifest, then `predict --features` with that packet. The [feature format](data.md#feature-packets) also supports descriptors supplied by your own pipeline.

To try the optional field-classifier extension, add `--field-classifiers` to `fit-images` and use a fresh output directory. The extension fits classifiers where the support tags provide enough examples for a field. Its native-pixel feature requires a crop at least 48 pixels wide and high. `predict-image` detects a model built with the extension and extracts the required features. See the [field-classifier case study](reproduction.md#field-classifier-case-study) for measured results across layouts.

## Worked IDSpace image example

For the Slovak IDSpace layout, the R29 reproduction companion includes the fixed groups, field geometry, and tagged support documents. Follow its image staging instructions for `f0_d3_o0` under **Run one complete archived context** through `r29_repro.stage_images`. The [reproduction guide](reproduction.md#idspace-experiment) identifies the companion and both licensed SVK archives. Then run this from the software repository:

```console
python examples/tutorial_idspace.py --metadata path/to/companion/metadata --images path/to/companion/images --output data/idspace-tutorial
nrsw fit-images --manifest data/idspace-tutorial/roles.jsonl --layout data/idspace-tutorial/layout.json --output runs/idspace-tutorial --fpr 0.05 0.10 --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
nrsw extract --manifest data/idspace-tutorial/query.jsonl --layout data/idspace-tutorial/layout.json --output runs/idspace-tutorial/query.npz --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
nrsw predict --features runs/idspace-tutorial/query.npz --model runs/idspace-tutorial/model --calibration runs/idspace-tutorial/cuts.json --fpr 0.05 --output runs/idspace-tutorial/prediction.json
```

You can also pass the image path from `query.jsonl` directly to `predict-image`.

The example uses all 290 fitting images from the archived context, including 15 tagged forgeries. It takes fixed subsets of 64 genuine calibration images and 48 genuine plus 48 forged evaluation images, each from its own group. It also scores one additional query from a separate group. Selection uses a fixed hash order over IDs and uses no detector scores. The workflow uses the built-in registration and anchor-derived coordinate convention.

In the CUDA verification run, 447 of 450 images had all 11 local fields. One bank, one calibration, and one held-out forged image used the global fallback. The additional forged query was flagged at the 5% nominal cut. Extraction, fitting, calibration, evaluation, and query scoring took about five minutes on Windows with a 16 GB RTX 5070 Ti and FP16 Stable Diffusion weights, after installation and image staging. The small retrospective held-out set produced these observed rates:

| Nominal FPR | Genuine calibration rank (of 64) | Held-out false positives | Held-out misses |
| ---: | ---: | ---: | ---: |
| 5% | 64 | 2/48 (4.17%) | 6/48 (12.50%) |
| 10% | 62 | 5/48 (10.42%) | 6/48 (12.50%) |

## Refit from the complete archived feature measurements

The [archived benchmark workflow](reproduction.md#archived-workflow) uses the complete evaluation set and the original role-specific field recovery. If you have built and sealed its `measurements/` store, the converter below uses the extracted R29 companion source to read and verify it. It keeps the original role membership, inner folds, and field tags. Run this in the current NRSW environment and pass the companion's unpacked `package/` directory. The current CLI recomputes the coordinate offset from the fitting anchors and the supplied historical genuine mean. The R29 companion can instead load the explicit frozen offset.

```console
python examples/tutorial_r29_features.py --companion path/to/companion/package --metadata path/to/companion/metadata --measurements path/to/companion/measurements --output runs/idspace-archived/features.npz
nrsw train --features runs/idspace-archived/features.npz --model runs/idspace-archived/model --historical-mean path/to/companion/metadata/HISTORICAL_MEAN_FP64.npy
nrsw calibrate --features runs/idspace-archived/features.npz --model runs/idspace-archived/model --output runs/idspace-archived/cuts.json
nrsw evaluate --features runs/idspace-archived/features.npz --model runs/idspace-archived/model --calibration runs/idspace-archived/cuts.json --fpr 0.05 --output runs/idspace-archived/evaluation-5pct.json
```

Repeat `evaluate` with `--fpr 0.01` and `--fpr 0.10` and new output paths for the other cuts.

An installed-package check on all 4,896 archived `f0_d3_o0` feature rows reproduced the three original CAL500 thresholds and the component-balanced errors on 4,106 held-out views. The percentages below are observed at genuine-calibrated thresholds in this single context.

| Nominal FPR | Observed FPR | Observed FNR |
| ---: | ---: | ---: |
| 1% | 0.543% | 14.020% |
| 5% | 3.227% | 8.289% |
| 10% | 9.159% | 5.062% |
