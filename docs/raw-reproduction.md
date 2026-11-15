# Reproduce the experiments from images

Clone the repository or extract the source ZIP from the reviewer kit. Obtain the [manuscript archive](reproduction.md#check-the-saved-predictions) from the authors or the reviewer materials and save it in the checkout as `paper.zip`. The archive supplies the data split, field annotations and reference results. [Official downloads and dataset citations](datasets.md) · [CPU check using saved predictions](reproduction.md#check-the-saved-predictions)

## Setup

The experiments were reproduced on Windows with Python 3.14.4, CUDA 13.0 and a 16 GB RTX 5070 Ti. Start with at least 10 GiB of free GPU memory and keep 3 GiB of system memory free during feature extraction. Allow at least 40 GB of disk space for the IDSpace runs and downloads. In PowerShell, create and activate an environment with `py -3.14 -m venv .venv` and `.venv\Scripts\Activate.ps1`.

```console
python -m research_tools.svk --work runs/paper unpack paper.zip
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r runs/paper/r5/package/requirements-native.txt
python -m pip install .
```

## Slovak IDSpace

```console
python -m research_tools.svk --work runs/paper download
python -m research_tools.svk --work runs/paper all
python -m research_tools.svk --work runs/paper jpeg paper.zip
```

`download` fetches the two publisher archives (6.23 GB) and the encoder weights (about 2.15 GB). To reuse local files, pass `--archives DIR --sd-snapshot DIR --mobile FILE`. The images and weights are checked before use.

`all` extracts features from 16,260 images and compares the estimators and field-weighting variants across 384 fits. `jpeg` re-encodes the same images with common JPEG settings, extracts their features again and refits five methods. Both runs calibrate on genuine images before scoring the test set and use 5,000 paired resamples of document groups for the intervals. JPEG extraction uses Pillow 12.2.0, libjpeg API 8.0 and libjpeg-turbo 3.1.4.1.

| Output | Location |
| --- | --- |
| Original-image results and verification | `runs/paper/all/refit/` |
| JPEG refit results | `runs/paper/common-jpeg/refit/` |
| JPEG paired intervals and verification | `runs/paper/common-jpeg/PAIRED_RESULTS.json` and `runs/paper/common-jpeg/VERIFICATION.json` |

The full runs take several hours. For a smaller check, use `all --context f0_d3_o0` to reproduce the six methods on one split using 4,896 images. To resume image extraction, repeat the command with the same working directory. If fitting stops before writing `RESULTS.json`, rename the incomplete `refit/` directory after the process exits and repeat the command. [JPEG stages and restart instructions](../research_tools/jpeg/README.md)

### What to expect

At each ROC false-positive rate, the table shows how much the spectral estimator reduces the miss rate relative to the diagonal estimator fitted with the same field tags. The report also gives the observed error rates at the thresholds calibrated on 500 genuine images.

| Evaluation | 1% FPR | 5% FPR | 10% FPR |
| --- | ---: | ---: | ---: |
| Original encoding | 18.10 pp | 10.86 pp | 7.43 pp |
| Common JPEG refit | 17.13 pp | 10.53 pp | 7.55 pp |

At 1% FPR on original images, replacing diagonal peak strength with spectral peak strength reduced the miss rate by 17.84 percentage points. Using spectral field weights as well reduced it by a further 0.26 points. Fresh runs matched the reported scores, thresholds and paired comparisons for both encodings. Independent reports are available with the reviewer materials.

To test sensitivity to JPEG with the same models and thresholds, fit and calibrate on the original images and apply those detectors to both encodings. This run matched the reported error rates, paired intervals and decision changes.

```console
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/paper/common-jpeg freeze --native-measurements runs/paper/all/measurements --submission paper.zip
```

To rebuild the browser demo's image examples after the original-image run, run the following command. It refits the detector for one study split, checks the new scores and thresholds against the saved results, and exports the selected held-out images with their predictions.

```console
python -m research_tools.demo_specimens --work runs/paper --output runs/demo-specimens
```

## CrossVerify registry card

```console
python -m pip install timm==1.0.31 scikit-learn==1.9.0
python research_tools/crossverify/publisher.py --submission paper.zip --work runs/paper
```

This downloads the dataset and weights and runs the 15-forgery experiment. The downloaded parts and assembled archive together take 9.59 GB. Pass `--images PATH` to reuse an assembled archive. The test split contains 284 images from 71 records, and calibration uses 75 separate genuine images.

The default run includes NRSW, FLiD, DINOv3, ResNet-18, and a classifier fitted to the same whole-image descriptors as NRSW. To add the historical shared-core comparison, obtain the authors' `checkpoint_t0_m2203_parent.pt` and use `--shared-parent /path/to/checkpoint_t0_m2203_parent.pt`. [Checkpoint provenance and checksum](../research_tools/crossverify/README.md)

Results are written to `runs/paper/crossverify/run/comparison.json`. The expected pooled NRSW miss rates are 31.61%, 23.00% and 19.56% at 1%, 5% and 10% FPR on the ROC curve. The first observed nonzero FPR is 1/71, so the 1% value comes from interpolation. With 75 calibration images, the nominal 1% target has no finite threshold. [Comparison](../research_tools/crossverify/verified_b15.json) · [Independent verification](../research_tools/crossverify/verified_b15_independent.json)

## Other comparisons from images

To run the five-method comparison on both CrossVerify layouts, reuse the same data and weights.

```console
python research_tools/crossverify/full_layout.py --submission paper.zip --work runs/paper
```

Allow at least 6 GB beyond the downloads. The fresh full-layout run matched the original held-out scores and reported intervals for all five methods on both layouts. For other label budgets, pass `--all-budgets` to the registry-card command. The image set and saved-score evaluation were checked for all four budgets, and the 15-forgery run was reproduced from images. [CrossVerify instructions](../research_tools/crossverify/README.md)

The [expanded IDSpace instructions](../research_tools/svk_systems/README.md) cover the six-model comparison on 16,210 and 28,326 images. This run reuses the original Stable Diffusion and MobileNet features and adds images from the same two publisher archives. Its image list, data splits and evaluation from saved scores have been checked. Full training and scoring from the expanded image set are still being verified.
