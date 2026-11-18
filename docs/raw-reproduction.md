# Reproduce the experiments from images

Obtain the [study-data bundle](reproduction.md#check-the-saved-predictions) separately. Its primary kit includes matching source code, study data, frozen role assignments and instructions for the primary raw-image experiments. Use that source with its accompanying data archive. The repository's preserved runners use the earlier archive and are documented in [preserved image workflows](archived-image-workflows.md).

## Setup

From a directory containing `NRSW_research_data.zip`, extract the primary kit and its source:

```console
python -m zipfile -e NRSW_research_data.zip study-data
python -m zipfile -e study-data/primary/NRSW_IPM_reviewer_reproduction.zip primary-kit
python -m zipfile -e primary-kit/NRSW_IPM_source.zip primary-kit/nrsw_source
cd primary-kit/nrsw_source
```

The reported image reproductions used Windows, Python 3.14.4, CUDA 13.0 and a 16 GB RTX 5070 Ti. Use a separate virtual environment. Allow at least 40 GB of disk, 10 GiB of free GPU memory and 3 GiB of free system memory for IDSpace.

```console
python -m research_tools.svk --work runs/paper unpack ../NRSW_IPM_data.zip
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r runs/paper/image_runtime/package/requirements-native.txt
python -m pip install .
```

## Slovak IDSpace

Run these commands from the extracted `nrsw_source` directory:

```console
python -m research_tools.svk --work runs/paper download
python -m research_tools.svk --work runs/paper all
python -m research_tools.svk --work runs/paper jpeg ../NRSW_IPM_data.zip
```

The downloader checks publisher archives and pinned visual weights. To reuse files, pass `--archives DIR --sd-snapshot DIR --mobile FILE` to `download`. The complete native and common-JPEG comparisons use 384 fitted contexts each. For a smaller qualification, add `--context f0_d3_o0` to `all`. Follow the primary kit README for resume instructions and the fixed-model JPEG comparison.

The estimator comparison changes both field weights and peak strength. The separate crossed analysis and pooling-only comparisons have narrower interpretations; see [comparison scope](reproduction.md#interpret-the-estimator-comparisons).

## CrossVerify registry card

From the same extracted source directory, run:

```console
python -m pip install timm==1.0.31 scikit-learn==1.9.0
python research_tools/crossverify/publisher.py --data ../NRSW_IPM_data.zip --work runs/paper
```

This downloads the publisher data and weights, then fits NRSW, FLiD, DINOv3, ResNet-18 and the whole-image control at 15 forged labels. The download parts and assembled archive together require about 9.6 GB. Use `--images PATH` to reuse an assembled publisher ZIP. The historical shared-core classifier additionally needs the separately supplied checkpoint through `--shared-parent PATH`; its saved-score comparison needs no checkpoint.

The test split contains 284 images from 71 records; 75 separate genuine images calibrate thresholds. The first nonzero pooled ROC FPR is 1/71, so the reported 1% ROC estimate uses interpolation. The nominal 1% calibration target has no finite threshold with 75 genuine documents.

## Coverage

The original native IDSpace, common-JPEG IDSpace and 15-forgery registry-card raw workflows completed and were checked against the saved results. The matching source derivative preserves the numerical algorithms and data inputs; its packaging checks cover score replay and unpacking of raw inputs. The current bundle provides score-level replay for the full-layout and expanded-population comparisons. See the [five study selectors](reproduction.md#check-the-saved-predictions) for their coverage.

For your own images and layout, use the current package's [training guide](training.md). To continue an existing run against the earlier study archive, use the [preserved repository workflows](archived-image-workflows.md), whose runtime names and checksums remain fixed.
