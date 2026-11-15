# Reproduce R62 from images

Use the GitHub source checkout or extract `NRSW_R62_raw_reproduction_source.zip`. Place the separate `NRSW_R62_submission_package.zip` from the R62 manuscript release or reviewer kit in the checkout. It contains the role lists, field annotations and the archived numerical references. See [datasets used in R62](datasets.md) for versioned downloads, licenses and citations. [Score replay](reproduction.md#replay-the-r62-results) checks the reported numbers from saved predictions on CPU.

## Environment

Use Python 3.14.4, the CUDA 13.0 PyTorch build, and an NVIDIA GPU. The measured platform is Windows with an RTX 5070 Ti (16 GB). The SVK extractor checks for 10 GiB of free GPU memory at startup and requires a 3 GiB free system-memory reserve during extraction. Use a working drive with at least 40 GB free for the SVK studies, encoder weights and run files. To create the environment in PowerShell, run `py -3.14 -m venv .venv-r62`, then `.venv-r62\Scripts\Activate.ps1`.

```console
python research_tools/r62_svk.py --work r62-work unpack NRSW_R62_submission_package.zip
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r r62-work/r5/package/requirements-native.txt
python -m pip install .
```

`unpack` verifies the original 360 MB submission archive and extracts the two pinned runtimes and the expected aggregate results. The R5 file pins, original geometry, candidate-field tags and splits come from these authenticated members. They include a fixed genuine reference mean measured on historical layouts. The detector and calibration states are fitted again.

## IDSpace SVK from publisher archives

```console
python research_tools/r62_svk.py --work r62-work download
python research_tools/r62_svk.py --work r62-work all
python research_tools/r62_svk.py --work r62-work jpeg NRSW_R62_submission_package.zip
```

`download` fetches the two SVK archives (6.23 GB) and the exact encoder weights (about 2.15 GB). Use `download --archives DIR --sd-snapshot DIR --mobile FILE` if you already have these assets. The extractor verifies exact bytes. `all` stages 16,260 source images, measures both encoders and runs the original-encoding complete-estimator and crossed field-weight and peak-strength contrasts for all 384 contexts. `jpeg` adds the fixed common JPEG encode, remeasures both encoders and refits all five methods from those measurements. Each method calibrates on its own genuine scores before held-out scoring. Each analysis recomputes the 5,000 grouped resamples. The JPEG path requires the Windows Pillow 12.2.0 wheel with libjpeg API 8.0 and libjpeg-turbo 3.1.4.1.

The 1%, 5% and 10% ROC values are interpolated operating points from held-out scores. Reports separately give the observed FPR and FNR at each cutoff fitted on 500 independent genuine documents.

The full runs take several hours. Use `all --context f0_d3_o0` for a smaller original-encoding run on 4,896 images. It verifies six freshly fitted score vectors and calibration arrays against the archived context fingerprints. The full original-encoding result and its verification are in `r62-work/all/refit/`. The JPEG fit results are in `r62-work/common-jpeg/refit/`, and the paired estimates and verification are in `r62-work/common-jpeg/PAIRED_RESULTS.json` and `r62-work/common-jpeg/VERIFICATION.json`.

After an interrupted extraction, repeat the same command with the same work directory. It verifies and reuses the finished segments. If a refit stops before creating `RESULTS.json`, rename only its incomplete `refit/` directory once the process exits, then repeat the command. See the [individual JPEG stages](../research_tools/jpeg/README.md) to run a stage on its own or check the transform.

Expected reductions in ROC miss rate relative to the diagonal estimator using the same field tags (percentage points):

| Evaluation | 1% FPR | 5% FPR | 10% FPR |
| --- | ---: | ---: | ---: |
| Original encoding | 18.10 | 10.86 | 7.43 |
| Common JPEG refit | 17.13 | 10.53 | 7.55 |

At 1% ROC FPR on original images, substituting the spectral peak strength with diagonal weights retained reduces the miss rate by 17.84 points. Substituting the spectral weights after that reduces it by a further 0.26 points.

The full original-encoding `all` run was verified on the pinned Windows/CUDA environment. All 16,260 images were extracted again, including the admitted field recoveries. Across 384 fitted contexts and six arms, all 10,488,960 calibration and held-out scores and 6,912 cutoffs matched the original runs. The 5,000-draw paired intervals also agreed.

The full common-JPEG `jpeg` run was verified in the same environment. Both encoders reproduced all 4,066 feature packets after common re-encoding. The five methods across 384 contexts reproduced the calibration and held-out score packets and all 5,760 cutoffs exactly. The three within-JPEG paired contrasts and the change in the main spectral-versus-diagonal contrast across encodings matched the R62 results. The reviewer kit includes both independent audit reports.

For the additional fixed-threshold JPEG sensitivity test, fit and calibrate on the original measurements and apply that detector and its original cutoff to both versions of the test images.

```console
python -m research_tools.jpeg --r5-runtime r62-work/r5 --run r62-work/common-jpeg freeze --native-measurements r62-work/all/measurements --submission NRSW_R62_submission_package.zip
```

## CrossVerify registry card from publisher images

The default run fits NRSW and four public comparators. For all five comparisons in the paper, obtain the authors' `checkpoint_t0_m2203_parent.pt` and append `--shared-parent /path/to/checkpoint_t0_m2203_parent.pt` to the command below. The [CrossVerify guide](../research_tools/crossverify/README.md) gives its SHA-256 and provenance.

```console
python -m pip install timm==1.0.31 scikit-learn==1.9.0
python research_tools/crossverify/publisher.py --r62 NRSW_R62_submission_package.zip --work r62-work
```

The command downloads the publisher's 46 parts, assembles the ZIP, acquires the pinned encoder and comparator resources, and runs all stages for the three R62 draws with 15 forged labels. The parts and joined ZIP together need 9.59 GB of space. Use `--images PATH` if you already have the assembled ZIP. Each draw uses 75 genuine calibration images and 284 held-out views of 71 records.

`r62-work/crossverify/run/comparison.json` reports the interpolated 1%, 5% and 10% pooled ROC points and paired intervals. The first nonzero FPR step is 1/71 (about 1.41%). Error rates at the separately calibrated cutoffs are also reported where a finite cutoff is available. With 75 genuine calibration images, the nominal 1% target has no finite cutoff. The three expected pooled NRSW miss rates on the ROC are 31.61%, 23.00% and 19.56%.

In the isolated Windows/CUDA verification, the six fits per draw reproduced every reported mean FNR and paired interval endpoint in both the pooled and fold-weighted analyses. Five methods matched every held score exactly. The shared-core scores differed by at most `1.95e-5`, with the same operating points and intervals. See the [comparison report](../research_tools/crossverify/verified_b15.json) and [recorded environment](../research_tools/crossverify/verified_environment.json).

## CrossVerify full-layout comparison

```console
python research_tools/crossverify/full_layout.py --r62 NRSW_R62_submission_package.zip --work r62-work
```

This reuses the publisher images and weights from the registry-card run, and evaluates both layouts with 15 training forgeries per fit. It compares NRSW, NRSW with field classifiers, DINOv3, ResNet-18 and FLiD. The source checkout includes the layout-specific role lists. The R62 manuscript release supplies the fixed historical reference mean. Allow at least 6 GB beyond the space for downloaded assets. See the [CrossVerify guide](../research_tools/crossverify/README.md) for the input check, original-score comparison and current run verification.

## IDSpace original and expanded system comparison

The [matched-system guide](../research_tools/svk_systems/README.md) stages the additional 12,116 publisher images from the same two SVK archives and runs the six model families used in the 16,210- and 28,326-image comparisons. It reuses the original-encoding SD and MobileNet descriptors. The historical shared-core arm uses the same author checkpoint named above. The added image manifest, role ordering and paired resamples have been verified. The full GPU rerun of this comparison is pending.

## Coverage

The raw runners cover the four quantitative comparisons in the R62 abstract. CrossVerify also accepts `--all-budgets` to refit the 60-, 120- and 240-forgery arms. The published image roster and score evaluation for those budgets were checked. The clean GPU verification above covers the 15-forgery comparison. The [score replay guide](reproduction.md) gives the CPU commands for the additional main-text comparisons.
