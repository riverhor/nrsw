# Reproduce the paper's results

Use the separately supplied `NRSW_research_data.zip` to check the manuscript's saved-score comparisons on a CPU. Obtain the data from the study authors or your reviewer materials. It is distributed separately from this software repository. Source images, external weights and the historical shared-core checkpoint are not included in the research-data bundle. [Datasets and citations](datasets.md) · [Training from publisher images](raw-reproduction.md)

## Check the saved predictions

Extract the research-data bundle into a new directory. With Python 3.11 or newer, run:

```console
python -m zipfile -e NRSW_research_data.zip study-data
cd study-data
python -m pip install numpy==2.4.4 psutil
python replay.py --list
python replay.py --verify
python replay.py --study full-layout --work replay_work --output full_layout_results.json
```

The launcher verifies all bundle members against the supplied SHA-256 manifest and uses the included source directly. Use a new working directory and output file for each study. These commands need neither images nor a GPU.

| Study selector | Saved-score comparison |
| --- | --- |
| `primary` | Registry-card classifiers at 15 forged training documents; IDSpace complete evidence estimators, crossed weights and strength, and common JPEG control |
| `label-budgets` | Registry-card classifiers at all four forged-label budgets |
| `full-layout` | Five methods on both CrossVerify layouts; Figure 10 and Supplementary Figure S6 |
| `expanded-idspace` | NRSW, DINOv3 and FLiD on original and expanded IDSpace test populations |
| `field-classifiers` | Field-classifier extension across registry-card label budgets and the original, added and combined IDSpace populations |

Replace `full-layout` in the example with the selected study and choose different output paths. The expanded-IDSpace replay independently checks 16 fixed bootstrap replicates by default; add `--full-replay` to independently recompute all 5,000 replicates.

### Figure 10 and Supplementary Figure S6

The `full-layout` study uses **NRSW with field classifiers as the reference**. Each paired difference is a comparator's FNR minus the reference FNR. A positive value means the comparator misses more forgeries. The command verifies 30 absolute FNR estimates, 24 paired differences and 48 pointwise 95% interval endpoints at achieved ROC FPR targets of 1%, 5% and 10%.

Each layout contains 36 fixed contexts, 480 records and 1,920 distinct test images. All methods use the same 5,000 resamples of complete records within each test fold. Differences are formed before taking the 2.5th and 97.5th percentiles. The intervals are conditional on the saved fits and assignments; they are pointwise intervals. The exact plotted reference values are in `replay_work/crossverify_and_idspace/full_layout_reference_expected.csv` after the run.

### What is recomputed

The primary replay recalculates registry-card rates and paired intervals for six classifiers at the 15-forgery budget. The pooled NRSW FNRs are 31.61%, 23.00% and 19.56% at ROC FPR targets of 1%, 5% and 10%. Its IDSpace companion recalculates ROC and calibrated-threshold point estimates from ordinal scores; the IDSpace interval endpoints are archived estimates.

The field-classifier replay recomputes registry-card intervals from the original 5,000 record resamples. Its IDSpace intervals use the supplied paired replicate estimates. Each package README describes its coverage and input checks.

## Use the installed package for the primary study

After extracting the research-data bundle, unpack its primary kit. From `study-data`, run:

```console
python -m zipfile -e primary/NRSW_IPM_reviewer_reproduction.zip primary-kit
python -m zipfile -e primary-kit/NRSW_IPM_data.zip primary-data
nrsw reproduce primary-kit/NRSW_IPM_data.zip --svk-scores primary-data/svk_rank_scores.zip
```

Install this repository with `python -m pip install .` before using `nrsw`. The command verifies the exact archive and its member hashes; `--json` returns machine-readable results. It also accepts the original archived study and score companion for existing workflows.

| Primary-study input | SHA-256 |
| --- | --- |
| `NRSW_IPM_data.zip` | `4c4a1b88c872712a542c64fc78bb6bb63f6a60b69d6ac0c4b848e5c7342a7229` |
| `svk_rank_scores.zip` | `02ac5cf8d8da91f877cc09bea39fdca3602b99a30ec8c480a8fbbb896541058b` |

## Interpret the estimator comparisons

The complete spectral-versus-diagonal comparison changes the evidence estimator supplying **both field weights and peak strength**, then refits the receiving document classifier. The spectral FNR reductions are 18.10/10.86/7.43 percentage points on original IDSpace images and 17.13/10.53/7.55 points after common JPEG re-encoding and refitting, at the three ROC targets. These differences measure the complete estimator change.

The crossed experiment separates the two channels. At 1% FPR on original images, spectral peak strength with diagonal field weights reduces FNR by 17.84 percentage points; spectral field weights provide a further 0.26 points conditional on spectral strength. Pooling-only controls keep the strength channel fixed and refit their receiving classifier. Their results answer the narrower question about field weighting.

## Field-classifier case study

The extension learns field classifiers when enough tagged training forgeries have usable measurements. It refits its document branch using field-specific genuine residuals, standardizes branch scores using genuine training rows, and calibrates the final maximum score on separate genuine documents. The extension and the original detector need their own thresholds.

The extension improved detection on CrossVerify, especially the photo ID layout, and increased miss rates on IDSpace. Use the `full-layout` and `field-classifiers` selectors for the corresponding saved results. See [field-classifier implementation checks](field-classifiers.md).

## Encoder weights

The image workflow uses [Stable Diffusion 1.5 at revision `451f4fe`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14) and [MobileNetV3-Small ImageNet weights](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth). The MobileNet file has SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) can fetch the Stable Diffusion files used on CUDA.

```console
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder/model.fp16.safetensors tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json unet/config.json unet/diffusion_pytorch_model.fp16.safetensors vae/config.json vae/diffusion_pytorch_model.fp16.safetensors --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 --local-dir weights/sd15
```

For CPU extraction, obtain the full-precision files at the same revision: `text_encoder/model.safetensors`, `unet/diffusion_pytorch_model.safetensors` and `vae/diffusion_pytorch_model.safetensors`. Use the same weights and extraction settings for fitting and scoring.

The software retains its MIT license. The study-data bundle provides its own rights statement; the software license assigns no additional rights to numerical data, publisher images or external weights.
