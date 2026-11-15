# Reproduce the paper's results

Use the saved predictions to check the reported numbers on a CPU. To download the source images and fit new detectors, follow the [image-based reproduction guide](raw-reproduction.md). It lists the setup commands and the experiments checked from start to finish. [Datasets and citations](datasets.md)

## Check the saved predictions

Install the package from this repository with Python 3.11 or later. Obtain the manuscript archive and its optional Slovak IDSpace score archive from the study authors or your reviewer materials. Save them locally as `paper.zip` and `idspace-scores.zip`, respectively.

```console
python -m pip install .
nrsw reproduce paper.zip
nrsw reproduce paper.zip --svk-scores idspace-scores.zip
```

| File | SHA-256 |
| --- | --- |
| `NRSW_R62_submission_package.zip` | `c5cbbd107a9774fd604eb311a59c7d87184d8e5b68a70569655ea21313f6b9d1` |
| `NRSW_R62_SVK_rank_score_replay.zip` | `fb0ece96f407bb2deb3bff8e40e7697529a0925fda6144f0f82d96b7c4003bf0` |

- **CrossVerify registry card.** Recalculates the miss rates and paired 95% intervals for six classifiers trained with 15 forged documents per fit. The evaluation uses 284 held-out images and 5,000 resamples of complete document records. At false-positive rates of 1%, 5% and 10% on the ROC curve, NRSW's pooled miss rates are 31.61%, 23.00% and 19.56%.
- **Slovak IDSpace.** The manuscript archive shows the reported summary. With the score archive, the command recalculates the ROC miss rates and the error rates at the calibrated thresholds on 16,210 held-out images. It covers the spectral estimator, the separate role of field weighting and peak strength, and the JPEG refit. The intervals come from the study archive.

Use `--json` for machine-readable output. Both commands run with NumPy and read the ZIP files directly.

## Other comparisons

Extract the relevant materials from the manuscript archive and follow the instructions alongside them.

| Study | Location inside the manuscript archive |
| --- | --- |
| NRSW, FLiD and DINOv3 on the original and expanded IDSpace test sets | `score_replay/NRSW_IDSpace_score_replay_r1.zip` |
| ResNet-18 and the historical shared core on IDSpace | `research/idspace_matched_comparators_20261112/README.md` |
| Full-layout and label-budget studies on CrossVerify | `score_replay/NRSW_R39_CrossVerify_score_replay.zip` |
| Field classifiers on both CrossVerify layouts | `field_classifier_evidence/r55_general/R55_REPRODUCTION.md` |
| Additional field-classifier runs on the registry card and IDSpace | `field_transfer_evidence/` |

For the IDSpace comparison, unpack the score ZIP into a separate directory and run the following there.

```console
python -m pip install numpy==2.4.4 psutil
python replay.py --output replay_audit.json
```

For CrossVerify, extract the manuscript archive to `manuscript` and unpack the score bundle beside it. With NumPy installed, run the following.

```console
python -I crossverify_score_replay/replay_crossverify.py --release manuscript --full-inputs crossverify_score_replay/full_inputs --layout-a-evidence manuscript/layout_a_evidence --output crossverify_verification.json
```

## Encoder weights

The image workflow uses [Stable Diffusion 1.5 at revision `451f4fe`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14) and [MobileNetV3-Small ImageNet weights](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth). The MobileNet file has SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) can fetch the Stable Diffusion files used on CUDA.

```console
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder/model.fp16.safetensors tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json unet/config.json unet/diffusion_pytorch_model.fp16.safetensors vae/config.json vae/diffusion_pytorch_model.fp16.safetensors --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 --local-dir weights/sd15
```

For CPU extraction with the package, obtain the full-precision files at the same revision. Their names are `text_encoder/model.safetensors`, `unet/diffusion_pytorch_model.safetensors` and `vae/diffusion_pytorch_model.safetensors`. Use the same weights and extraction settings when fitting and scoring a model.

## IDSpace experiment

The experiment uses the Slovak identity card in [IDSpace](datasets.md), with generated edits and simulated scans. Each fit uses 50 genuine reference images, 225 additional genuine images and 15 tagged forgeries. Each classifier calibrates its threshold on 500 independent genuine documents. The study reports both the miss rate interpolated on the ROC curve and the observed error rates at the calibrated thresholds.

### Archived workflow

Unpack the study's original software and data split from the manuscript archive.

```console
python -m nrsw.reproduce paper.zip --extract-r5 study-runtime.zip
python -m zipfile -e study-runtime.zip study-runtime
```

The extracted archive includes the split, registered field locations, field recovery data and image processing code. It has SHA-256 `1052caaed7615a5749e1e257caece9ddc38138cd3be66974039255cfe87a1724`. Follow `study-runtime/README.md` for the 4,896-image example `f0_d3_o0`. The original example reproduced the held-out decisions exactly and its scores differed by at most `7.28e-12`. See the [image-based guide](raw-reproduction.md) for the later comparison using a common scale for the spectral and diagonal estimators. For your own layout, see [training](training.md).

## Field-classifier case study

The optional extension learns a separate classifier for each field with enough tagged forgeries. Its pixel features require a crop at least 48 pixels wide and high. The extension improved detection on CrossVerify, most strongly on the photo ID layout, and increased miss rates on IDSpace.

From the extracted manuscript archive, install NumPy and run the following to recalculate the rates on the two CrossVerify layouts.

```console
python field_classifier_evidence/r55_general/replay_field_rates.py --roles field_classifier_evidence/replay/roles.json --run field_classifier_evidence/replay
```
