# Research reproduction

Start with the [raw-image reproduction guide](raw-reproduction.md) for data acquisition, feature extraction, fitting, calibration and evaluation. See [datasets used in R62](datasets.md) for citations, pinned downloads and licenses.

## Replay the R62 results

Use Python 3.11 or later. Install the package from this repository and place the original R62 submission ZIP alongside the score companion if both are available from the manuscript materials. Run the following from any directory with the appropriate ZIP paths.

```console
python -m pip install .
nrsw reproduce NRSW_R62_submission_package.zip
nrsw reproduce NRSW_R62_submission_package.zip --svk-scores NRSW_R62_SVK_rank_score_replay.zip
```

| Input | SHA-256 |
| --- | --- |
| `NRSW_R62_submission_package.zip` (360,651,154 bytes) | `c5cbbd107a9774fd604eb311a59c7d87184d8e5b68a70569655ea21313f6b9d1` |
| `NRSW_R62_SVK_rank_score_replay.zip` (33,127,143 bytes) | `fb0ece96f407bb2deb3bff8e40e7697529a0925fda6144f0f82d96b7c4003bf0` |

- **CrossVerify registry card.** The command recomputes the six classifiers' mean FNRs and paired pointwise intervals at 1%, 5% and 10% ROC FPR, with 15 forged training images per fit. It uses the original scores for 284 held-out images, both pooled and fold-weighted analyses, and 5,000 paired whole-record resamples. The pooled NRSW miss rates are 31.61%, 23.00% and 19.56%.
- **IDSpace SVK.** The optional companion replays the ROC and calibrated error rates for 16,210 held-out images and 384 contexts. It recomputes the ROC FNR and the observed FPR and FNR at each independently calibrated cutoff for the complete-estimator comparison, crossed weights and strength ablation, and common JPEG refit. The complete spectral estimator lowers FNR by 18.10, 10.86 and 7.43 points relative to the same-tag diagonal estimator. The companion preserves score ties, class/component weights and strict cutoff decisions. IDSpace intervals come from the archived estimates. With the submission ZIP alone, the command checks and displays the archived IDSpace aggregates.

Use `--json` to save the rates and intervals or compare them programmatically. The replay uses NumPy and reads the authenticated ZIPs directly. It needs no image downloads, model checkpoints or GPU. These are evaluation replays of fixed studies. For training from images, see below.

## Additional score replays

The submission ZIP also has these companions and their own instructions. Extract the relevant nested ZIPs and follow their `README.md` in the extracted evidence directories.

| Study | Path inside the R62 submission ZIP |
| --- | --- |
| Matched IDSpace NRSW, FLiD and DINOv3 on the original and enlarged populations | `score_replay/NRSW_IDSpace_score_replay_r1.zip` |
| IDSpace ResNet-18 and frozen historical shared core | `research/idspace_matched_comparators_20261112/README.md` |
| CrossVerify matched full-layout and forged-label budget studies | `score_replay/NRSW_R39_CrossVerify_score_replay.zip` |
| Field classifiers across the two CrossVerify layouts, with matching calibration and held-out roles | `field_classifier_evidence/r55_general/R55_REPRODUCTION.md` |
| Additional field-classifier runs for the registry-card budget and IDSpace | `field_transfer_evidence/` |

For the matched IDSpace analysis, unpack `NRSW_IDSpace_score_replay_r1.zip` into its own directory. From that directory, install the replay dependencies and run the bundled evaluator.

```console
python -m pip install numpy==2.4.4 psutil
python replay.py --output replay_audit.json
```

For the full-layout CrossVerify comparison, extract the R62 ZIP to `R62` and the R39 CrossVerify companion beside it. Install NumPy and run the released replay script with both sets of inputs.

```console
python -I crossverify_score_replay/replay_crossverify.py --release R62 --full-inputs crossverify_score_replay/full_inputs --layout-a-evidence R62/layout_a_evidence --output crossverify_verification.json
```

## Refit the original detector from images

The original joint input retains ordered signed fields, a weighted pool, peak strength and both global descriptors. The IDSpace layout has 11 fields and 512 coordinates per encoder, giving 7,681 classifier inputs. The archived fitting route uses separately acquired source images and weights.

## Encoder weights

Obtain the [Stable Diffusion 1.5 snapshot at revision `451f4fe`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14) and the [torchvision MobileNetV3-Small ImageNet checkpoint](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth). Keep the original model licenses alongside the files. The MobileNet checkpoint SHA-256 is `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

For CUDA with FP16 SD weights, the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) can download just the required files to a regular local directory. Run the following as one command:

```console
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder/model.fp16.safetensors tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json unet/config.json unet/diffusion_pytorch_model.fp16.safetensors vae/config.json vae/diffusion_pytorch_model.fp16.safetensors --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 --local-dir weights/sd15
```

For the CPU path, obtain the corresponding FP32 files at the same revision: `text_encoder/model.safetensors`, `unet/diffusion_pytorch_model.safetensors`, and `vae/diffusion_pytorch_model.safetensors`. The extractor checks the weights and records the runtime and preprocessing settings in each feature packet. Use the same settings for all packets that share a model.

## IDSpace experiment

The primary experiment uses the Slovak identity-card layout in [IDSpace at revision `a3de23b`](https://huggingface.co/datasets/cactuslab/IDSpace/tree/a3de23b950d55a39fbd36cfaee16bd260c4d457d). The corpus contains synthetic document edits and software-simulated scans. The exact archives are `SVK_template.tar.gz` (SHA-256 `2ef411ce08069ed5bf859a2aa294ff951ef8c1fa908ad16d91ba52f0ba9142c7`) and `SVK_scan.tar.gz` (SHA-256 `fe12457c293480aafbe82d743872207041fd8c0148ee3dc21ea1e24f2869a99f`).

Each fitted context uses 50 genuine anchors, 225 genuine bank documents, and 15 forged support documents. The original study averages across fixed outer splits and support draws. Its test curves compare component-balanced forgery miss rates at common achieved false-positive rates of 1%, 5%, and 10%. The study also estimates thresholds from 500 independent genuine calibration documents for each classifier. The CLI `evaluate` command with `--calibration` reports the observed error rates at the chosen calibrated threshold.

### Archived workflow

Copy the authenticated nested runtime to a new ZIP file, then unpack it.

```console
python -m nrsw.reproduce NRSW_R62_submission_package.zip --extract-r5 NRSW_R29_portable_reproduction_r5.zip
python -m zipfile -e NRSW_R29_portable_reproduction_r5.zip r5-runtime
```

The R5 companion (SHA-256 `1052caaed7615a5749e1e257caece9ddc38138cd3be66974039255cfe87a1724`) contains split metadata, registered geometry, role-specific field recovery, the frozen coordinate offset and a pinned image pipeline. Follow `r5-runtime/README.md` to acquire the exact assets, stage the designated images, and run `f0_d3_o0` through feature extraction, training, calibration and test scoring. The archived validation covered all 4,896 images in that context, with identical calibrated decisions and a maximum score difference of `7.28e-12` against the original run. The companion also describes how to execute the remaining contexts. Its original NRSW strength scale precedes the common-scale complete-estimator and JPEG refits.

The complete-estimator design and independent audits are in `submission_evidence/svk/` within the submission ZIP. The crossed and JPEG protocols and audits are in `reviewer_evidence_r52/`, and the CrossVerify budget protocol is in `crossverify_label_budget/`. Follow the [raw-image commands](raw-reproduction.md) to refit these experiments from the publisher images. The ordinal score companion serves the separate CPU replay command. The [dataset guide](datasets.md) lists each publisher's download and citation.

For a new layout, follow the [training guide](training.md). Use separate connected components for model fitting, genuine calibration and held-out evaluation.

## Field-classifier case study

The R62 archive contains the evidence for the field-classifier extension evaluated on two CrossVerify layouts and the Slovak IDSpace layout. It fits separate MobileNet and native-pixel scores for fields with enough tagged training forgeries. The pixel feature requires a crop at least 48 pixels wide and high. The extension improved detection on CrossVerify, with the largest gains on the photo ID, and increased miss rates on IDSpace.

From the extracted R62 submission archive, install NumPy and replay the saved CrossVerify A and B operating points.

```console
python field_classifier_evidence/r55_general/replay_field_rates.py --roles field_classifier_evidence/replay/roles.json --run field_classifier_evidence/replay
```

The separate 284-image CrossVerify label-budget evaluation and the original and expanded IDSpace evaluations are documented in `field_transfer_evidence/`. Those study scripts use the archived role and geometry files, model states and score packets. The reproduction guides specify the required inputs for refitting. The 1% CrossVerify budget result comes from ROC interpolation over the held records, while the IDSpace evaluation also reports observed rates at separately calibrated thresholds.
