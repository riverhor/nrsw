# CrossVerify experiments

Start with the [image-based setup](../../docs/raw-reproduction.md#setup) and save the manuscript archive as `paper.zip`. Run the commands below from the repository root with that environment active.

## Registry card with 15 labeled forgeries

Each of the three training draws uses 50 genuine reference documents, 50 additional genuine documents and 15 forgeries with candidate-field annotations. Another 75 genuine images set the thresholds. The test set has 71 genuine and 213 copy–move images from 71 disjoint records.

```console
python -m pip install timm==1.0.31 scikit-learn==1.9.0
python research_tools/crossverify/publisher.py --submission paper.zip --work runs/paper
```

The command downloads and verifies the images and model weights, fits the detectors and writes `runs/paper/crossverify/run/comparison.json`. Add `--prepare-only` to download and verify inputs before running on a GPU. You can reuse an assembled ZIP or an extracted dataset with `--images PATH`, an extracted study runtime with `--r5 DIR` and an existing Stable Diffusion download with `--sd-snapshot DIR`.

The full comparison includes a classifier built on the historical shared core. Supply the authors' `checkpoint_t0_m2203_parent.pt` with `--shared-parent PATH`. The checkpoint was trained on 28,231 examples from four historical corpora. Its SHA-256 is `8cb8acc92d4fe1a144622ef500bf8a57a72b7ad37cab1283283547697e353947`. Recreating the historical fit itself requires the original source list and projected features from the authors. The five other classifiers use the public inputs.

Across false-positive rates of 1%, 5% and 10% on the pooled ROC curve, the fresh run reproduced NRSW miss rates of 31.61%, 23.00% and 19.56%. The first observed nonzero false-positive rate is 1/71, so the 1% ROC estimate uses interpolation. With 75 genuine calibration images, a finite one-sided 95% threshold is available at 5% and 10%. The fresh run matched the reported miss rates and paired intervals for all six methods. Five methods reproduced all calibration and test scores exactly. The historical shared-core classifier differed by at most `1.95e-5` on test scores and reproduced all decisions at its calibrated thresholds. [Comparison](verified_b15.json) · [Environment](verified_environment.json) · [Independent check](verified_b15_independent.json)

## Other label budgets

Add `--all-budgets` to repeat the experiment at 15, 60, 120 and 240 labeled forgeries. Each draw uses nested training sets and the same calibration and test sets. Results go to `runs/paper/crossverify/run-all-budgets/comparison.json`. Image-based reproduction was checked at 15 forgeries. To recalculate the reported curves and intervals at all four budgets on a CPU, run the following.

```console
python research_tools/crossverify/compare.py --submission paper.zip --check-archived-budgets --output archived-budget-check.json
```

[Saved-score check](verified_archived_budgets.json)

## Both document layouts

The photo ID and registry card experiments each use six outer folds with six training draws per fold. Each fit uses 100 genuine and 15 forged training images, 300 independent genuine calibration images and 320 test images. Each layout contributes 1,920 distinct test images. The comparison includes NRSW, NRSW with field classifiers, ResNet-18, DINOv3 and FLiD. FLiD uses its photo and text branches for the photo ID and its text branch for the registry card.

```console
python research_tools/crossverify/full_layout.py --submission paper.zip --work runs/paper
```

The command writes `runs/paper/crossverify/full-layout/comparison.json` and resumes completed stages when run again against the same directory. Allow at least 6 GB beyond the downloads. Fresh runs for both layouts reproduced the held-out scores, operating points and paired intervals for all five methods. To check the study figures on a CPU, add `--check-archived full-layout-archived.json`. [Saved-score check](verified_layout_archive.json) · [Two-fit CPU check](verified_layout_cpu_sample.json)

## Download the public inputs separately

CrossVerify v1 provides 46 download parts under CC BY 4.0. Keep about 9.6 GB free for the parts and joined archive. The joined ZIP has SHA-256 `95bac9830fb4683462d20fa5a410b6532d1607eb28a00a03441c09ccff37e02f`. [CrossVerify v1 on Zenodo](https://doi.org/10.5281/zenodo.23136723)

```console
python research_tools/crossverify/download_inputs.py parts crossverify-parts
python research_tools/crossverify/assemble_dataset.py crossverify-parts crossverify_dataset_v1.zip
python research_tools/crossverify/download_inputs.py weights weights
```

The weights command downloads the ResNet-18, MobileNet and DINOv3 checkpoints and the FLiD model source. Download Stable Diffusion separately if using `--sd-snapshot`. [Stable Diffusion download command](../../docs/reproduction.md#encoder-weights) · [ResNet-18](https://download.pytorch.org/models/resnet18-f37072fd.pth) · [MobileNetV3-Small](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) · [DINOv3](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/tree/3bf4720a82ec2066db88137180ff1f83a675cef0) · [FLiD](https://github.com/merealone2516/FLiD/tree/e432dbb401da50bfeb176ae2a081ab4e4f613dda)
