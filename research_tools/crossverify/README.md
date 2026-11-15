# CrossVerify registry-card reproduction from images

This run fits six classifiers on the frozen R62 15-forgery design. Each of three draws fits on 50 genuine anchors, 50 genuine bank documents and 15 tagged forgeries. The common 75 genuine calibration images and 284 held views belong to disjoint records. The held set has 71 genuine and 213 copy–move views of 71 records. The runner generates the ordered signed, whole-image and field descriptors from the publisher's PNGs. The archived 15-label design supplies the fixed region geometry, admitted TRAIN candidate tags and balanced TRAIN folds. The other forged-label budgets in the figure have their own saved score replay in the R62 archive.

## Environment and publisher inputs

Use the released source checkout and the R62 submission ZIP. The recorded native run used Python 3.14.4, CUDA 13.0 and an RTX 5070 Ti (16 GB). Extract the authenticated R5 companion and install its native requirements, the CrossVerify comparator dependencies and this source. Run these commands from the project source root in an activated virtual environment.

```console
python -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install .
python -m nrsw.reproduce NRSW_R62_submission_package.zip --extract-r5 NRSW_R29_portable_reproduction_r5.zip
python -m zipfile -e NRSW_R29_portable_reproduction_r5.zip r5-runtime
python -m pip install -r r5-runtime/package/requirements-native.txt timm==1.0.31 scikit-learn==1.9.0
```

## Run from the publishers

```console
python research_tools/crossverify/publisher.py --r62 NRSW_R62_submission_package.zip --work r62-work --r5 r5-runtime
```

This acquires and verifies the remaining public assets and writes `r62-work/crossverify/run/comparison.json`. The same command finds `r62-work/r5` if you unpacked it with the SVK workflow. Omit `--r5` to extract that runtime automatically. Use `--images` and `--sd-snapshot` for existing assets, and `--prepare-only` to acquire and check everything before starting GPU work. Add `--shared-parent PATH` if you have the historical checkpoint described below.

## Acquire the public inputs separately

Download the 46 CrossVerify v1 parts and verify each advertised file size and checksum. The full joined ZIP is 4,794,551,940 bytes in addition to the downloaded parts. The assembled SHA-256 is `95bac9830fb4683462d20fa5a410b6532d1607eb28a00a03441c09ccff37e02f`. After assembly, the runner verifies the encoded and decoded bytes of each selected image. You can use the assembled ZIP or an extracted publisher tree with the same contents. The dataset is available under CC BY 4.0. [CrossVerify v1 on Zenodo](https://doi.org/10.5281/zenodo.23136723)

```console
python research_tools/crossverify/download_inputs.py parts crossverify-parts
python research_tools/crossverify/assemble_dataset.py crossverify-parts crossverify_dataset_v1.zip
python research_tools/crossverify/download_inputs.py weights weights
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 model_index.json scheduler/scheduler_config.json text_encoder/config.json text_encoder/model.fp16.safetensors tokenizer/merges.txt tokenizer/special_tokens_map.json tokenizer/tokenizer_config.json tokenizer/vocab.json unet/config.json unet/diffusion_pytorch_model.fp16.safetensors vae/config.json vae/diffusion_pytorch_model.fp16.safetensors --revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 --local-dir weights/sd15
```

The download command verifies the official torchvision ImageNet ResNet-18 and MobileNetV3-Small checkpoints, the pinned timm DINOv3 weights, and `flid/models.py` from the FLiD authors' repository. Keep the publisher licenses alongside those files. [ResNet-18 weights](https://download.pytorch.org/models/resnet18-f37072fd.pth), [MobileNetV3-Small weights](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth), [DINOv3 snapshot](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/tree/3bf4720a82ec2066db88137180ff1f83a675cef0), [FLiD pinned source](https://github.com/merealone2516/FLiD/tree/e432dbb401da50bfeb176ae2a081ab4e4f613dda), [SD v1.5 snapshot](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14)

The frozen shared-core arm requires the paper authors' historical parent checkpoint `checkpoint_t0_m2203_parent.pt` (585,333 bytes, SHA-256 `8cb8acc92d4fe1a144622ef500bf8a57a72b7ad37cab1283283547697e353947`). The parent was trained on 28,231 rows from four historical source corpora. Its provenance is documented in R62. Recreating that upstream fit also requires the historical source roster and projected feature packets, which the public materials do not include. Ask the paper authors for the pinned parent if this arm is needed. Pass `--shared-parent /path/to/checkpoint_t0_m2203_parent.pt` to include its freshly fitted CrossVerify heads. The other five arms run with the public inputs above. The historical genuine mean used by NRSW, global-only and the shared core is included under `r5-runtime/metadata/`.

## Refit and compare with prepared inputs

```console
python research_tools/crossverify/run.py --r62 NRSW_R62_submission_package.zip --images crossverify_dataset_v1.zip --r5-runtime r5-runtime/package --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth --dino-weights weights/model.safetensors --resnet-weights weights/resnet18-f37072fd.pth --flid-source weights/flid_models.py --historical-mean r5-runtime/metadata/HISTORICAL_MEAN_FP64.npy --output crossverify-run
```

The command writes measured descriptors, fresh fitted states, independent CAL scores and held scores to a new directory. It verifies the R62 ZIP and computes paired 5,000 whole-record bootstrap intervals from the new predictions. `comparison.json` gives archived and newly computed ROC miss rates, differences in points and interval endpoints, the score difference, and realized FPR/FNR using the independent CAL cuts at nominal 5% and 10%. The ROC 1% point interpolates below the first nonzero empirical pooled FPR of 1/71. CAL75 admits a finite one-sided 95% threshold at nominal 5% or 10% FPR.

Run `measure.py`, `refit.py` or `compare.py` with `--help` to execute the stages separately. The inference uses phase-specific image rosters and original MobileNet batch shapes and row positions. Fixed filler rasters preserve positions for unselected images in the relevant batches. The fitter uses only selected TRAIN rows. The full `run.py` seals all fits before measuring CAL and seals all CAL scores before measuring TEST.

## Verified run

The publisher command completed with all six arms in a fresh Python environment with the official pinned CUDA wheels. It used the joined publisher ZIP and the authors' historical checkpoint. All 18 pooled and 18 fold-weighted mean FNRs and all 60 paired interval endpoints matched R62. Five methods reproduced every CAL and TEST score exactly. The frozen shared core had a maximum TEST score difference of `1.95e-5` and reproduced every held decision at the calibrated cutoffs. NRSW's pooled miss rates were **31.61%, 23.00% and 19.56%**. [Comparison](verified_b15.json) · [Environment](verified_environment.json) · [Independent audit](verified_b15_independent.json)

The earlier `nrsw reproduce NRSW_R62_submission_package.zip` command remains available for the score-level R62 replay without CUDA or external weights.

### Additional budgets

```console
python research_tools/crossverify/publisher.py --r62 NRSW_R62_submission_package.zip --work r62-work --all-budgets
```

The optional four-budget run uses 627 distinct TRAIN images across 12 fits per method. It preserves each draw's nested 15-, 60-, 120- and 240-forgery rosters, their candidate fields and complementary TRAIN folds, and the same CAL75 and TEST284. It writes `r62-work/crossverify/run-all-budgets/comparison.json`. Add `--shared-parent PATH` to include the shared-core arm. The full set of 986 publisher images and the comparison code were verified. The fresh GPU verification above covers only 15 labels.

To check the four published curves and their paired intervals on CPU, replay the authenticated archived scores.

```console
python research_tools/crossverify/compare.py --r62 NRSW_R62_submission_package.zip --check-archived-budgets --output archived-budget-check.json
```

[Archived score verification](verified_archived_budgets.json)

## Full-layout A/B comparison

The separate full-layout runner uses the frozen five-method figure design. Each layout uses six outer folds and six draws, 100 genuine and 15 forged TRAIN images per fit, 300 genuine CAL images and 320 TEST images per context. Each layout contributes 1,920 distinct held views. The five arms are original NRSW, NRSW with signed MobileNet and pixel field classifiers, ResNet-18, DINOv3, and FLiD. Layout A uses the FLiD photo and text branches. Layout B uses its text branch on the three table regions. The historical genuine mean is authenticated from the R62 R5 runtime. This figure has no historical shared-core arm.

```console
python research_tools/crossverify/full_layout.py --r62 NRSW_R62_submission_package.zip --work r62-work --prepare-only
python research_tools/crossverify/full_layout.py --r62 NRSW_R62_submission_package.zip --work r62-work --check-archived full-layout-archived.json
python research_tools/crossverify/full_layout.py --r62 NRSW_R62_submission_package.zip --work r62-work
```

Use `--images`, `--r5` or `--sd-snapshot` to supply prepared assets. The fit writes `r62-work/crossverify/full-layout/comparison.json`. Repeating the command resumes from authenticated completed measurement packets, fitted contexts and phase receipts. It opens TEST measurements only after the complete CAL stage seals. The archived score check reproduces the published pooled-over-fold-and-draw FNR points and all paired interval endpoints from the five stored score sets. The publisher preflight verifies the selected 3,880 images and model inputs. Two fresh CPU refits used archived SD and MobileNet measurements and freshly extracted pixel features, with CAL and TEST score differences at most `4.593e-11` across the original and field-extension detectors. The full 72-context GPU route still needs an end-to-end run. Allow roughly 45–60 minutes on the recorded RTX 5070 Ti and at least 6 GB of free disk once assets are staged. [Archived figure check](verified_layout_archive.json) · [Two-context CPU check](verified_layout_cpu_sample.json)
