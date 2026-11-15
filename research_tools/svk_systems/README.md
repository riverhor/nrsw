# Six methods on the expanded IDSpace test set

This comparison uses the same 384 training sets to fit NRSW, a classifier on whole-image features, the historical shared-core classifier, ResNet-18, DINOv3 and FLiD. Each method sets its own thresholds from 500 genuine calibration images. The original test set contains 16,210 images. The expanded set uses the same fitted models and thresholds on 28,326 images.

Start with the [original-image run](../../docs/raw-reproduction.md#slovak-idspace), which provides the publisher archives, image files and Stable Diffusion and MobileNet features under `runs/paper/`. Obtain the baseline weights using the [CrossVerify download command](../crossverify/README.md#download-the-public-inputs-separately). Run the following from the repository root with the study environment active.

```console
python -m pip install timm==1.0.31 scikit-learn==1.9.0
python -m research_tools.svk_systems stage-added --archives runs/paper/archives/SVK_template.tar.gz runs/paper/archives/SVK_scan.tar.gz --output runs/paper/added/images
```

The extra image list and document-group assignments are in `ADDED_SVK_METADATA.json.gz`. The downloader verifies all selected images against the list.

## Extract features for the added images

Run the 19 batches for each encoder in order. The PowerShell example resumes completed batches and saves an interrupted batch before retrying it.

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
foreach ($encoder in @('mobile', 'sd')) {
    $weights = if ($encoder -eq 'mobile') { 'runs/paper/mobilenet_v3_small-047dcff4.pth' } else { 'runs/paper/sd15' }
    foreach ($number in 0..18) {
        $segmentPath = 'runs/paper/added/measurements/{0}/segment_{1:D2}' -f $encoder, $number
        if (Test-Path -LiteralPath "$segmentPath/COMPLETE.json") { continue }
        if (Test-Path -LiteralPath $segmentPath) {
            Move-Item -LiteralPath $segmentPath -Destination "$segmentPath.incomplete-$(Get-Date -Format yyyyMMddHHmmss)"
        }
        python -m research_tools.svk_systems measure-added-segment --r5-runtime runs/paper/r5 --images runs/paper/added/images --measurements runs/paper/added/measurements --encoder $encoder --weights $weights --segment $number
        if ($LASTEXITCODE -ne 0) { throw "Feature extraction failed at batch $number ($encoder)" }
    }
}
```

## Fit and evaluate

The historical shared-core comparison uses the authors' checkpoint. Save it as `weights/checkpoint_t0_m2203_parent.pt` or omit `--shared-parent` to run the five methods that use public weights. [Checkpoint details](../crossverify/README.md#registry-card-with-15-labeled-forgeries)

```console
python -m research_tools.svk_systems fit-shared-encoders --r5-runtime runs/paper/r5 --original-measurements runs/paper/all/measurements --added-measurements runs/paper/added/measurements --shared-parent weights/checkpoint_t0_m2203_parent.pt --output runs/paper/matched/shared
python -m research_tools.svk_systems fit-resnet18 --r5-runtime runs/paper/r5 --original-images runs/paper/all/images --added-images runs/paper/added/images --weights weights/resnet18-f37072fd.pth --output runs/paper/matched/resnet
python -m research_tools.svk_systems fit-dinov3 --r5-runtime runs/paper/r5 --original-images runs/paper/all/images --added-images runs/paper/added/images --weights weights/model.safetensors --output runs/paper/matched/dino
python -m research_tools.svk_systems fit-flid --r5-runtime runs/paper/r5 --original-images runs/paper/all/images --added-images runs/paper/added/images --weights runs/paper/mobilenet_v3_small-047dcff4.pth --flid-source weights/flid_models.py --output runs/paper/matched/flid
python -m research_tools.svk_systems evaluate --r5-runtime runs/paper/r5 --runs runs/paper/matched/shared runs/paper/matched/resnet runs/paper/matched/dino runs/paper/matched/flid --output runs/paper/matched/RESULTS.json
```

For a shorter fitting check, add `--context f0_d3_o0` to a fitting command. You can also omit `--added-images` or `--added-measurements` for the original test set. Even one FLiD fit extracts the original crop batches to preserve feature calculations. The final evaluation requires all four folds and the expanded test set. Allow at least 25 GiB for the ResNet-18 files and pixel caches, plus time for feature extraction, fitting and scoring.

## Verification

The image stage checked all 12,116 added publisher images. An evaluation from saved predictions reproduced all 36 miss-rate entries from the original and expanded test sets to the precision in the paper, using 5,000 paired resamples per test set. For one split, a fresh fit from the original features matched NRSW scores exactly and the scores of the whole-image classifier within `2.28e-13`. Additional checks covered sampled ResNet-18 inputs, sampled DINOv3 crops and FLiD fitting with synthetic features. End-to-end training and scoring on the expanded image set are still being verified. [Verification details](QUALIFICATION.json)
