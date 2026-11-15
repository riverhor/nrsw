# Matched SVK systems on the original and expanded populations

The original and expanded main-text panels use the R5 original NRSW strength scale, the global-only and historical shared-core heads, ResNet-18, DINOv3 and FLiD. Each detector fits the same 384 ordered TRAIN contexts. Its own CAL500 scores fix the 499th, 484th and 462nd ascending thresholds before TEST scoring. Added documents use the same fitted models and calibration cuts as the original test set.

Use a Git checkout, the authenticated R5 runtime and the R5 Python/CUDA environment described in the [reproduction guide](../../docs/reproduction.md). Download the public visual weights with the [CrossVerify asset command](../crossverify/README.md), and install the exact comparison versions `timm==1.0.31` and `scikit-learn==1.9.0` for DINOv3. `SHARED_CORE` also requires the study's seed-2203 historical checkpoint (585,333 bytes, SHA-256 `8cb8acc92d4fe1a144622ef500bf8a57a72b7ad37cab1283283547697e353947`). The study authors are the source for this checkpoint.

The checked-in `ADDED_SVK_METADATA.json.gz` carries the archived sampling maps and component order for 12,116 added images. The CLI authenticates it by SHA-256. Run these commands from the Git checkout. `work/r5`, `work/all/images` and `work/all/measurements` are the authenticated original runtime, publisher images and sealed native measurements produced by `r62_svk.py`.

```console
python -m research_tools.svk_systems stage-added --archives work/archives/SVK_template.tar.gz work/archives/SVK_scan.tar.gz --output work/added/images
```

Run the following pair for each segment index from `0` to `18`, inclusive. Each segment has a separate completed receipt. The last segment has 596 images. Keep the same exact encoded images, model weights and package.

```console
python -m research_tools.svk_systems measure-added-segment --r5-runtime work/r5 --images work/added/images --measurements work/added/measurements --encoder mobile --weights work/mobilenet_v3_small-047dcff4.pth --segment 0
python -m research_tools.svk_systems measure-added-segment --r5-runtime work/r5 --images work/added/images --measurements work/added/measurements --encoder sd --weights work/sd15 --segment 0
```

PowerShell can run the segments sequentially. The refit verifies every reused segment and its MobileNet parent before consuming features.

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
foreach ($encoder in @('mobile', 'sd')) {
    $weights = if ($encoder -eq 'mobile') { 'work/mobilenet_v3_small-047dcff4.pth' } else { 'work/sd15' }
    foreach ($number in 0..18) {
        $segmentPath = 'work/added/measurements/{0}/segment_{1:D2}' -f $encoder, $number
        if (Test-Path -LiteralPath "$segmentPath/COMPLETE.json") { continue }
        if (Test-Path -LiteralPath $segmentPath) {
            Move-Item -LiteralPath $segmentPath -Destination "$segmentPath.incomplete-$(Get-Date -Format yyyyMMddHHmmss)"
        }
        python -m research_tools.svk_systems measure-added-segment --r5-runtime work/r5 --images work/added/images --measurements work/added/measurements --encoder $encoder --weights $weights --segment $number
    }
}
```

Refit the heads using the original native descriptors and regenerate both panels' score packets:

```console
python -m research_tools.svk_systems fit-shared-encoders --r5-runtime work/r5 --original-measurements work/all/measurements --added-measurements work/added/measurements --shared-parent weights/checkpoint_t0_m2203_parent.pt --output work/matched/shared
python -m research_tools.svk_systems fit-resnet18 --r5-runtime work/r5 --original-images work/all/images --added-images work/added/images --weights weights/resnet18-f37072fd.pth --output work/matched/resnet
python -m research_tools.svk_systems fit-dinov3 --r5-runtime work/r5 --original-images work/all/images --added-images work/added/images --weights weights/model.safetensors --output work/matched/dino
python -m research_tools.svk_systems fit-flid --r5-runtime work/r5 --original-images work/all/images --added-images work/added/images --weights work/mobilenet_v3_small-047dcff4.pth --flid-source weights/flid_models.py --output work/matched/flid
python -m research_tools.svk_systems evaluate --r5-runtime work/r5 --runs work/matched/shared work/matched/resnet work/matched/dino work/matched/flid --output work/matched/RESULTS.json
```

Omit `--shared-parent` to run the five systems with public weights. The `fit-*` commands accept `--context f0_d3_o0` for a one-context run and allow omission of added inputs for original-only scoring. Even a one-context FLiD run extracts the complete original raw crop roster to preserve physical batch composition. The full evaluator requires all four 96-context folds and the complete added cohort. It preserves the 5,000 fixed paired whole-component resamples stratified by cohort and fold, and shows agreement at the main table's two decimal places. Raw refits use image- or descriptor-stage inputs, while the separate R62 archived score replays remain available to verify the publication.

Plan for sequential GPU jobs and at least 25 GiB for ResNet-18 refits and pixel caches. On the study's 16 GB RTX 5070 Ti, reported primary SD extraction throughput implies roughly 80 minutes for the 19 added SD segments. The archived FLiD fit and CAL took 36 minutes before TEST, excluding descriptor extraction. The separate ResNet-18 and DINOv3 protocol estimates were 40–55 minutes and 15–40 minutes, respectively. These estimates exclude staging and can vary with hardware.

## Verification scope

The added image stage verified all 12,116 encoded publisher images. The portable role reader and paired draw generator reproduced the 28,326 archived TEST identities in order and all 5,000 archived multiplicity vectors. The original NRSW one-context refit matched the published score packets exactly from the 4,896-image R5 measurement store. The global-only branch differed by at most `2.28e-13` in the same check. The historical shared-core branch uses the archived phase-level TRAIN and CAL projection shapes for the full run. A one-context smoke test uses smaller matrices and had maximum absolute score differences of `1.08e-5` for CAL and `3.07e-6` for TEST.

The ResNet-18 image preprocessing reproduced the archived byte matrices for five sampled original and added rows. DINOv3's crop geometry and area-weight checks passed for sampled native and simulated-scan images. The FLiD qualification covered archived role assignment, physical crop batch construction and the three fitting passes using synthetic descriptor inputs. Full added neural feature extraction and the six-system GPU refit still require execution and comparison against the archived results. The added raw feature stage uses the exact published R5 visual operators and saved maps. Its byte parity with the previously cached added descriptors remains to be measured.

The grouped evaluator was separately qualified using saved scores for all six systems. It reproduced all 36 original and combined FNR table cells to the published precision and computed 5,000 paired replicates per population. The [qualification record](QUALIFICATION.json) records the execution scopes.
