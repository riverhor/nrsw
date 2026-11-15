# JPEG recompression experiments

The study applies JPEG quality 85 and 4:2:0 chroma subsampling to the same 16,260 decoded Slovak IDSpace images for every method. The original document splits, field locations and annotations are reused. The decoded source images retain effects of their original encoding. The added JPEG step gives them common output settings.

Follow the [image-based setup](../../docs/raw-reproduction.md#setup) from the repository root. Run the original-image experiment first to download the images and weights and prepare the data needed for the paired comparison. For a single command that runs the JPEG refit, see the [IDSpace instructions](../../docs/raw-reproduction.md#slovak-idspace). The commands below let you run the steps separately. The JPEG settings were verified with Pillow 12.2.0, libjpeg API 8.0 and libjpeg-turbo 3.1.4.1.

## Refit after JPEG recompression

The refit extracts Stable Diffusion and MobileNet features from the recompressed images, fits five detectors and sets new thresholds using 500 genuine calibration images per fit.

```console
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg plan
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg stage --archives runs/paper/archives/SVK_template.tar.gz runs/paper/archives/SVK_scan.tar.gz
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg check-transform --images runs/jpeg/images
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg measure --images runs/jpeg/images --mobile-weights runs/paper/mobilenet_v3_small-047dcff4.pth --sd-snapshot runs/paper/sd15
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg refit
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg intervals --native-run runs/paper/all/refit
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg verify --submission paper.zip
```

The image stage verifies the publisher archives and selected images. You can use `--images runs/paper/all/images` to reuse the verified original image directory and skip `stage`. `check-transform` checks the JPEG transformation on a subset. Rerunning `measure` skips completed batches. Allow roughly 2½ hours for extraction and another 45 minutes for refitting and scoring on the study hardware. Image staging takes about 3.6 GB and features and scores take about 0.4 GB beyond the downloaded inputs.

- `refit/RESULTS.json` gives the AUROC, ROC-interpolated miss rates and observed rates at the calibrated thresholds. Run `refit --qualification` for one fit per fold. That check writes to `qualification/`.
- `PAIRED_RESULTS.json` gives the within-JPEG method differences and pointwise 95% intervals from 5,000 paired resamples of document groups. With `--native-run`, it also compares the spectral-versus-diagonal difference before and after recompression.
- `VERIFICATION.json` compares operating points and intervals with the reported values. A fresh run matched those values.

## Apply the original detectors to recompressed images

To measure sensitivity to encoding, keep the original fitted models and thresholds fixed and score the test images from both encodings. Run this after both feature sets are available.

```console
python -m research_tools.jpeg --r5-runtime runs/paper/r5 --run runs/jpeg freeze --native-measurements runs/paper/all/measurements --submission paper.zip
```

The results under `frozen/` report observed error rates, AUROC, paired changes and the fraction of decisions that crossed the original cutoff. The fresh run reproduced the reported rates, paired intervals and decision changes. Allow about an hour for refitting the original models, scoring both test sets and resampling.
