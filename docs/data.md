# Data formats

Prepare one layout at a time. Use a stable `layout_id` and the same field order and reference geometry throughout extraction, training, and prediction.

## Image manifest

`nrsw prepare` reads UTF-8 JSON Lines, one document image per line. Relative paths are resolved from the manifest's directory. In this example, the original and the edit share a group.

```jsonl
{"id":"card-01","image":"images/card-01.png","group":"source-01","layout_id":"sample-card","label":0}
{"id":"card-01-edited","image":"images/card-01-edited.png","group":"source-01","layout_id":"sample-card","label":1,"candidate_fields":["name"]}
```

`id` identifies an image. Use the same `group` for all images linked to a source document, including its scans, edits and resized copies. If an edit borrows content from another document, put both sources and their derivatives in the same group. `prepare` selects one image from each group assigned to training or calibration and keeps every image from the remaining groups for evaluation.

The prepared manifest adds `role`. You can also assign roles yourself and start at `nrsw extract`.

| Role | Label | Purpose |
| --- | --- | --- |
| `anchor` | `0` | Learn genuine field statistics and align features |
| `bank` | `0` | Set the scale of genuine evidence scores and train the classifier |
| `support` | `1` | Learn features of edited fields and train the classifier |
| `calibration` | `0` | Independent genuine documents for threshold selection |
| `evaluation` | `0` or `1` | Held-out documents for computing error rates |
| `inference` | `-1` | Unlabeled queries |

Each `support` record needs one or two distinct `candidate_fields` from the layout. For other roles, leave the list empty or omit it. An optional `sha256` checks the image file. To specify the two folds used during training, set `inner_fold` to `0` or `1` on every `anchor` and `support` record. For other roles, omit it or use `-1`.

`nrsw extract` accepts JSON Lines, a JSON array, a JSON object with a `records` array, or CSV. In CSV, write `candidate_fields` as a JSON array in its cell. For predictions from a manifest, use `inference` rows.

```jsonl
{"id":"query-01","image":"queries/query-01.png","group":"query-source-01","layout_id":"sample-card","label":-1,"role":"inference"}
```

## Layout and genuine references

Define at least two fields. On each genuine reference, specify a quadrilateral for every field. The numbers below illustrate the coordinate format. Measure the corners on your own image.

```json
{
  "layout_id": "sample-card",
  "field_order": ["name", "document_number"],
  "references": [
    {
      "image": "images/reference.png",
      "quads": {
        "name": [[40, 80], [340, 80], [340, 125], [40, 125]],
        "document_number": [[40, 170], [270, 170], [270, 215], [40, 215]]
      }
    }
  ]
}
```

Paths in this file are relative to the layout JSON. Mark coordinates on the EXIF-corrected image, in pixel-edge coordinates, clockwise from the top-left corner. For an image of width `w` and height `h`, the bottom-right edge is `(w, h)`. Use convex quadrilaterals inside the image.

Choose at least two reference images from genuine training groups. When processing an image, the extractor skips references with identical file contents and uses the first remaining reference that passes its alignment checks. `local_valid` is true when all fields can be measured. If a field is missing, the document classifier uses the whole-image features, the fitted genuine mean for the ordered field features, and zero local evidence strength.

If the document images are already aligned to the same pixel frame, use fixed field coordinates. Set `mode` to `aligned`, provide `image_size` as `[width, height]`, and put `quads` directly in the layout JSON in place of `references`. Every image must have this size after EXIF correction. Use the same mode for fitting and prediction.

## Feature files

To use precomputed features, construct `nrsw.FeatureDataset` and call `nrsw.save_feature_dataset(dataset, path)`. Each `.npz` contains the arrays below and a Unicode scalar named `metadata_json`. `N` is the number of images, `m` the number of fields, `dg` the Stable Diffusion feature dimension, and `d` the MobileNet feature dimension. The supplied encoders use `dg = d = 512`.

| Array | Shape and type | Content |
| --- | --- | --- |
| `global_features` | `[N, dg]`, `float32` | Whole-image Stable Diffusion features |
| `local_features` | `[N, m, d]`, `float32` | MobileNet features in layout field order |
| `global_mobile` | `[N, d]`, `float32` | Whole-image MobileNet features |
| `local_valid` | `[N]`, `bool` | True when all fields are available |
| `labels` | `[N]`, integer | `0`, `1`, or `-1` |
| `roles`, `groups`, `ids` | `[N]`, Unicode strings | Role, source group, and image ID |
| `candidate_mask` | `[N, m]`, `bool` | Candidate fields on forged support rows |
| `inner_folds` | `[N]`, integer, optional | `0` or `1` on anchor and support rows, `-1` elsewhere |
| `field_available` | `[N, m]`, `bool`, optional | Fields with crops usable by MobileNet |
| `pixel_features`, `pixel_available` | `[N, m, 7]`, `float64` and `[N, m]`, `bool`, optional | Grayscale measurements at original image resolution and their availability |

The `metadata` dictionary passed to `FeatureDataset` requires `layout_id`, `field_order`, and an `extractor` object that identifies how features were computed. Use matching metadata for all files scored by one model. The supplied extractor records checksums for the encoder weights and the image alignment settings.

Whole-image features and all complete field features must be finite. Use one training document per group. Every anchor and at least two bank documents need all fields. Each training fold needs at least two anchors and one support document with all fields. Set `inner_folds` to assign the folds yourself. When omitted, the package derives a fixed split from the training group IDs.

`nrsw extract --field-classifiers` also measures the pixel rectangle enclosing each registered field at the original resolution. A field needs a crop at least 48 pixels wide and high. An all-zero pixel feature is valid for training and skipped when scoring that field. The document classifier uses its joint field features only when all fields are available.

## Model and calibration outputs

Keep the model directory and `cuts.json` together. The model records the fitted parameters and expected feature settings. `cuts.json` stores thresholds estimated from independent genuine images. Calibration, evaluation and prediction each select the appropriate rows from a combined feature file and also accept separate files.

At the default calibration setting (`--delta 0.05`), a threshold targeting a 1% false-positive rate needs at least 299 independent genuine calibration documents. If calibration cannot find a threshold, collect more genuine calibration documents or inspect the scores directly. Measure false positives and missed forgeries on separate labeled test images.
