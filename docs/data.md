# Data formats

Prepare one layout at a time. Use a stable `layout_id` and the same field order and reference geometry throughout extraction, training, and prediction.

## Image manifest

`nrsw prepare` reads UTF-8 JSON Lines, one document image per line. Paths are relative to the manifest's directory. Here are two records for an original and an edit derived from it:

```jsonl
{"id":"card-01","image":"images/card-01.png","group":"source-01","layout_id":"sample-card","label":0}
{"id":"card-01-edited","image":"images/card-01-edited.png","group":"source-01","layout_id":"sample-card","label":1,"candidate_fields":["name"]}
```

`id` identifies an image. `group` identifies its dependency component. Use the same group for all originals and derivatives that share source content, including scans, resizes, and linked donor documents. The split assigns whole groups to one role. It chooses one image per fitting or calibration group and keeps all views from the remaining groups for evaluation.

The prepared manifest adds `role`. You can also assign roles yourself and start at `nrsw extract`.

| Role | Label | Purpose |
| --- | --- | --- |
| `anchor` | `0` | Genuine field references for the evidence fit and feature alignment |
| `bank` | `0` | Genuine strength reference and classifier training |
| `support` | `1` | Forged documents for the evidence fit and classifier training |
| `calibration` | `0` | Independent genuine documents for threshold selection |
| `evaluation` | `0` or `1` | Held-out documents for computing error rates |
| `inference` | `-1` | Unlabeled queries |

Each `support` record needs one or two distinct `candidate_fields` from the layout. Use an empty list or omit the field on all other prepared roles. A record may also include `sha256` to check the encoded image bytes. To supply the two inner folds yourself, set `inner_fold` to `0` or `1` on all anchor and support records. Omit it or use `-1` on other records. The extractor accepts JSON Lines, a JSON array, a JSON object with a `records` array, or CSV. In CSV, encode `candidate_fields` as a JSON array in its cell.

For prediction, create a separate manifest such as:

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

Choose multiple genuine references from fitting groups. A query skips any reference with the same image bytes, so more than one reference helps when extracting the reference images themselves. Registration uses the first reference that passes its geometric checks. The feature packet marks `local_valid` only when all fields are observable. For documents with unavailable local features the classifier uses the global descriptors, the fitted genuine mean for the ordered local block, and zero local strength.

If the document images are already aligned to the same pixel frame, use fixed field coordinates. Set `mode` to `aligned`, provide `image_size` as `[width, height]`, and put `quads` directly in the layout JSON in place of `references`. Every image must have this size after EXIF correction. Use the same mode for fitting and prediction.

## Feature packets

With precomputed descriptors, construct `nrsw.FeatureDataset` and call `nrsw.save_feature_dataset(dataset, path)`. Each `.npz` contains these arrays and a scalar Unicode `metadata_json`. Here `N` is the number of images, `m` the field count, `dg` the global SD dimension, and `d` the MobileNet dimension. The bundled image frontend uses `dg = d = 512`.

| Array | Shape and type | Content |
| --- | --- | --- |
| `global_features` | `[N, dg]`, `float32` | Global SD descriptor |
| `local_features` | `[N, m, d]`, `float32` | Descriptors in layout field order |
| `global_mobile` | `[N, d]`, `float32` | Global MobileNet descriptor |
| `local_valid` | `[N]`, `bool` | True when all fields are available |
| `labels` | `[N]`, integer | `0`, `1`, or `-1` |
| `roles`, `groups`, `ids` | `[N]`, Unicode strings | Role, dependency group, and image ID |
| `candidate_mask` | `[N, m]`, `bool` | Candidate fields on forged support rows |
| `inner_folds` | `[N]`, integer, optional | `0` or `1` on anchor and support rows, `-1` elsewhere |
| `field_available` | `[N, m]`, bool, optional | Individual MobileNet fields with usable registered crops |
| `pixel_features`, `pixel_available` | `[N, m, 7]`, `float64` and `[N, m]`, bool, optional | Native-resolution grayscale measurements for the field-classifier extension |

The `metadata` dict passed to `FeatureDataset` must provide `layout_id`, `field_order`, and an `extractor` object that identifies the representation and preprocessing. Keep the same contract across files for one model. The supplied extractor records its weight hashes and registration settings.

All global descriptors and complete local rows must be finite. Use one document per fitting group. Both inner folds need at least two complete anchors and one complete support document. Every anchor needs complete local fields, and at least two bank documents need complete local fields. To control the two component-disjoint folds, supply `inner_folds` in the packet. The package otherwise assigns them deterministically from the fitting group identifiers.

`nrsw extract --field-classifiers` adds the field measurements. It uses the enclosing native pixel rectangle of each registered field. Crops smaller than 48 pixels in either dimension have no pixel descriptor. A zero descriptor is a valid training measurement and does not contribute a pixel score at prediction time. The document classifier still requires all fields for its joint local representation.

## Model and calibration outputs

Keep the model directory together when copying it. It contains numeric arrays and JSON metadata that identify the fitted method and expected features. Keep `cuts.json` with the model to make calibrated decisions. Use a separate genuine calibration packet or the calibration rows of a combined packet. Evaluation and prediction can likewise use a separate packet or select the corresponding rows from a combined packet.

At the default calibration confidence setting, at least 299 independent genuine calibration documents are needed to obtain a finite threshold for a nominal 1% false-positive budget. If a threshold is unavailable, gather more independent genuine documents or inspect unthresholded scores. Use independent labeled evaluation data to measure both false positives and missed forgeries.
