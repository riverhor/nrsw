# Serving multiple layouts

Fit and calibrate one detector per layout. Copy the geometry, model directory and calibration file to the computer that will score new images. Use the encoder weights, device, precision and batch settings used to extract the training features. The [architecture diagrams](architecture.md) show what the layouts share.

List the files for each layout in a registry such as `layouts.json`.

```json
{
  "schema": "nrsw.layout_registry.v1",
  "layouts": {
    "card-v1": {"layout": "card/layout.json", "model": "card/model", "calibration": "card/cuts.json"},
    "pass-v2": {"layout": "pass/layout.json", "model": "pass/model", "calibration": "pass/cuts.json"}
  }
}
```

Paths are relative to the registry file. Each key must match the layout ID in both the geometry and model. A registry can serve NRSW and NRSW with field classifiers on different layouts.

```console
nrsw route --registry layouts.json --layout-id card-v1 --image query.png --fpr 0.05 --device cuda
```

The command uses encoder weights already in the local cache. If you downloaded them to a custom location, pass `--sd-snapshot` and `--mobile-weights`. In Python, keep the registry open to reuse the encoders across requests.

```python
from nrsw import LayoutRegistry

with LayoutRegistry.from_file("layouts.json", device="cuda") as detector:
    card = detector.predict_image("card-v1", "card-query.png", fpr="0.05")
    passport = detector.predict_image("pass-v2", "pass-query.png", fpr="0.05")
```

The application supplies the layout ID. If it already has detectors for other layouts, use `detector.supports(layout_id)` to decide when to call NRSW. `predict_image` raises `UnknownLayoutError` for an unknown ID. To add a layout, fit and calibrate a detector, add its entry, and reload the registry.

## Inspect predictions in a browser

Open the local demo with a feature file from your query or evaluation images.

```console
nrsw serve --model runs/card/model --features runs/queries.npz --calibration runs/card/cuts.json
```

For a static demo with generated examples, run `nrsw export-demo --output site` and open `site/index.html`. To publish it on GitHub Pages, select **GitHub Actions** as the source under **Settings → Pages** and run the **Demo on GitHub Pages** workflow on the default branch.
