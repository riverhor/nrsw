# Serving multiple layouts

Fit and calibrate one detector per layout. Copy the layout geometry, model directory and calibration file to the computer that runs inference. Preserve the encoder weights, device, precision and batch settings used when extracting training features.

Create a registry JSON file:

```json
{
  "schema": "nrsw.layout_registry.v1",
  "layouts": {
    "card-v1": {"layout": "card/layout.json", "model": "card/model", "calibration": "card/cuts.json"},
    "pass-v2": {"layout": "pass/layout.json", "model": "pass/model", "calibration": "pass/cuts.json"}
  }
}
```

Paths are relative to the registry file. Each key must match the layout ID in both the geometry and model. A registry can serve the original model and the field-classifier variant on different layouts.

```console
nrsw route --registry layouts.json --layout-id card-v1 --image query.png --fpr 0.05 --device cuda
```

The example assumes that the pinned encoder weights are already cached. Supply the same local weight paths on each call if you keep them elsewhere. To avoid loading the encoders for every request, keep a Python registry open:

```python
from nrsw import LayoutRegistry

with LayoutRegistry.from_file("layouts.json", device="cuda") as detector:
    card = detector.predict_image("card-v1", "card-query.png", fpr="0.05")
    passport = detector.predict_image("pass-v2", "pass-query.png", fpr="0.05")
```

The application supplies the layout ID. If the application already has a detector for historical layouts, call `detector.supports(layout_id)` and use the existing route for other IDs. A direct call with an unregistered ID raises `UnknownLayoutError`. To add a layout, fit and calibrate it, add one entry, then load the revised registry.

## Inspect predictions in a browser

Supply an extracted query or evaluation packet to the local demonstration page:

```console
nrsw serve --model runs/card/model --features runs/queries.npz --calibration runs/card/cuts.json
```

For a static version of the generated example, run `nrsw export-demo --output site` and open `site/index.html`. To publish it on GitHub Pages, select **GitHub Actions** as the source under **Settings → Pages** and run **Demo on GitHub Pages** on the default branch.
