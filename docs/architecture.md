# Architecture

NRSW scores documents from a known layout. The same Stable Diffusion and MobileNet encoders can serve several layouts. Each layout has its own field positions, fitted detector, and thresholds. The caller supplies the layout ID when requesting a prediction.

## How an image is scored

```mermaid
flowchart TB
    Image["Document image"] --> SD["Stable Diffusion<br/>Whole-image features"]
    Image --> Mobile["MobileNet<br/>Whole-image and field features"]
    Image --> Align["Find and crop fields"]
    Layout["Layout geometry"] --> Align
    Align --> Mobile
    Mobile --> Evidence["Compare fields with genuine references<br/>Estimate field weights and peak evidence"]
    SD --> Score["Document classifier"]
    Mobile --> Score
    Evidence --> Score
    Score --> Decision["Compare the score with the threshold"]
    Threshold["Threshold calibrated on genuine documents"] --> Decision
    Decision --> Result["Score and review flag"]
```

The detector measures how unusual each field looks relative to the corresponding field in genuine documents. The document classifier uses the individual field features, their weighted summary, the peak evidence, and whole-image features. Whole-image features remain available when fields cannot be located. Above the threshold, the detector flags the document for review.

## From training images to predictions

```mermaid
flowchart TB
    Images["Labeled document images"] --> Prepare["Assign groups to distinct roles<br/>nrsw prepare"]
    Geometry["Field geometry for one layout"] --> Extract["Extract features"]
    Prepare --> Extract
    subgraph FitImages["nrsw fit-images"]
        Extract --> Train["Fit a detector on the training groups"]
        Extract --> Cal["Score independent genuine calibration groups"]
        Train --> Cal
        Train --> Model["Saved model"]
        Cal --> Cuts["Saved thresholds"]
        Extract --> Eval["Evaluate held-out groups when provided"]
        Model --> Eval
        Cuts --> Eval
    end
    Model --> Predict["Score a new document<br/>nrsw predict-image"]
    Cuts --> Predict
    Query["New image and layout geometry"] --> Predict
    Predict --> Output["Score and review flag"]
```

Tag one or two candidate fields on each training forgery. To score a new image, pass its layout and use the saved model and thresholds. See the [training guide](training.md) to fit your own layout and the [layout registry guide](layouts.md) to serve several layouts in one process.

## Main classes

```mermaid
classDiagram
    direction TB
    class LayoutRegistry {
        +from_file(path)
        +supports(layout_id)
        +predict_image(layout_id, image_path)
    }
    class ImageEncoders {
        mobile
        sd
    }
    class _Route {
        registrar
        model
        calibration
    }
    class JointModel {
        +score(batch)
    }
    class FieldClassifierModel {
        +score(batch, pixel_features, pixel_available)
    }
    class Calibration {
        +cut(fpr)
    }
    class FeatureDataset {
        global_features
        local_features
        roles
        groups
    }
    LayoutRegistry *-- "0..1" ImageEncoders : shared across layouts
    LayoutRegistry *-- "1..*" _Route : one per layout
    LayoutRegistry ..> FeatureDataset : scores extracted features
    _Route --> Calibration : fitted thresholds
    _Route --> JointModel : default detector
    _Route --> FieldClassifierModel : optional detector
    FieldClassifierModel *-- JointModel : document classifier
```

The optional field-classifier model contains a refitted document branch and separate field branches. Genuine training rows define their score scales; independent genuine calibration rows define the threshold of the combined maximum score. See [field-classifier details](field-classifiers.md).

The public Python API accepts a `FeatureDataset` in `fit`, `calibrate`, and `predict`. `LayoutRegistry` adds image extraction and layout selection for applications that call NRSW directly. A registry loads its encoders when first needed and shares them across its layout entries.
