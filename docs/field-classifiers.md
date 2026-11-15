# Scoring individual fields

Start with the standard detector. If edits tend to occur in individual fields, such as an address or a document number, you can also fit classifiers for those fields. Use the same validation documents to compare both versions at the false-positive rate your application can accept.

Add `--field-classifiers` to the [image training command](training.md). For each field, the model can use its MobileNet descriptor and features from the pixel crop. Fitting a field classifier requires at least two genuine examples and two forgeries tagged for that field with usable measurements. There may be more than one candidate field per forgery.

Each score is put on a common scale using genuine training documents. The detector takes the highest available score from the document classifier and the field classifiers. It then compares that score with a threshold estimated on a separate set of genuine documents. Use the model and threshold from the same run. At prediction time, supply the image and layout as usual.

See the [class diagram](architecture.md#main-classes) for how the field classifier model uses the document classifier.

## Checking the implementation

We compared the package's scores with the research implementation on four saved sets of features. The table reports the largest absolute difference within each set and the false-positive rate targets at which all threshold decisions matched.

| Layout and training sample | Genuine calibration images | Test images | Largest calibration score difference | Largest test score difference | Matching decision targets |
| --- | ---: | ---: | ---: | ---: | --- |
| CrossVerify registry card, 15 labeled forgeries | 75 | 284 | 0 | 0 | 5%, 10% |
| CrossVerify registry card, 120 labeled forgeries | 75 | 284 | 0 | 0 | 5%, 10% |
| IDSpace Slovak ID, first test fold | 500 | 7,126 | 1.39e-13 | 6.82e-12 | 1%, 5%, 10% |
| IDSpace Slovak ID, second test fold | 500 | 7,206 | 2.71e-13 | 2.18e-11 | 1%, 5%, 10% |

The pixel measurements also matched across all 1,077 checked CrossVerify crops. The second IDSpace fold includes one image with unavailable fields. For the saved predictions and study instructions, see the [reproduction guide](reproduction.md).

From a source checkout with the test dependencies installed, run `python -m pytest tests/test_field_classifiers.py -q`. Install the vision dependencies to include the crop measurement test.
