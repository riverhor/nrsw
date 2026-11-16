---
title: NRSW document detection
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12.12"
app_file: app.py
license: mit
---

# NRSW image detection

This Space runs the Slovak identity-card layout for the [NRSW demo](https://riverhor.github.io/nrsw/). It returns the forgery score, calibrated threshold, review decision, and evidence and weight for each registered field. Use the front side of the card in JPEG or PNG format, up to 8 MiB.

The layout reference images come from [IDSpace](detector/ATTRIBUTION.md).

The Space receives the image to measure it. The app accepts the bytes in the queue request, processes them in memory, and returns the measurements. It does not save the uploaded image or return its contents. The demo keeps the image preview in the browser. Availability and wait time depend on the [ZeroGPU quota and queue](https://huggingface.co/docs/hub/spaces-zerogpu).

## Deploy an instance

Use a Gradio Space with ZeroGPU hardware. From the NRSW checkout, prepare the upload directory with the fitted, audited detector.

```console
python deploy/space/stage.py --detector path/to/audited-detector --output path/to/space-upload
```

Upload the contents of the resulting directory to the Space root. The detector's registry must specify `IDSpace_SVK_R29` with the model, layout reference images and calibration thresholds. See the [training guide](https://github.com/riverhor/nrsw/blob/main/docs/upload.md). Encoder weights are downloaded from their publishers and checked before the service starts. The Space checks the PyTorch release, source revision and CUDA version before accepting the platform's distribution label for the fitted extractor.

Submit a documented [Gradio queue call](https://huggingface.co/docs/hub/spaces-api-endpoints) via `POST /gradio_api/call/status` with `{"data":[]}` or via `POST /gradio_api/call/predict` with `{"data":["BASE64_IMAGE","image/jpeg",0.01]}`. Read the corresponding `GET` event stream using the `event_id` returned by the submission. Both endpoints produce one JSON output in the `complete` event. Image input uses plain base64 without a data-URL prefix.
