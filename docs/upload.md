# Detect a document from the demo

The demo can run NRSW on a JPEG or PNG when it is connected to an inference service. It starts with one registered layout, the Slovak ID card layout used by the [IDSpace examples](datasets.md). Use an image of the front of the card with the whole card and its fields visible. The detector expects the known layout and requires successful field registration before reporting scores.

On the [public demo](https://riverhor.github.io/nrsw/#upload), **Try a random sample** opens a saved NRSW prediction for one of the six bundled synthetic IDSpace images. You can use it without a file or access to the inference service.

## Start on your computer

Follow [the training guide](training.md#worked-idspace-image-example) to prepare the Slovak ID layout and train an image detector. To use all three 1%, 5% and 10% false-positive rate targets, pass `--calibration 500` to `examples/tutorial_idspace.py` and `--fpr 0.01 0.05 0.10` to `nrsw fit-images`. Set aside 500 independent unaltered calibration images. Uploads use this fit's scores and thresholds. The built-in examples show the paper's saved predictions.

Create a `layouts.json` beside the output directories. Paths are relative to this file.

```json
{
  "schema": "nrsw.layout_registry.v1",
  "layouts": {
    "IDSpace_SVK_R29": {
      "layout": "data/idspace-tutorial/layout.json",
      "model": "runs/idspace-tutorial/model",
      "calibration": "runs/idspace-tutorial/cuts.json"
    }
  }
}
```

Start the demo with the same visual encoders and settings used for training. The command below matches the CUDA example in the training guide.

```console
nrsw serve --registry layouts.json --device cuda --precision fp16 --sd-snapshot weights/sd15 --mobile-weights weights/mobilenet_v3_small-047dcff4.pth
```

Open [localhost:8000](http://127.0.0.1:8000/), or use [the Simplified Chinese view](http://127.0.0.1:8000/?lang=zh-CN). Select a JPEG or PNG up to 8 MiB and run detection. The result shows the document score, the cutoff, the review decision and the weights and evidence for each registered field. Changing the target false-positive rate updates the decision using the same score.

The upload server processes one image at a time. It reads the image into memory and returns the prediction without writing the uploaded file to disk. Image previews stay in the browser. The input is sent to the inference service only when you run detection.

## Connect an HTTPS service

Run the inference service on a machine with the fitted model, layout reference files and encoder weights. Put it behind an HTTPS reverse proxy that forwards `/api/status` and `/api/predict` to the loopback service. Set upload size, request time and rate limits in the proxy. Disable request body logging and keep any request buffering in memory. Pass `--allow-origin https://riverhor.github.io` for the GitHub Pages demo or supply the exact HTTPS origin of the site serving the demo.

Build a static copy with its inference URL.

```console
nrsw export-demo --output site --upload-api https://inference.example.org
```

For the repository's GitHub Pages workflow, set `NRSW_UPLOAD_API` to that HTTPS base URL as a repository Actions variable before running **Demo on GitHub Pages**. The upload control appears once the browser can reach the service. Without a configured HTTPS endpoint, the public page continues to show the built-in image examples.

For a [Hugging Face ZeroGPU Space](../deploy/space/README.md), use the Space's direct `https://…hf.space` address as `NRSW_UPLOAD_API`. The page uses the Space's Gradio queue and tells users where the image will be processed.
