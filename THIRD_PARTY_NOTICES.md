# Third-party notices

The [MIT License](LICENSE) covers the NRSW code and documentation. Dependencies, pretrained weights and datasets are available from their publishers under their respective licenses.

| Project | Use |
| --- | --- |
| [NumPy](https://github.com/numpy/numpy) | Model fitting and feature files |
| [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision) | Image feature extraction |
| [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), and [Accelerate](https://github.com/huggingface/accelerate) | Stable Diffusion features |
| [Safetensors](https://github.com/huggingface/safetensors) and [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) | Loading and downloading model weights |
| [Pillow](https://github.com/python-pillow/Pillow) and [OpenCV](https://github.com/opencv/opencv) | Image decoding and alignment |
| [timm](https://github.com/huggingface/pytorch-image-models) and [scikit-learn](https://scikit-learn.org/) | DINOv3 comparator |
| [FLiD authors' source](https://github.com/merealone2516/FLiD/tree/e432dbb401da50bfeb176ae2a081ab4e4f613dda) | FLiD comparator |

The [Stable Diffusion 1.5 weights](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14) use the CreativeML Open RAIL-M license. The [DINOv3 weights](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/tree/3bf4720a82ec2066db88137180ff1f83a675cef0) use the DINOv3 license. The MobileNetV3-Small encoder uses torchvision's [`IMAGENET1K_V1` weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html). See the [dataset guide](docs/datasets.md) for publisher terms, citations and downloads.
