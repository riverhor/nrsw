"""Extract Stable Diffusion and MobileNet features from document images."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path

import numpy as np

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
MOBILE_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
SD_LAYERS = ("down_blocks.2", "mid_block", "up_blocks.1")
# Checksums for the Stable Diffusion files used by this implementation.
SD_ASSET_SHA256 = {
    "vae/diffusion_pytorch_model.fp16.safetensors": "4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c",
    "unet/diffusion_pytorch_model.fp16.safetensors": "c83908253f9a64d08c25fc90874c9c8aef9a329ce1ca5fb909d73b0c83d1ea21",
    "text_encoder/model.fp16.safetensors": "77795e2023adcf39bc29a884661950380bd093cf0750a966d473d1718dc9ef4e",
    "vae/diffusion_pytorch_model.safetensors": "a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c",
    "unet/diffusion_pytorch_model.safetensors": "19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1",
    "text_encoder/model.safetensors": "d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "tokenizer/tokenizer_config.json": "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "scheduler/scheduler_config.json": "699cce92eb7c122e2eb7dfdea78e6187fda76a5ed4a8e42319b85610e620e091",
    "vae/config.json": "786a7d21647ddea6a04b9675c03d3cb45e90a2f3c6da5fbda2c54ade040036de",
    "unet/config.json": "78f474de6bab3d893868f37be97b636ae65c0df3073ed3256ca458ff599b5f96",
    "text_encoder/config.json": "845df614cb9327ae7bbea027316246fae917827407da6df13572e41b5f93b4cc",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_runtime(device="auto"):
    # Must precede CUDA initialization for deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Image extraction needs PyTorch and model libraries; install nrsw-document-forgery[vision].") from exc
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Supported extraction devices are cpu, cuda, or cuda:N")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; install a compatible CUDA PyTorch build or use --device cpu.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, target


def runtime_versions(*, allow_zerogpu_torch=False):
    result = {}
    for package in ("numpy", "Pillow", "torch", "torchvision", "diffusers", "transformers", "opencv-python-headless"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    if allow_zerogpu_torch and result["torch"] == "2.13.0":
        import torch

        # ZeroGPU pins the public distribution version for this CUDA build.
        if (torch.__version__ not in ("2.13.0", "2.13.0+cu130")
                or torch.version.git_version != "cf30153c4c131c8164ee7798e5022d810682e2cb"
                or torch.version.cuda != "13.0"):
            raise ValueError("ZeroGPU requires the verified PyTorch 2.13.0 CUDA 13.0 build")
        result["torch"] = "2.13.0+cu130"
    return result


def mobile_hash_mapping():
    generator = np.random.Generator(np.random.PCG64(20260908))
    return (generator.integers(0, 512, size=576, dtype=np.int64),
            generator.choice(np.array([-1., 1.], dtype=np.float32), size=576))


def hash_mobile(raw):
    """Literal native FP32 signed accumulation and normalization arithmetic."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 576 or not np.isfinite(raw).all():
        raise ValueError("MobileNet raw features must be finite N x 576")
    bucket, sign = mobile_hash_mapping()
    projected = np.zeros((len(raw), 512), dtype=np.float32)
    for j in range(576):
        projected[:, bucket[j]] += raw[:, j] * sign[j]
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return (projected * (np.sqrt(2) / np.maximum(norms, 1e-12))).astype(np.float32)


def sd_hash_mapping():
    """Native namespaced SplitMix64 signed hash, distinct from Mobile PCG64."""
    payload = b"iseb-independent-family-v1|20260723|sd15-response-control-512"
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)
    mask = (1 << 64) - 1
    seed_mix = (seed * 0x9E3779B97F4A7C15) & mask
    buckets, signs = [], []
    for coordinate in range(3844):
        value = (coordinate + seed_mix + 0xD1B54A32D192ED03) & mask
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & mask
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        buckets.append(value % 512)
        signs.append(-1. if value >> 63 == 0 else 1.)
    return np.asarray(buckets, np.int64), np.asarray(signs, np.float32)


def hash_sd(raw):
    """Native CPU torch scatter_add + normalize(float32 epsilon) + sqrt(2)."""
    import torch
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3844 or not np.isfinite(raw).all():
        raise ValueError("SD raw responses must be finite N x 3844")
    bucket, sign = sd_hash_mapping()
    features = torch.from_numpy(np.array(raw, copy=True))
    projected = features.new_zeros((len(raw), 512))
    projected.scatter_add_(1, torch.from_numpy(bucket)[None].expand(len(raw), -1),
                           features * torch.from_numpy(sign)[None])
    normalized = torch.nn.functional.normalize(projected, p=2., dim=1, eps=torch.finfo(features.dtype).eps)
    return (normalized * math.sqrt(2.)).numpy()


def canonical_sd_image(image):
    from PIL import Image, ImageOps
    source = ImageOps.exif_transpose(image).convert("RGB")
    width, height = source.size
    short, long = min(width, height), max(width, height)
    new_long = int(512 * long / short)
    size = (512, new_long) if width <= height else (new_long, 512)
    if source.size != size:
        source = source.resize(size, Image.Resampling.BICUBIC)
    left = int(round((source.width - 512) / 2.))
    top = int(round((source.height - 512) / 2.))
    return source.crop((left, top, left + 512, top + 512))


def global_mobile_tiles(image):
    """Six row-major integer-edge tiles, without the unused whole-image view."""
    from PIL import Image
    width, height = image.size
    if width < 3 or height < 2:
        raise ValueError("Global MobileNet carrier requires at least a 3x2 image")
    return [np.asarray(image.crop((width*c//3, height*r//2, width*(c+1)//3, height*(r+1)//2))
                       .resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
            for r in range(2) for c in range(3)]


def _mobile_weights(path, allow_download, torch):
    from torchvision.models import MobileNet_V3_Small_Weights
    if path is None:
        path = os.environ.get("NRSW_MOBILE_WEIGHTS")
    if path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
    path = Path(path).expanduser()
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(f"MobileNet weights not found: {path}. Set --mobile-weights or NRSW_MOBILE_WEIGHTS; --allow-download explicitly permits the official 10 MB download.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(MobileNet_V3_Small_Weights.IMAGENET1K_V1.url, str(path), hash_prefix=MOBILE_SHA256)
    digest = sha256_file(path)
    if digest != MOBILE_SHA256:
        raise ValueError(f"Wrong MobileNet checkpoint SHA256: {digest}; expected {MOBILE_SHA256}")
    return path


class MobileEncoder:
    def __init__(self, weights=None, *, device="auto", batch_size=32, allow_download=False):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("mobile_batch_size must be a positive integer")
        self.torch, self.device = _torch_runtime(device)
        try:
            from torchvision.models import mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError("MobileNet extraction requires torchvision; install nrsw-document-forgery[vision].") from exc
        self.weights_path = _mobile_weights(weights, allow_download, self.torch)
        self.model = mobilenet_v3_small(weights=None)
        self.model.load_state_dict(self.torch.load(self.weights_path, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.batch_size = batch_size
        self.mean = self.torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = self.torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def encode_raw(self, crops):
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = np.stack(crops[start:start+self.batch_size])
                if batch.shape[1:] != (224, 224, 3) or batch.dtype != np.uint8:
                    raise ValueError("Mobile input crops must be RGB uint8 224x224")
                tensor = self.torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device, dtype=self.torch.float32) / 255.
                response = self.model.avgpool(self.model.features((tensor-self.mean)/self.std)).flatten(1)
                outputs.append(response.cpu().numpy())
        result = np.concatenate(outputs) if outputs else np.empty((0, 576), np.float32)
        if result.shape != (len(crops), 576) or not np.isfinite(result).all():
            raise RuntimeError("MobileNet returned invalid features")
        return result

    def encode(self, crops):
        return hash_mobile(self.encode_raw(crops))

    def global_descriptor(self, image):
        # Native pooling happens in torch before transfer to NumPy.
        raw = self.encode_raw(global_mobile_tiles(image))
        mean = self.torch.from_numpy(raw).to(self.device).mean(dim=0, keepdim=True).cpu().numpy()
        return hash_mobile(mean)[0]

    def contract(self):
        bucket, sign = mobile_hash_mapping()
        return dict(model="torchvision.mobilenet_v3_small.IMAGENET1K_V1", checkpoint_sha256=MOBILE_SHA256,
                    raw_dimension=576, dimension=512, precision="float32", hash_algorithm="PCG64_signed_sum_FP32",
                    hash_seed=20260908, hash_sha256=hashlib.sha256(bucket.tobytes()+sign.tobytes()).hexdigest(),
                    occupied_buckets=int(len(np.unique(bucket))), norm="sqrt(2)", norm_floor=1e-12,
                    input_size=[224, 224], global_grid=[2, 3], global_tile_pool="mean_raw576_before_hash",
                    normalization_mean=[.485, .456, .406], normalization_std=[.229, .224, .225],
                    device=str(self.device), batch_size=self.batch_size, padded_native_batch224=False)


def resolve_sd_snapshot(path=None, *, allow_download=False, precision="fp16"):
    if path is None:
        path = os.environ.get("NRSW_SD_SNAPSHOT")
    if path is not None:
        snapshot = Path(path).expanduser()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Stable Diffusion model directory not found: {snapshot}")
        # The required files are checked when the encoder loads.
        return snapshot
    try:
        from huggingface_hub import snapshot_download
        variant = ".fp16" if precision == "fp16" else ""
        return Path(snapshot_download(SD_MODEL_ID, revision=SD_REVISION,
                    local_files_only=not allow_download,
                    allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json",
                                    f"vae/diffusion_pytorch_model{variant}.safetensors",
                                    f"unet/diffusion_pytorch_model{variant}.safetensors",
                                    f"text_encoder/model{variant}.safetensors"]))
    except Exception as exc:
        raise FileNotFoundError(f"Stable Diffusion 1.5 weights are unavailable. Supply a local model directory with --sd-snapshot, set NRSW_SD_SNAPSHOT, or use --allow-download.") from exc


class SDEncoder:
    def __init__(self, snapshot=None, *, device="auto", precision="auto", batch_size=1, allow_download=False):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.torch, self.device = _torch_runtime(device)
        if precision == "auto":
            precision = "fp16" if self.device.type == "cuda" else "fp32"
        if precision not in ("fp16", "fp32") or (self.device.type == "cpu" and precision != "fp32"):
            raise ValueError("SD precision must be fp16/fp32; CPU requires fp32")
        self.precision, self.batch_size = precision, batch_size
        self.dtype = self.torch.float16 if precision == "fp16" else self.torch.float32
        self.snapshot = resolve_sd_snapshot(snapshot, allow_download=allow_download, precision=precision)
        try:
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError("SD extraction requires diffusers, transformers and safetensors; install nrsw-document-forgery[vision].") from exc
        self.assets = {}
        variant = ".fp16" if precision == "fp16" else ""
        for component, basename in (("vae", "diffusion_pytorch_model"), ("unet", "diffusion_pytorch_model"), ("text_encoder", "model")):
            file = self.snapshot/component/f"{basename}{variant}.safetensors"
            if not file.is_file():
                raise FileNotFoundError(f"Missing the {precision} Stable Diffusion 1.5 weights at {file}.")
            self.assets[file.relative_to(self.snapshot).as_posix()] = sha256_file(file)
        # Check the tokenizer, scheduler, and model configuration files as well.
        for component in ("tokenizer", "scheduler", "vae", "unet", "text_encoder"):
            for file in sorted((self.snapshot/component).glob("*")):
                if file.is_file() and file.suffix in (".json", ".txt"):
                    self.assets[file.relative_to(self.snapshot).as_posix()] = sha256_file(file)
        expected_config = {key for key in SD_ASSET_SHA256 if not key.endswith(".safetensors")}
        if not expected_config <= self.assets.keys():
            raise FileNotFoundError(f"Missing Stable Diffusion files: {sorted(expected_config-self.assets.keys())}")
        for name, digest in self.assets.items():
            if SD_ASSET_SHA256.get(name) != digest:
                raise ValueError(f"The Stable Diffusion file {name} does not match revision {SD_REVISION}. Download that revision and try again.")
        kwargs = dict(local_files_only=True, torch_dtype=self.dtype, use_safetensors=True)
        if precision == "fp16":
            kwargs["variant"] = "fp16"
        self.vae = AutoencoderKL.from_pretrained(str(self.snapshot), subfolder="vae", **kwargs)
        self.text_encoder = CLIPTextModel.from_pretrained(str(self.snapshot), subfolder="text_encoder", **kwargs)
        self.unet = UNet2DConditionModel.from_pretrained(str(self.snapshot), subfolder="unet", **kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained(str(self.snapshot), subfolder="tokenizer", local_files_only=True)
        self.scheduler = DDIMScheduler.from_pretrained(str(self.snapshot), subfolder="scheduler", local_files_only=True)
        for model in (self.vae, self.text_encoder, self.unet):
            model.to(self.device).eval().requires_grad_(False)
        self.responses, self.hooks = {}, []
        modules = dict(self.unet.named_modules())
        for name in SD_LAYERS:
            if name not in modules:
                raise ValueError(f"The Stable Diffusion U-Net is missing the required layer {name}")
            def capture(module, inputs, output, key=name):
                tensor = output[0] if isinstance(output, tuple) else output
                self.responses[key] = tensor.float().mean(dim=(2, 3))
            self.hooks.append(modules[name].register_forward_hook(capture))

    def encode_raw(self, images):
        torch = self.torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start+self.batch_size]
                # Keep the original singleton-batch strides as well as arithmetic.
                # FP16 convolution kernels can differ after an otherwise equivalent
                # np.stack/permute staging operation.
                staged = [torch.from_numpy(np.array(canonical_sd_image(im), dtype=np.uint8, copy=True))
                          .permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div(127.5).sub(1.)
                          for im in batch]
                tensor = (staged[0] if len(staged) == 1 else torch.cat(staged)).to(self.device, dtype=self.dtype)
                latents = self.vae.encode(tensor).latent_dist.mode() * self.vae.config.scaling_factor
                tokens = self.tokenizer([""] * len(batch), padding="max_length", max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
                hidden = self.text_encoder(tokens.input_ids.to(self.device))[0].to(dtype=self.dtype)
                # Reseed each image, preserving batch-one noise semantics across chunks.
                generator = torch.Generator(device=self.device).manual_seed(0)
                one_noise = torch.randn((1, *latents.shape[1:]), generator=generator, device=self.device, dtype=latents.dtype)
                noise = one_noise.expand(len(batch), -1, -1, -1).contiguous()
                timestep = torch.full((len(batch),), 100, device=self.device, dtype=torch.long)
                noisy = self.scheduler.add_noise(latents, noise, timestep)
                self.responses.clear()
                predicted = self.unet(sample=noisy, timestep=timestep, encoder_hidden_states=hidden).sample
                predicted_flat, noise_flat = predicted.float().flatten(1), noise.float().flatten(1)
                stats = torch.cat(((predicted_flat-noise_flat).pow(2).mean(1, keepdim=True),
                                   torch.nn.functional.cosine_similarity(predicted_flat, noise_flat, dim=1)[:, None],
                                   predicted_flat.mean(1, keepdim=True), predicted_flat.std(1, keepdim=True)), dim=1)
                response = torch.cat([self.responses[name] for name in SD_LAYERS] + [stats], dim=1)
                outputs.append(response.cpu().numpy())
                self.responses.clear()
        result = np.concatenate(outputs) if outputs else np.empty((0, 3844), np.float32)
        if result.shape != (len(images), 3844) or not np.isfinite(result).all():
            raise RuntimeError("SD returned invalid responses; try --precision fp32 if fp16 overflows")
        return result

    def encode(self, images):
        return hash_sd(self.encode_raw(images))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.responses.clear()

    def contract(self):
        bucket, sign = sd_hash_mapping()
        return dict(model=SD_MODEL_ID, requested_revision=SD_REVISION, revision_verified_by_asset_hashes=True, assets_sha256=self.assets,
                    raw_dimension=3844, dimension=512, layers=list(SD_LAYERS), response_pool="spatial_mean_after_FP32_cast",
                    stats=["noise_mse", "noise_cosine", "predicted_mean", "predicted_std_correction1"],
                    precision=self.precision, device=str(self.device), batch_size=self.batch_size,
                    preprocessing="EXIF_RGB_short512_bicubic_center512", conditioning="empty_text", vae="mode",
                    timestep=100, noise_seed=0, noise_reseeded_per_image=True,
                    hash_algorithm="namespaced_SplitMix64_CPU_torch_scatter_add_FP32", hash_seed=20260723,
                    hash_namespace="sd15-response-control-512", hash_sha256=hashlib.sha256(bucket.tobytes()+sign.tobytes()).hexdigest(),
                    norm="sqrt(2)", norm_floor=float(np.finfo(np.float32).eps))
