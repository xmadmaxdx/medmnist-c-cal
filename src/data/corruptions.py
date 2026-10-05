"""Clinical-style corruptions with severity scaling."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CORRUPTION_NAMES = (
    "gaussian_noise",
    "shot_noise",
    "blur",
    "brightness",
    "contrast",
    "pixelate",
    "jpeg",
    "spatter",
)

_CATEGORIES = {
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "blur": "blur",
    "brightness": "color",
    "contrast": "color",
    "pixelate": "digital",
    "jpeg": "digital",
    "spatter": "task",
}


def corruption_category(name: str) -> str:
    if name not in _CATEGORIES:
        raise ValueError(f"unknown corruption: {name}")
    return _CATEGORIES[name]


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_corruption(img: Image.Image, name: str, severity: int, seed: int = 0) -> Image.Image:
    """Apply one corruption at severity 1-5 deterministically."""
    if name not in CORRUPTION_NAMES:
        raise ValueError(f"unknown corruption: {name}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in 1..5")
    rng = np.random.default_rng(seed)
    arr = _to_array(img)
    if name == "gaussian_noise":
        std = (2.0, 6.0, 12.0, 20.0, 30.0)[severity - 1]
        arr = arr + rng.normal(0.0, std, arr.shape)
        return _to_image(arr)
    if name == "shot_noise":
        lam = (60.0, 25.0, 12.0, 6.0, 3.0)[severity - 1]
        scaled = np.clip(arr / 255.0 * lam, 0, None)
        noisy = rng.poisson(scaled).astype(np.float32) / lam * 255.0
        return _to_image(noisy)
    if name == "blur":
        radius = (0.4, 0.8, 1.2, 1.8, 2.5)[severity - 1]
        return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    if name == "brightness":
        factor = (1.08, 1.18, 1.32, 1.5, 1.7)[severity - 1]
        return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)
    if name == "contrast":
        factor = (0.92, 0.8, 0.65, 0.5, 0.35)[severity - 1]
        return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)
    if name == "pixelate":
        down = (0.85, 0.7, 0.55, 0.4, 0.3)[severity - 1]
        w, h = img.size
        small = img.convert("RGB").resize((max(1, int(w * down)), max(1, int(h * down))), Image.BILINEAR)
        return small.resize((w, h), Image.NEAREST)
    if name == "jpeg":
        quality = (85, 65, 45, 28, 12)[severity - 1]
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    # Spatter mimics stain deposit using random dark blobs.
    # Densities scale with image area: reference densities target 224px images,
    # so 28px inputs use density/64 to avoid full-frame blackout.
    base_density = (30, 80, 160, 300, 500)[severity - 1]
    h0, w0 = _to_array(img).shape[:2]
    density = max(1, int(round(base_density * (h0 * w0) / (224.0 * 224.0))))
    out = _to_array(img)
    h, w, _ = out.shape
    # Blob radii scale with image size like the counts: reference radii target
    # 224px images, so 28px inputs use radius/8 (min 1px) to keep the occluded
    # fraction comparable instead of blacking out whole frames.
    scale = min(h, w) / 224.0
    max_r = max(2, int(round((3 + severity) * scale)))
    for _ in range(density):
        y = int(rng.integers(0, h))
        x = int(rng.integers(0, w))
        r = int(rng.integers(1, max_r))
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * 0.4
    return _to_image(out)
