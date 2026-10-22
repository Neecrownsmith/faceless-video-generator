"""Local image generation.

Runs a distilled Stable Diffusion model (SD-Turbo by default) on the CPU
through diffusers. It needs no API key and no credits, which makes it the
fallback that always works.

The model is downloaded from Hugging Face on first use and then cached.
"""

import io
import os
import threading
from typing import Optional

from utils import load_config


_PIPELINE = None
_LOCK = threading.Lock()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)


def _model_source(config: dict) -> str:
    """Return the local model folder when it exists, else the hub id."""
    local_dir = config.get("local_dir")
    if local_dir:
        path = local_dir if os.path.isabs(local_dir) else os.path.join(REPO_DIR, local_dir)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")):
            return path
    return config.get("model", "stabilityai/sd-turbo")


def get_pipeline():
    """Load the diffusion pipeline once and reuse it for every image."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        import torch
        from diffusers import AutoPipelineForText2Image

        config = load_config().get("local", {})
        source = _model_source(config)
        threads = int(config.get("threads", 4) or 4)
        torch.set_num_threads(threads)

        if source == config.get("model"):
            print(f"Loading the local image model {source} (first run downloads it)...")
        else:
            print(f"Loading the local image model from {source}")

        pipeline = AutoPipelineForText2Image.from_pretrained(
            source,
            variant=config.get("variant", "fp16"),
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)
        _PIPELINE = pipeline
        print(f"Local image model ready ({source}).")
        return _PIPELINE


def local_image_api(prompt: str, max_retries: int = 1) -> Optional[bytes]:
    """Generate an image on this machine and return it as PNG bytes."""
    config = load_config().get("local", {})

    try:
        pipeline = get_pipeline()
    except Exception as e:
        print(f"Could not load the local image model: {e}")
        return None

    width = int(config.get("width", 0) or 0)
    height = int(config.get("height", 0) or 0)
    if not width or not height:
        # Derive a size that matches the video aspect ratio while keeping the
        # pixel count low enough to render quickly on a CPU.
        from utils import get_video_config

        video = get_video_config()
        target_pixels = float(config.get("max_pixels", 590000))
        scale = (target_pixels / max(1, video["width"] * video["height"])) ** 0.5
        width = max(64, int(round(video["width"] * scale / 64)) * 64)
        height = max(64, int(round(video["height"] * scale / 64)) * 64)

    steps = int(config.get("steps", 1))
    guidance = float(config.get("guidance_scale", 0.0))

    for attempt in range(max_retries):
        try:
            result = pipeline(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )
            image = result.images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            print(f"Local image generation failed (attempt {attempt + 1}/{max_retries}): {e}")
    return None



if __name__ == "__main__":
    import os

    os.makedirs("data/_test/local", exist_ok=True)
    image_bytes = local_image_api("A fantasy landscape with mountains and a river")
    if image_bytes:
        with open("data/_test/local/output.png", "wb") as handle:
            handle.write(image_bytes)
        print("wrote data/_test/local/output.png")