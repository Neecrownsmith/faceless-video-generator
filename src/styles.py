"""Look presets for the scenes.

A style decides three things:

* the prompt suffix that steers the image model towards a look,
* the palette pass that locks the look in after generation, because a diffusion
  model drifts between shots,
* the camera motion and caption treatment used when the video is assembled.

Five presets ship with the repository, each mirroring a reference board:
`storyboard` (flat cartoon), `noir` (sepia ink), `flat` (bright vector),
`chibi` (sticker art) and `doodle` (stick figures).

Older style names such as "cinematic" or "anime" still work: the name itself
becomes the prompt suffix and the image is left untouched.

Prompts are written positively on purpose. Flux has no negative prompt field,
and words like "watermark" or "text" inside the prompt make it draw exactly
those things, so nothing style related is phrased as an exclusion.
"""

import os

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from utils import get_style_config

FALLBACK = {
    "prompt": "",
    "palette": "none",
    "motion": ["zoom-in"],
    "captions": "word",
}

MOTIONS = ("zoom-in", "zoom-out", "pop", "bob", "wipe", "pan-left", "pan-right")


STYLE_ALIASES = {
    "afro": "afro-cartoon",
    "afro-anim": "afro-cartoon",
    "cartoon-skit": "afro-cartoon",
    "webtoon": "afro-cartoon",
    "kisara": "afro-cartoon",
    "stick": "stickman",
    "stick-man": "stickman",
    "childrens-book": "storybook",
    "kids-book": "storybook",
}


def get_profile(style):
    """Merge the configured profile for `style` with the defaults."""
    normalized_style = STYLE_ALIASES.get(style.lower() if isinstance(style, str) else "", style)
    profile = dict(FALLBACK)
    profile.update(get_style_config()["profiles"].get(normalized_style, {}))
    if not profile["prompt"]:
        profile["prompt"] = style
    return profile


def style_prompt(style):
    """The text appended to every image prompt for this style."""
    return get_profile(style)["prompt"]


def motions_for(style):
    """Camera moves to cycle through, one per scene."""
    motion = get_profile(style)["motion"]
    if isinstance(motion, str):
        motion = [motion]
    motion = [item for item in motion if item in MOTIONS]
    return motion or ["zoom-in"]


def motion_for(style, index):
    motion = motions_for(style)
    return motion[index % len(motion)]


def captions_for(style):
    """`bubble` draws comic speech bubbles instead of word captions."""
    return get_profile(style)["captions"]


def _vignette(image, strength=0.35):
    array = np.asarray(image).astype(np.float32) / 255.0
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2) / (width / 2)
    ny = (yy - height / 2) / (height / 2)
    falloff = 1.0 - strength * np.clip((nx**2 + ny**2) / 2.0, 0, 1)
    array = array * falloff[..., None]
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def _flat(image):
    """Posterise towards a handful of flat fills with a hard black outline feel."""
    image = ImageEnhance.Color(image).enhance(1.35)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    return image.quantize(colors=32, method=Image.MEDIANCUT, dither=Image.NONE).convert("RGB")


def _sepia(image):
    grey = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)
    toned = ImageOps.colorize(grey, black="#241d16", white="#f2e8d5", mid="#8d7a5e")
    return _vignette(toned)


def _vivid(image):
    image = ImageEnhance.Color(image).enhance(1.3)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return ImageEnhance.Brightness(image).enhance(1.02)


def _soft(image):
    image = ImageEnhance.Color(image).enhance(1.12)
    image = ImageEnhance.Brightness(image).enhance(1.04)
    return Image.blend(image, Image.new("RGB", image.size, (255, 255, 255)), 0.06)


def _mono(image):
    grey = ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=1)
    # Paper stays pure white, ink stays black, and the narrow middle band keeps
    # the line art from turning into a hard threshold.
    lookup = [
        0 if value < 60 else 255 if value > 205 else int((value - 60) * 255 / 145)
        for value in range(256)
    ]
    return grey.point(lookup).convert("RGB")


def _warm(image):
    """Warm vibrant sunlight grading suitable for modern 2D animation skits."""
    image = ImageEnhance.Color(image).enhance(1.22)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = ImageEnhance.Brightness(image).enhance(1.02)
    return image


PALETTES = {
    "none": None,
    "flat": _flat,
    "sepia": _sepia,
    "vivid": _vivid,
    "soft": _soft,
    "mono": _mono,
    "warm": _warm,
}


def apply_palette(path, style):
    """Re-grade a generated image so the whole video shares one palette."""
    palette = PALETTES.get(get_profile(style)["palette"])
    if palette is None:
        return True

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
        palette(image).save(path, "PNG")
        return True
    except Exception as e:
        print(f"Could not apply the {style} palette to {path}: {e}")
        return False


if __name__ == "__main__":
    from PIL import ImageDraw

    os.makedirs("data/_test/styles", exist_ok=True)

    base = Image.new("RGB", (1280, 720), (198, 220, 236))
    draw = ImageDraw.Draw(base)
    draw.rectangle([0, 520, 1280, 720], fill=(96, 128, 88))
    draw.ellipse([520, 120, 780, 380], fill=(226, 186, 150))
    draw.rectangle([500, 380, 800, 700], fill=(70, 84, 120))

    for style in get_style_config()["profiles"]:
        path = f"data/_test/styles/{style}.png"
        base.save(path)
        apply_palette(path, style)
        print(style, "->", path)
        print("  prompt:", style_prompt(style))
        print("  motion:", motions_for(style), "| captions:", captions_for(style))
