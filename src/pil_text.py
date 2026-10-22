"""Pillow based text rendering for shortcap.

shortcap 1.0.5 renders captions with moviepy's `TextClip`, which shells out
to ImageMagick. ImageMagick is not installed everywhere, so this module
provides drop in replacements that use Pillow instead:

* `create_text_ex` renders a list of `Word`/`Character` objects, honouring the
  per character colours that shortcap uses for the word highlighting.
* `get_text_size_ex` measures text with the same font, so the line wrapping
  stays consistent with what is drawn.
"""

import os
from typing import Any, List, Optional, Tuple

import numpy
from PIL import Image, ImageDraw, ImageFont


def _flatten(text: Any, default_color: Any) -> List[Tuple[str, Any]]:
    """Turn a string / list of Words / list of Characters into (char, color)."""
    if isinstance(text, str):
        return [(char, default_color) for char in text]

    flattened: List[Tuple[str, Any]] = []
    items = list(text)
    for index, item in enumerate(items):
        characters = getattr(item, "characters", None)
        if characters is not None:
            for character in characters:
                flattened.append(
                    (character.text, getattr(character, "color", None) or default_color)
                )
            if index < len(items) - 1:
                flattened.append((" ", default_color))
        else:
            flattened.append(
                (getattr(item, "text", str(item)), getattr(item, "color", None) or default_color)
            )
    return flattened


def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, font_size)


def _measure(text: str, font: ImageFont.FreeTypeFont, stroke_width: int) -> Tuple[int, int, Tuple[int, int, int, int]]:
    scratch = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    padding = stroke_width + 2
    width = max(1, bbox[2] - bbox[0] + 2 * padding)
    height = max(1, bbox[3] - bbox[1] + 2 * padding)
    return width, height, bbox


def get_text_size_ex(
    text: str, font: str, fontsize: int, stroke_width: int
) -> Tuple[int, int]:
    """Measure a string, mirroring shortcap's signature."""
    try:
        pil_font = _load_font(font, fontsize)
        width, height, _ = _measure(text, pil_font, stroke_width)
        return (width, height)
    except Exception:
        return (len(text) * int(fontsize * 0.6), int(fontsize * 1.4))


def create_text_ex(
    text: Any,
    fontsize: int,
    color: str,
    font: str,
    bg_color: str = "transparent",
    blur_radius: int = 0,
    opacity: float = 1,
    stroke_color: Optional[str] = None,
    stroke_width: int = 1,
    kerning: float = 0,
):
    """Render text into a moviepy clip, mirroring shortcap's signature."""
    from moviepy.editor import ImageClip

    characters = _flatten(text, color)
    if not characters:
        characters = [(" ", color)]

    plain_text = "".join(character for character, _ in characters)
    pil_font = _load_font(font, fontsize)
    width, height, bbox = _measure(plain_text, pil_font, stroke_width)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    padding = stroke_width + 2
    origin_x = padding - bbox[0]
    origin_y = padding - bbox[1]

    if stroke_width > 0 and stroke_color:
        # The stroke pass draws the whole string so the outline covers every
        # glyph, then the coloured glyphs are drawn on top.
        draw.text(
            (origin_x, origin_y),
            plain_text,
            font=pil_font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )

    cursor = origin_x
    for character, character_color in characters:
        if character.strip():
            draw.text((cursor, origin_y), character, font=pil_font, fill=character_color)
        cursor += pil_font.getlength(character)

    if opacity < 1:
        alpha = image.getchannel("A").point(lambda value: int(value * opacity))
        image.putalpha(alpha)

    return ImageClip(numpy.array(image))


def create_text(
    text: str,
    fontsize: int,
    color: str,
    font: str,
    bg_color: str = "transparent",
    blur_radius: int = 0,
    opacity: float = 1.0,
    stroke_color: Optional[str] = None,
    stroke_width: int = 1,
    kerning: float = 0.0,
):
    """Render a plain string, mirroring shortcap's signature."""
    return create_text_ex(
        text,
        fontsize,
        color,
        font,
        bg_color=bg_color,
        blur_radius=blur_radius,
        opacity=opacity,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        kerning=kerning,
    )
