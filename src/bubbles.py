"""Comic speech bubbles.

The `storyboard` look is defined by its bubbles as much as by its flat fills,
so this module turns a scene's subtitle into one or more bubbles and hands them
to moviepy as transparent overlays. Because the narration timeline already
knows when each scene is on screen, no transcription pass is involved.

Bubbles are split on sentence boundaries and then merged until they are worth
reading, so a two word line does not get its own balloon.
"""

import os
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip

from utils import get_style_config, get_video_config

FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "font"
)

_SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")


def _font(settings):
    return ImageFont.truetype(
        os.path.join(FONT_DIR, settings["font"]), int(settings["font_size"])
    )


def split_bubbles(text, max_chars=110, min_chars=45):
    """Break a subtitle into conversation sized bubbles."""
    text = " ".join(str(text or "").split())
    if not text:
        return []

    pieces = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
        else:
            pieces.extend(textwrap.wrap(sentence, max_chars))

    bubbles = []
    for piece in pieces:
        if (
            bubbles
            and len(bubbles[-1]) < min_chars
            and len(bubbles[-1]) + 1 + len(piece) <= max_chars
        ):
            bubbles[-1] = f"{bubbles[-1]} {piece}"
        else:
            bubbles.append(piece)
    return bubbles


def _wrap(text, font, max_width):
    lines = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and font.getlength(candidate) > max_width:
            lines.append(word)
        else:
            if lines:
                lines[-1] = candidate
            else:
                lines.append(word)
    return lines or [text]


def bubble_image(text, settings, side="left", frame_width=None):
    """Render one bubble (with a tail) as an RGBA image."""
    font = _font(settings)
    pad = int(settings["padding"])
    stroke = int(settings["outline_width"])
    frame_width = frame_width or get_video_config()["width"]
    max_width = int(settings["max_width_ratio"] * frame_width) - 2 * pad

    # Bangers only has capitals; lowercase falls back to uneven small caps, so
    # the text is raised before it is measured or drawn.
    lines = _wrap(text.upper(), font, max_width)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + int(settings["font_size"] * 0.12)
    text_width = max(font.getlength(line) for line in lines)

    box_w = int(text_width) + 2 * pad
    box_h = line_height * len(lines) + 2 * pad
    tail_h = max(18, int(pad * 0.9))

    image = Image.new("RGBA", (box_w, box_h + tail_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # The tail is drawn first, then the body covers its base so the two read as
    # one shape. The last rectangle restores the body fill over the seam.
    base_x = int(box_w * (0.72 if side == "left" else 0.28))
    direction = 1 if side == "left" else -1
    base_half = max(16, stroke * 5)
    tail = [
        (base_x - base_half, box_h - stroke),
        (base_x + base_half, box_h - stroke),
        (base_x + direction * int(tail_h * 0.7), box_h + tail_h),
    ]
    draw.polygon(tail, fill=settings["fill"], outline=settings["outline"], width=stroke)

    draw.rounded_rectangle(
        [stroke // 2, stroke // 2, box_w - stroke // 2, box_h - stroke // 2],
        radius=min(box_h // 3, 34),
        fill=settings["fill"],
        outline=settings["outline"],
        width=stroke,
    )
    draw.rectangle(
        [base_x - base_half + stroke, box_h - 2 * stroke, base_x + base_half - stroke, box_h],
        fill=settings["fill"],
    )

    for index, line in enumerate(lines):
        draw.text(
            (box_w / 2, pad + line_height * (index + 0.5)),
            line,
            font=font,
            fill=settings["text_color"],
            anchor="mm",
        )
    return image


def _timed(bubbles, duration, gap):
    """Give each bubble a slice of the scene, proportional to its length."""
    total = sum(len(bubble) for bubble in bubbles) or 1
    cursor = 0.0
    slots = []
    for bubble in bubbles:
        span = duration * len(bubble) / total
        slots.append((bubble, cursor, max(0.4, span - gap)))
        cursor += span
    return slots


def bubble_clips(text, start, duration, index=0, frame_size=None):
    """Overlay clips for one scene, already positioned inside the frame."""
    settings = get_style_config()["bubble"]
    bubbles = split_bubbles(
        text, int(settings["max_chars"]), int(settings["min_chars"])
    )
    if not bubbles or duration <= 0:
        return []

    if frame_size:
        frame_width, frame_height = frame_size
    else:
        video = get_video_config()
        frame_width, frame_height = video["width"], video["height"]

    side = "left" if index % 2 == 0 else "right"
    margin_x = int(frame_width * 0.05)
    margin_y = int(frame_height * 0.07)

    clips = []
    for bubble, offset, length in _timed(
        bubbles, duration, float(settings["gap_seconds"])
    ):
        image = np.array(bubble_image(bubble, settings, side=side, frame_width=frame_width))
        if side == "left":
            x = margin_x
        else:
            x = frame_width - image.shape[1] - margin_x

        clips.append(
            ImageClip(image, transparent=True)
            .set_start(start + offset)
            .set_duration(length)
            .set_position((x, margin_y))
        )
    return clips


if __name__ == "__main__":
    from moviepy.editor import ColorClip, CompositeVideoClip

    os.makedirs("data/_test/bubbles", exist_ok=True)

    text = (
        "I was very scared of Carlos. He had just taken out Adolpho and Roberto. "
        "\u201cWhat did you say was your name, friend?\u201d he asked."
    )
    print(split_bubbles(text, 110, 45))

    background = ColorClip((1280, 720), color=(232, 226, 210), duration=6.0)
    overlays = bubble_clips(text, 0.0, 6.0, index=0, frame_size=(1280, 720))
    final = CompositeVideoClip([background] + overlays)
    final.write_videofile("data/_test/bubbles/demo.mp4", fps=24, logger=None)
    final.close()
    print("wrote data/_test/bubbles/demo.mp4")
