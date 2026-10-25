from moviepy.editor import *
import numpy as np
from PIL import Image
import math
import numpy
import os
import cv2


def fade(clip, duration=1, type="both"):
    if type == "in":
        return clip.fadein(duration)
    elif type == "out":
        return clip.fadeout(duration)
    elif type == "both":
        return clip.fadein(duration).fadeout(duration)
    else:
        raise ValueError("type must be 'in', 'out', or 'both'")


def shake(clip, effect_duration=1, max_offset=5):
    def shake_effect(get_frame, t):
        frame = get_frame(t)

        # Only apply the effect during the specified duration
        if t < effect_duration:
            dx = np.random.randint(-max_offset, max_offset + 1)
            dy = np.random.randint(-max_offset, max_offset + 1)

            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(frame)

            # Create a new image with a black background
            result = Image.new("RGB", pil_image.size, (0, 0, 0))

            # Paste the original image with an offset
            result.paste(pil_image, (dx, dy))

            # Convert back to NumPy array
            return np.array(result)
        else:
            return frame

    return clip.fl(shake_effect)


def zoom(clip, mode="in", position="center", speed=3):
    if hasattr(clip, "fps") and clip.fps is not None:
        fps = clip.fps
    else:
        fps = 1

    duration = clip.duration
    total_frames = max(1, int(duration * fps))  # ensure at least 1 frame

    def main(getframe, t):
        frame = getframe(t)
        h, w = frame.shape[:2]
        i = t * fps
        if mode == "out":
            i = total_frames - i
        zoom = 1 + (i * ((0.1 * speed) / total_frames))
        
        # compute the extra zoom to avoid black bars
        extra_zoom = max(w / (w - 2), h / (h - 2))
        zoom *= extra_zoom

        positions = {
            "center": [(w - (w / zoom)) / 2, (h - (h / zoom)) / 2],
            "left": [0, (h - (h / zoom)) / 2],
            "right": [w - (w / zoom), (h - (h / zoom)) / 2],
            "top": [(w - (w / zoom)) / 2, 0],
            "topleft": [0, 0],
            "topright": [w - (w / zoom), 0],
            "bottom": [(w - (w / zoom)) / 2, h - (h / zoom)],
            "bottomleft": [0, h - (h / zoom)],
            "bottomright": [w - (w / zoom), h - (h / zoom)],
        }
        tx, ty = positions[position]
        M = np.array([[zoom, 0, -tx * zoom], [0, zoom, -ty * zoom]])
        frame = cv2.warpAffine(frame, M, (w, h))
        return frame

    return clip.fl(main)


def pan(clip, direction="left", zoom_factor=1.15):
    """Slow lateral drift, used for wide establishing shots.

    The frame is pre-scaled so the move never exposes the background; the
    affine matrix is a source to destination transform, which is what
    `cv2.warpAffine` expects when `WARP_INVERSE_MAP` is not set.
    """
    duration = clip.duration or 1.0
    scale = max(1.02, zoom_factor)

    def main(getframe, t):
        frame = getframe(t)
        h, w = frame.shape[:2]
        # A two pixel guard keeps bilinear sampling from bleeding the border in.
        span_x = max(0.0, (scale - 1) * w - 2)
        span_y = max(0.0, (scale - 1) * h - 2)
        progress = min(1.0, max(0.0, t / duration))

        if direction in ("left", "right"):
            fraction_x = 1.0 - progress if direction == "left" else progress
            fraction_y = 0.5
        else:
            fraction_x = 0.5
            fraction_y = progress if direction == "down" else 1.0 - progress

        matrix = np.array(
            [
                [scale, 0, -span_x * fraction_x],
                [0, scale, -span_y * fraction_y],
            ]
        )
        return cv2.warpAffine(frame, matrix, (w, h))

    return clip.fl(main)


def pop(clip, hold=0.45, amplitude=0.06):
    """A quick punch-in when a panel appears, then a clean hold."""
    def main(getframe, t):
        frame = getframe(t)
        if t >= hold:
            return frame

        h, w = frame.shape[:2]
        progress = t / hold
        if progress < 0.6:
            offset = amplitude * (progress / 0.6)
        else:
            offset = amplitude * (1.0 - (progress - 0.6) / 0.4)
        scale = 1.0 + offset

        matrix = np.array(
            [
                [scale, 0, -(scale - 1) * w / 2],
                [0, scale, -(scale - 1) * h / 2],
            ]
        )
        return cv2.warpAffine(frame, matrix, (w, h))

    return clip.fl(main)


def bob(clip, amplitude=10, period=3.0, zoom_factor=1.06):
    """A gentle float, for sticker style characters with no other motion."""
    scale = max(1.02, zoom_factor)

    def main(getframe, t):
        frame = getframe(t)
        h, w = frame.shape[:2]
        room = max(0.0, (scale - 1) * h / 2 - 2)
        drift = max(-room, min(room, amplitude * math.sin(2 * math.pi * t / period)))

        matrix = np.array(
            [
                [scale, 0, -(scale - 1) * w / 2],
                [0, scale, -(scale - 1) * h / 2 + drift],
            ]
        )
        return cv2.warpAffine(frame, matrix, (w, h))

    return clip.fl(main)


def wipe(clip, duration=0.8, feather=3):
    """Reveal the frame from white, left to right, like a drawing being inked."""
    def main(getframe, t):
        frame = getframe(t)
        if t >= duration:
            return frame

        result = frame.copy()
        h, w = frame.shape[:2]
        edge = int(w * min(1.0, max(0.0, t / duration)))
        result[:, edge:] = 255
        if 0 < edge < w:
            result[:, max(0, edge - feather) : edge] = 0
        return result

    return clip.fl(main)


if __name__ == "__main__":
    os.makedirs("data/_test/transitions", exist_ok=True)
    Image.new("RGB", (640, 360), (28, 34, 58)).save(
        "data/_test/transitions/source.png"
    )

    effects = {
        "zoom-in": lambda clip: zoom(clip),
        "zoom-out": lambda clip: zoom(clip, mode="out"),
        "fade": lambda clip: fade(clip, duration=0.5),
        "shake": lambda clip: shake(clip),
        "pan-left": lambda clip: pan(clip, direction="left"),
        "pan-right": lambda clip: pan(clip, direction="right"),
        "pop": lambda clip: pop(clip),
        "bob": lambda clip: bob(clip),
        "wipe": lambda clip: wipe(clip),
    }

    for name, effect in effects.items():
        clip = ImageClip("data/_test/transitions/source.png").set_duration(2).set_fps(24)
        effect(clip).write_videofile(
            f"data/_test/transitions/{name}.mp4", fps=24, logger=None
        )
        clip.close()
        print("wrote data/_test/transitions/" + name + ".mp4")
