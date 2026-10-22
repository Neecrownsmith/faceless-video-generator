import os
import shutil

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

# The repository level .env has to be loaded before anything reads the
# environment, because shortcap resolves its credentials at import time.
load_dotenv(os.path.join(REPO_DIR, ".env"))

# shortcap calls get_openai_api_key() while it is being imported, which raises
# when no OpenAI key is present. We never use shortcap's OpenAI transcriber -
# the segments are produced by the configured STT provider - so a placeholder
# is enough to make the import work.
os.environ.setdefault("OPENAI_API_KEY", "not-used")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

from moviepy.editor import (  # noqa: E402
    ImageClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    AudioFileClip
)
from audio_generator import generate_audio  # noqa: E402
from transitions import zoom  # noqa: E402
import shortcap  # noqa: E402
from providers import ensure_ffmpeg_on_path, ffmpeg_exe, transcribe_video  # noqa: E402
from utils import get_caption_config, get_video_config  # noqa: E402


def _patch_shortcap() -> None:
    """Work around three problems inside shortcap 1.0.5.

    1. shortcap/utils.py compares the requested font against DEFAULT_FONT but
       never imports it, so any custom font path raises a NameError.
    2. shortcap shells out to a bare "ffmpeg" while the bundled binary is
       named "ffmpeg-win64-<version>.exe", so the call fails with WinError 2.
    3. shortcap renders captions through moviepy's TextClip, which requires
       ImageMagick. The Pillow based replacements avoid that dependency.
    """
    try:
        import sys

        import shortcap.utils as shortcap_utils
        from shortcap.config import DEFAULT_FONT

        if not hasattr(shortcap_utils, "DEFAULT_FONT"):
            shortcap_utils.DEFAULT_FONT = DEFAULT_FONT

        exe = ffmpeg_exe()
        if exe and os.path.basename(exe).lower() != "ffmpeg.exe":
            original_ffmpeg = shortcap_utils.ffmpeg

            def ffmpeg(command, *args, **kwargs):
                if command and os.path.basename(str(command[0])).lower() in (
                    "ffmpeg",
                    "ffmpeg.exe",
                ):
                    command = [exe] + list(command[1:])
                return original_ffmpeg(command, *args, **kwargs)

            shortcap_utils.ffmpeg = ffmpeg
            # shortcap/__init__.py rebinds the name `add_captions` to the
            # function, so the module has to be fetched from sys.modules.
            module = sys.modules.get("shortcap.add_captions")
            if module is not None:
                module.ffmpeg = ffmpeg

        # Replace the ImageMagick based text rendering with Pillow.
        import pil_text

        shortcap_utils.get_text_size_ex = pil_text.get_text_size_ex
        shortcap_utils.create_text_ex = pil_text.create_text_ex
        shortcap_utils.create_text = pil_text.create_text

        add_captions_module = sys.modules.get("shortcap.add_captions")
        if add_captions_module is not None:
            add_captions_module.create_text_ex = pil_text.create_text_ex

        text_renderer_module = sys.modules.get("shortcap.text_renderer")
        if text_renderer_module is not None:
            text_renderer_module.create_text_ex = pil_text.create_text_ex
            text_renderer_module.create_text = pil_text.create_text
            text_renderer_module.get_text_size_ex = pil_text.get_text_size_ex
    except Exception as e:
        print(f"Could not apply the shortcap workarounds: {e}")


_patch_shortcap()

font_path = os.path.join(REPO_DIR, "font")

# shortcap shells out to a bare `ffmpeg`, so make sure one is reachable.
ensure_ffmpeg_on_path()


def _caption_options():
    caption = get_caption_config()
    return {
        "font": os.path.join(font_path, "TitanOne.ttf"),
        "font_size": caption["font_size"],
        "font_color": "white",
        "stroke_width": 3,
        "stroke_color": "black",
        "shadow_strength": 1.0,
        "shadow_blur": 0.1,
        "highlight_current_word": True,
        "word_highlight_color": "yellow",
        "line_count": caption["line_count"],
        "padding": caption["padding"],
        "position": caption["position"],
    }


def add_subtitles(output_file, output_file_subtitle):
    """Burn word level captions into the video.

    The transcript is produced by the configured STT provider (Groq Whisper by
    default) and handed to shortcap, which does the actual rendering.
    """
    segments = None
    try:
        print("\nTranscribing audio for captions...")
        segments = transcribe_video(output_file)
        word_count = len(segments[0]["words"]) if segments else 0
        print(f"Transcription produced {word_count} words.")
    except Exception as e:
        print(f"Caption transcription failed: {e}")

    if segments:
        shortcap.add_captions(
            video_file=output_file,
            output_file=output_file_subtitle,
            use_local_whisper=False,
            segments=segments,
            **_caption_options(),
        )
        return

    print("Falling back to local whisper for captions...")
    try:
        shortcap.add_captions(
            video_file=output_file,
            output_file=output_file_subtitle,
            use_local_whisper=True,
            **_caption_options(),
        )
    except Exception as e:
        print(f"Local whisper unavailable ({e}); copying the video without captions.")
        shutil.copyfile(output_file, output_file_subtitle)


def create_video(client, storyboard_project, output_file, audio_dir, voice_name,
                 timeline=None, narration_track=None, subtitles=True):
    """Assemble the video.

    When a timeline and a narration track are supplied the images are timed
    against the real audio, which decouples how many pictures the video has
    from how the narration was chunked. Without them the original behaviour -
    one scene, one request, one clip - is used.
    """
    if timeline and narration_track:
        _create_video_from_timeline(
            storyboard_project, output_file, timeline, narration_track, subtitles
        )
        return

    clips = []
    for scene in storyboard_project['storyboards']:
        # Generate audio for the subtitle
        audio_file = scene['audio']
        if not generate_audio(client, scene['subtitles'], audio_file, voice_name):
            print(f"Skipping scene {scene['scene_number']}: no narration available.")
            continue

        if not os.path.exists(audio_file):
            print(f"Skipping scene {scene['scene_number']}: audio file missing.")
            continue

        image_file = scene.get('image')
        if not image_file or not os.path.exists(image_file):
            print(f"Skipping scene {scene['scene_number']}: image file missing.")
            continue

        # Create audio clip
        audio_clip = AudioFileClip(audio_file)

        # Create image clip with duration matching the audio
        image_clip = ImageClip(image_file).set_duration(audio_clip.duration)

        # Combine image, text, and audio
        video_clip = image_clip.set_audio(audio_clip)

        # Apply transition effect
        transition_type = scene['transition_type']

        if transition_type == 'zoom-in':
            clips.append(zoom(video_clip))
        elif transition_type == 'zoom-out':
            clips.append(zoom(video_clip, mode='out'))
        else:
            clips.append(video_clip)

    if not clips:
        raise RuntimeError("No clips were produced; cannot create the video.")

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_file, fps=get_video_config()["fps"])

    if subtitles:
        add_subtitles(output_file, output_file.replace('.mp4', '_subtitle.mp4'))


def _create_video_from_timeline(storyboard_project, output_file, timeline, narration_track,
                                subtitles=True):
    """Render silent image clips and attach one merged narration track."""
    storyboards = storyboard_project['storyboards']
    clips = []

    for entry in timeline:
        scene = storyboards[entry['scene_index']]
        image_file = scene.get('image')
        if not image_file or not os.path.exists(image_file):
            print(f"Skipping scene {scene.get('scene_number')}: image file missing.")
            continue

        duration = float(entry.get('duration') or 0)
        if duration <= 0:
            continue

        clip = ImageClip(image_file).set_duration(duration)

        transition_type = scene.get('transition_type')
        if transition_type == 'zoom-in':
            clip = zoom(clip)
        elif transition_type == 'zoom-out':
            clip = zoom(clip, mode='out')

        clips.append(clip)

    if not clips:
        raise RuntimeError("No clips were produced; cannot create the video.")

    final_clip = concatenate_videoclips(clips)

    if narration_track and os.path.exists(narration_track):
        narration_clip = AudioFileClip(narration_track)
        # The timeline is measured from the same audio, so any gap here is a
        # rounding remainder rather than a missing section of narration.
        if final_clip.duration > narration_clip.duration + 0.05:
            print(
                f"Video is {final_clip.duration - narration_clip.duration:.2f}s longer "
                "than the narration; the tail will be silent."
            )
        final_clip = final_clip.set_audio(narration_clip)
    else:
        print("No narration track was supplied; rendering the video silent.")

    final_clip.write_videofile(output_file, fps=get_video_config()["fps"])

    if subtitles:
        add_subtitles(output_file, output_file.replace('.mp4', '_subtitle.mp4'))


if __name__ == "__main__":
    from PIL import Image

    os.makedirs("data/_test/video", exist_ok=True)

    images = []
    for index, color in enumerate([(24, 32, 64), (64, 32, 24), (24, 64, 48)], 1):
        path = f"data/_test/video/frame_{index}.png"
        Image.new("RGB", (640, 360), color).save(path)
        images.append(path)

    project = {
        "storyboards": [
            {"scene_number": index, "image": path, "transition_type": "zoom-in"}
            for index, path in enumerate(images, 1)
        ]
    }
    timeline = [
        {
            "scene_index": index,
            "scene_number": index + 1,
            "start": 2.0 * index,
            "duration": 2.0,
        }
        for index in range(len(images))
    ]

    _create_video_from_timeline(
        project, "data/_test/video/demo.mp4", timeline, None, subtitles=False
    )
    print("wrote data/_test/video/demo.mp4")
