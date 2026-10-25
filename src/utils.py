import os
import re
import json
import sys
import time
from datetime import datetime
import datetime as dt_module
from typing import List, Dict, Any, Optional
from PIL import Image

# Windows consoles default to cp1252, which cannot encode the unicode that
# models like to emit. Switching to UTF-8 keeps prints from crashing.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


STORY_TYPES = [
    "Scary",
    "Mystery",
    "Bedtime",
    "Interesting History",
    "Urban Legends",
    "Motivational",
    "Fun Facts",
    "Long Form Jokes",
    "Life Pro Tips",
    "Philosophy",
    "Love",
]

STORY_TYPE_HASHTAGS = {
    "Scary": "#scary",
    "Mystery": "#mystery",
    "Bedtime": "#bedtime",
    "Interesting History": "#history",
    "Urban Legends": "#urbanlegends",
    "Motivational": "#motivation",
    "Fun Facts": "#funfacts",
    "Long Form Jokes": "#joke",
    "Life Pro Tips": "#lifeprotips",
    "Philosophy": "#philosophy",
    "Love": "#love",
}

# In-memory overrides applied on top of config.json (used by CLI flags).
_CONFIG_OVERRIDES: Dict[str, Any] = {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `override` into a copy of `base` without mutating either."""
    merged = dict(base)
    for key, value in (override or {}).items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public wrapper around the config merge."""
    return _deep_merge(base, override)


def set_config_overrides(overrides: Optional[Dict[str, Any]]) -> None:
    """Set (or clear) runtime overrides that are merged over config.json."""
    global _CONFIG_OVERRIDES
    _CONFIG_OVERRIDES = overrides or {}


def get_config_overrides() -> Dict[str, Any]:
    return dict(_CONFIG_OVERRIDES)


def create_resource_dir(script_dir, story_type, title):
    # Remove leading and trailing quotation marks and spaces
    clean_title = title.strip().strip('"')

    # Remove special characters and replace spaces with underscores
    clean_title = re.sub(r'[^\w\s-]', '', clean_title)
    clean_title = re.sub(r'[-\s]+', '_', clean_title)

    # Keep the folder name within the limits of every file system
    clean_title = clean_title.strip('_')[:100]

    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Create a directory for the story type
    story_type_dir = os.path.join(data_dir, story_type)
    os.makedirs(story_type_dir, exist_ok=True)

    # Create a directory for this story
    story_dir = os.path.join(story_type_dir, clean_title)
    os.makedirs(story_dir, exist_ok=True)

    return story_dir


def _extract_message_content(message) -> Optional[str]:
    """Read the assistant text, falling back to reasoning content.

    Reasoning models (gpt-oss) can return the answer on `reasoning` instead of
    `content`, and some providers return nothing on `content` at all.
    """
    content = getattr(message, "content", None)
    if content:
        return content
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if value:
            return value
    return None


def _parse_retry_delay_from_message(message: str) -> Optional[float]:
    """Pull a wait time out of a provider error message.

    Groq reports quota resets as "Please try again in 3m26.064s".
    """
    if not message:
        return None
    match = re.search(
        r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)s", message, re.IGNORECASE
    )
    if not match:
        return None
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _retry_delay(error, attempt: int) -> float:
    """Backoff delay for a failed API call, honouring `retry-after` headers."""
    delay = min(2 ** attempt, 60)
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except (TypeError, ValueError):
                pass
    message_delay = _parse_retry_delay_from_message(str(error))
    if message_delay:
        delay = max(delay, message_delay)
    return min(delay, 180)


def retry_delay(error, attempt: int) -> float:
    """Public wrapper around the backoff calculation."""
    return _retry_delay(error, attempt)


def call_openai_api(client, messages, max_retries=3, response_format=None, reasoning_effort=None):
    """Call the configured chat model with retries and provider-aware options.

    `response_format` is passed straight through, so OpenAI-compatible providers
    that support JSON Object / JSON Schema mode can be used. When a model is
    rate limited the next configured fallback model is tried, which matters on
    free tiers where each model has its own daily budget.
    """
    config = load_config()
    provider = config.get("providers", {}).get("llm", "openai")

    if provider == "groq":
        candidates = [config["groq"]["model"]] + list(
            config["groq"].get("model_fallbacks", [])
        )
        # gpt-oss on Groq pins temperature at 1.0; sending another value errors.
        request = {"messages": messages}
    else:
        candidates = [config["openai"]["model"]]
        request = {
            "temperature": config["openai"]["temperature"],
            "messages": messages,
        }

    if response_format is not None:
        request["response_format"] = response_format

    for model in candidates:
        request["model"] = model
        if provider == "groq" and model.startswith("openai/gpt-oss"):
            # Reasoning output can leak into the answer and break JSON, so it
            # is kept short.
            effort = reasoning_effort or config["groq"].get("reasoning_effort")
            if effort:
                request["extra_body"] = {"reasoning_effort": effort}
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**request)
                content = _extract_message_content(response.choices[0].message)
                if content:
                    return content
                print("The model returned an empty response.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                if attempt < max_retries - 1:
                    delay = _retry_delay(e, attempt)
                    if delay > 150:
                        print(
                            f"{model} needs {delay / 60:.1f} minutes before it can "
                            "be used again."
                        )
                        break
                    print(
                        f"Retrying in {delay:.0f}s... "
                        f"(Attempt {attempt + 2} of {max_retries})"
                    )
                    time.sleep(delay)
                else:
                    print("Max retries reached. Unable to get a valid response.")
        if len(candidates) > 1:
            print(f"Trying the next model instead of {model}...")
    return None


def create_empty_storyboard(title):
    return {
        "project_info": {
            "title": title,
            "user": "AI Generated",
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        },
        "storyboards": []
    }

def pick_story_type():
    print("Choose a story type:")
    for i, story_type in enumerate(STORY_TYPES, 1):
        print(f"{i}. {story_type}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(STORY_TYPES):
                return STORY_TYPES[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def pick_image_style():
    styles = list(get_style_config()["profiles"]) or [
        "photorealistic", "cinematic", "anime", "comic-book", "pixar-art"
    ]
    print("Choose an image style:")
    for i, style in enumerate(styles, 1):
        print(f"{i}. {style}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(styles):
                return styles[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def convert_to_timestamped_subtitles(chinese_storyboard_project: Dict, scene_duration: int = 10) -> List[Dict]:
    timestamped_subtitles = []
    current_time = dt_module.timedelta()

    for scene in chinese_storyboard_project['storyboards']:
        subtitles = scene['subtitles'].split('\n')
        time_per_subtitle = scene_duration / len(subtitles)

        for subtitle in subtitles:
            start_time = current_time
            end_time = current_time + dt_module.timedelta(seconds=time_per_subtitle)

            timestamped_subtitles.append({
                'start_time': start_time.total_seconds(),
                'end_time': end_time.total_seconds(),
                'text': subtitle.strip()
            })

            current_time = end_time

    return timestamped_subtitles

def format_timedelta(seconds: float) -> str:
    td = dt_module.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def save_timestamped_subtitles(timestamped_subtitles: List[Dict], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for i, subtitle in enumerate(timestamped_subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timedelta(subtitle['start_time'])} --> {format_timedelta(subtitle['end_time'])}\n")
            f.write(f"{subtitle['text']}\n\n")

def load_config(config_file='config.json'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), config_file)
    with open(config_path, 'r', encoding="utf-8") as f:
        config = json.load(f)
    if _CONFIG_OVERRIDES:
        config = _deep_merge(config, _CONFIG_OVERRIDES)
    return config


def get_preset(name: str) -> Dict[str, Any]:
    """Return the settings for a named preset, for example "youtube"."""
    presets = load_config().get("presets", {})
    if name not in presets:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {', '.join(sorted(presets)) or 'none'}"
        )
    return presets[name]


def get_video_config() -> Dict[str, int]:
    """Output size and frame rate of the rendered video."""
    video = load_config().get("video", {})
    return {
        "width": int(video.get("width", 720)),
        "height": int(video.get("height", 1280)),
        "fps": int(video.get("fps", 24)),
    }


def get_caption_config() -> Dict[str, Any]:
    """Caption placement and styling."""
    caption = load_config().get("caption", {})
    return {
        "position": caption.get("position", "center"),
        "font_size": int(caption.get("font_size", 70)),
        "padding": int(caption.get("padding", 70)),
        "line_count": int(caption.get("line_count", 1)),
    }


def get_style_config() -> Dict[str, Any]:
    """Look presets: prompt, palette, camera motion and caption mode."""
    styles = load_config().get("styles", {})
    bubble = {
        "font": "Bangers.ttf",
        "font_size": 42,
        "padding": 30,
        "max_chars": 110,
        "min_chars": 45,
        "max_width_ratio": 0.44,
        "fill": "#ffffff",
        "outline": "#111111",
        "text_color": "#111111",
        "outline_width": 4,
        "gap_seconds": 0.12,
    }
    bubble.update(styles.get("bubble", {}))
    return {
        "default": styles.get("default", "storyboard"),
        "profiles": styles.get("profiles", {}),
        "bubble": bubble,
    }


def get_narration_config() -> Dict[str, Any]:
    """Settings for how narration is chunked and paced against the images."""
    narration = load_config().get("narration", {})
    chunk_chars = int(narration.get("chunk_chars", 420))
    min_shot = float(narration.get("min_shot_seconds", 2.0))
    max_shot = float(narration.get("max_shot_seconds", 8.0))
    return {
        "enabled": bool(narration.get("enabled", True)),
        "chunk_chars": max(80, chunk_chars),
        "min_shot_seconds": max(0.5, min_shot),
        "max_shot_seconds": max(min_shot, max_shot),
        "max_retries": int(narration.get("max_retries", 2)),
    }


def get_aspect_ratio() -> str:
    """Aspect ratio string matching the configured video size."""
    video = get_video_config()
    if video["width"] > video["height"]:
        return "16:9"
    if video["height"] > video["width"]:
        return "9:16"
    return "1:1"


def create_blank_image(filename, width=None, height=None):
    if width is None or height is None:
        video = get_video_config()
        width = width or video["width"]
        height = height or video["height"]
    blank_image = Image.new('RGB', (width, height), color='black')
    blank_image.save(filename)
    print(f"Created blank image: {filename}")


def normalize_image_file(filename, width=None, height=None):
    """Center-crop and resize an image to the exact clip size.

    Every scene image has to end up with identical dimensions, otherwise
    moviepy's `concatenate_videoclips` refuses to join them. Providers also
    return JPEG under a .png filename, so re-encoding here keeps the file
    honest.
    """
    if width is None or height is None:
        video = get_video_config()
        width = width or video["width"]
        height = height or video["height"]
    try:
        with Image.open(filename) as img:
            img = img.convert("RGB")
            src_w, src_h = img.size
            if src_w == 0 or src_h == 0:
                return False

            target_ratio = width / height
            src_ratio = src_w / src_h

            if src_ratio > target_ratio:
                # Too wide: crop the sides.
                crop_w = int(round(src_h * target_ratio))
                left = max(0, (src_w - crop_w) // 2)
                img = img.crop((left, 0, left + crop_w, src_h))
            elif src_ratio < target_ratio:
                # Too tall: crop top and bottom.
                crop_h = int(round(src_w / target_ratio))
                top = max(0, (src_h - crop_h) // 2)
                img = img.crop((0, top, src_w, top + crop_h))

            img = img.resize((width, height), Image.LANCZOS)
            img.save(filename, "PNG")
        return True
    except Exception as e:
        print(f"Failed to normalize image {filename}: {e}")
        return False


def get_voice_list(tts_provider: Optional[str] = None) -> List[str]:
    """Voice personas for the active TTS provider."""
    config = load_config()
    if tts_provider is None:
        tts_provider = config.get("providers", {}).get("tts", "openai")
    settings = config.get(f"{tts_provider}_tts") or config.get(tts_provider) or {}
    voices = settings.get("tts_voices")
    if not voices:
        voices = config.get("openai", {}).get("tts_voices", [])
    return list(voices)

def pick_voice_name():
    voices = get_voice_list()
    print("Choose a voice:")
    for i, voice in enumerate(voices, 1):
        print(f"{i}. {voice}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    config = load_config()

    print("video:    ", get_video_config())
    print("caption:  ", get_caption_config())
    print("narration:", get_narration_config())
    print("providers:", config["providers"])
    print("voices:   ", get_voice_list())
