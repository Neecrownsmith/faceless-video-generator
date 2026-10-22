"""Provider layer for the faceless video generator.

Every outbound API call goes through here so the rest of the codebase stays
provider agnostic. Groq, OpenAI, Felo and Gemini are all OpenAI-compatible or
expose a small REST surface, so no extra SDK is required.
"""

import asyncio
import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

from utils import load_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

# Load the repository level .env once, no matter which entry point is used.
load_dotenv(os.path.join(REPO_DIR, ".env"))


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #

def ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary (bundled with imageio-ffmpeg, or on PATH)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def ensure_ffmpeg_on_path() -> Optional[str]:
    """Make sure a bare `ffmpeg` command resolves.

    shortcap shells out to `ffmpeg` by name, so the bundled binary's directory
    has to be on PATH.
    """
    exe = ffmpeg_exe()
    if not exe:
        return None
    directory = os.path.dirname(exe)
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current
    return exe


def _atempo_chain(tempo: float) -> str:
    """Build an atempo filter chain; ffmpeg only accepts 0.5x - 2.0x per stage."""
    factors: List[float] = []
    remaining = float(tempo)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def concat_audio_files(
    paths: List[str],
    output_path: str,
    tempo: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> str:
    """Join audio chunks into one file, optionally changing the tempo.

    Providers emit different sample rates, and ffmpeg's concat filter refuses
    to join mismatched inputs, so `sample_rate` normalizes everything first.
    """
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg is not available")
    if not paths:
        raise ValueError("No audio files to concatenate")

    command = [exe, "-y"]
    for path in paths:
        command.extend(["-i", path])

    resample = f"aresample={sample_rate}," if sample_rate else ""
    count = len(paths)
    if count == 1:
        graph = f"[0:a]{resample}{_atempo_chain(tempo) if tempo else 'anull'}[out]"
    else:
        inputs = "".join(f"[{index}:a]" for index in range(count))
        graph = f"{inputs}concat=n={count}:v=0:a=1[cat]"
        graph += f";[cat]{resample}{_atempo_chain(tempo) if tempo else 'anull'}[out]"

    command.extend(
        [
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-ac",
            "1",
            output_path,
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")
    return output_path


def extract_audio(video_file: str, output_path: str, sample_rate: int = 16000) -> str:
    """Extract a mono wav track from a video file."""
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg is not available")
    command = [
        exe,
        "-y",
        "-i",
        video_file,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return output_path


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #

def _build_client(provider: str) -> Optional[OpenAI]:
    config = load_config()
    if provider == "groq":
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=config["groq"]["base_url"],
        )
    if provider == "openai":
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if provider in ("gemini", "pollinations", "edge"):
        # These are called through small REST surfaces (or a local client), so
        # there is no OpenAI compatible SDK object to build.
        return None
    raise ValueError(f"Unsupported provider: {provider}")


def get_llm_client() -> OpenAI:
    config = load_config()
    return _build_client(config.get("providers", {}).get("llm", "openai"))


def get_tts_client() -> Optional[OpenAI]:
    config = load_config()
    return _build_client(config.get("providers", {}).get("tts", "openai"))


def get_stt_client() -> OpenAI:
    config = load_config()
    return _build_client(config.get("providers", {}).get("stt", "openai"))


def get_audio_extension(tts_provider: Optional[str] = None) -> str:
    """File extension matching the active TTS provider's output format."""
    config = load_config()
    provider = tts_provider or config.get("providers", {}).get("tts", "openai")
    settings = config.get(f"{provider}_tts") or config.get(provider) or {}
    audio_format = settings.get("tts_format", "mp3")
    return "wav" if audio_format == "wav" else "mp3"


def describe_providers() -> Dict[str, Any]:
    config = load_config()
    providers = config.get("providers", {})
    image = providers.get("image", "replicate")
    if isinstance(image, str):
        image = [image]
    return {
        "llm": providers.get("llm", "openai"),
        "tts": providers.get("tts", "openai"),
        "stt": providers.get("stt", "openai"),
        "image": image,
    }


# --------------------------------------------------------------------------- #
# Text to speech helpers
# --------------------------------------------------------------------------- #

def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of at most `max_chars`, preferring word breaks."""
    text = (text or "").strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    def flush_word_buffer(buffer: str) -> str:
        """Hard-split an oversized sentence into word-sized pieces."""
        pieces: List[str] = []
        current_piece = ""
        for word in buffer.split():
            candidate = f"{current_piece} {word}".strip()
            if len(candidate) > max_chars:
                if current_piece:
                    pieces.append(current_piece)
                current_piece = word
            else:
                current_piece = candidate
        if current_piece:
            pieces.append(current_piece)
        return pieces[-1] if pieces else ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if len(candidate) > max_chars:
                    if piece:
                        chunks.append(piece)
                    piece = word
                else:
                    piece = candidate
            current = piece
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------- #
# Gemini text to speech
# --------------------------------------------------------------------------- #

_LAST_GEMINI_TTS_CALL = 0.0
_GEMINI_TTS_EXHAUSTED: set = set()


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM samples in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def _gemini_retry_delay(response: Any) -> Optional[float]:
    """Pull the RetryInfo delay out of a Gemini error payload."""
    try:
        details = response.json().get("error", {}).get("details", []) or []
        for detail in details:
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                raw = str(detail.get("retryDelay", "")).rstrip("s")
                return float(raw)
    except Exception:
        pass
    return None


def gemini_tts_bytes(text: str, voice: str, model: Optional[str] = None) -> Optional[bytes]:
    """Synthesize speech with a Gemini TTS model.

    Gemini's TTS models are available on the free tier, unlike its image
    models. Some of them return a WAV container directly, others return raw
    PCM that needs a header.

    `model` pins a single model; the configured `tts_models` are tried
    afterwards when it fails.
    """
    config = load_config()
    gemini_config = config["gemini"]
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set; skipping Gemini speech synthesis.")
        return None

    base_url = gemini_config["base_url"].rstrip("/")
    versions = gemini_config.get("api_versions", ["v1beta"])
    models = [model] if model else []
    models += [
        name
        for name in (gemini_config.get("tts_models") or [gemini_config.get("tts_model")])
        if name not in models
    ]
    max_retries = gemini_config.get("max_retries", 3)
    min_interval = float(gemini_config.get("tts_min_interval", 0) or 0)

    body_variants = [
        # The documented camelCase shape.
        {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        },
        # Newer models expose the proto field names instead.
        {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
                },
            },
        },
        {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": voice}}
                },
            },
        },
    ]

    global _LAST_GEMINI_TTS_CALL

    for model in models:
        if model in _GEMINI_TTS_EXHAUSTED:
            continue
        for version in versions:
            url = f"{base_url}/{version}/models/{model}:generateContent"
            body_index = 0
            for attempt in range(max_retries):
                body = body_variants[body_index]
                elapsed = time.time() - _LAST_GEMINI_TTS_CALL
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

                try:
                    response = requests.post(
                        url,
                        headers={
                            "x-goog-api-key": api_key,
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=180,
                    )
                    _LAST_GEMINI_TTS_CALL = time.time()
                except requests.RequestException as e:
                    print(f"Gemini TTS request error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                if response.status_code == 404:
                    break  # unknown model or API version

                if response.status_code == 429:
                    delay = _gemini_retry_delay(response) or (2 ** attempt)
                    if delay > 120:
                        # The daily free tier quota for this model is spent.
                        print(
                            f"{model} is out of free tier quota "
                            f"({delay / 3600:.1f} hours until it resets). "
                            "Trying the next speech model."
                        )
                        _GEMINI_TTS_EXHAUSTED.add(model)
                        break
                    print(
                        f"Gemini TTS rate limited, retrying in {delay:.0f}s "
                        f"(attempt {attempt + 2}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(min(delay, 60))
                        continue
                    break

                if response.status_code == 400 and body_index < len(body_variants) - 1:
                    # Some models expect the proto (snake_case) field names.
                    body_index += 1
                    continue

                if response.status_code >= 400:
                    print(f"Gemini TTS error {response.status_code}: {response.text[:300]}")
                    return None

                payload = response.json()
                candidates = payload.get("candidates") or []
                if not candidates:
                    print("Gemini TTS returned no candidates.")
                    return None
                parts = candidates[0].get("content", {}).get("parts", []) or []
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if not inline or not inline.get("data"):
                        continue
                    mime_type = (inline.get("mimeType") or inline.get("mime_type") or "").lower()
                    raw = base64.b64decode(inline["data"])
                    if "wav" in mime_type:
                        return raw
                    # Raw PCM (audio/L16;codec=pcm;rate=24000)
                    return _pcm_to_wav(raw, sample_rate=24000)
                print("Gemini TTS response contained no audio data.")
                return None
    return None


# --------------------------------------------------------------------------- #
# Pollinations text to speech
# --------------------------------------------------------------------------- #

# Models offered by Pollinations that are reachable on the free tier
# (`paid_only: false` in https://gen.pollinations.ai/audio/models).
POLLINATIONS_FREE_TTS_MODELS = [
    "microsoft/mai-voice-2.1",
    "microsoft/mai-voice-2.1-flash",
    "openai/tts-1",
    "openai/tts-1-hd",
]

# Voices accepted per model family on the Pollinations audio endpoint.
DEFAULT_MAI_VOICE = "en-US-Olivia"
OPENAI_TTS_VOICES = ["nova", "shimmer", "echo", "onyx", "fable", "alloy"]


def pollinations_tts_bytes(
    text: str,
    voice: str,
    model: Optional[str] = None,
    speed: Optional[float] = None,
) -> Optional[bytes]:
    """Synthesize speech with Pollinations' OpenAI compatible audio endpoint.

    The models that are not `paid_only` draw from the free Pollen allowance and
    have no per day character cap, which is what makes them usable where Groq's
    Orpheus free tier is limited to 3600 tokens a day.
    """
    config = load_config()
    p_config = config.get("pollinations_tts", {})
    base_url = (p_config.get("base_url") or "https://gen.pollinations.ai").rstrip("/")
    api_key = os.getenv("POLLINATIONS_API_KEY")
    if not api_key:
        print("POLLINATIONS_API_KEY is not set; skipping Pollinations TTS.")
        return None

    text = (text or "").strip()
    if not text:
        return None

    max_chars = int(p_config.get("max_chars", 4000))
    if len(text) > max_chars:
        print(
            f"Pollinations TTS input is {len(text)} characters, over the "
            f"{max_chars} limit; truncating."
        )
        text = text[:max_chars]

    retries = int(p_config.get("max_retries", 3))
    audio_format = p_config.get("tts_format", "mp3")
    if speed is None:
        speech_rate = (config.get("tts") or {}).get("speech_rate")
        speed = float(speech_rate) if speech_rate else 1.0

    models = [model] if model else []
    models += [
        name
        for name in p_config.get("fallback_models", [])
        if name not in models
    ]
    if not models:
        models = POLLINATIONS_FREE_TTS_MODELS

    url = f"{base_url}/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # The MAI models only accept the `en-US-*` / `en-GB-*` voices and reject the
    # `speed` parameter, while the OpenAI models only accept the six OpenAI
    # voices and do support it. The voice is therefore adapted per model family
    # instead of failing the whole fallback chain on a 400.
    preferred_mai_voice = voice if voice not in OPENAI_TTS_VOICES else DEFAULT_MAI_VOICE
    preferred_openai_voice = voice if voice in OPENAI_TTS_VOICES else OPENAI_TTS_VOICES[0]

    for current_model in models:
        is_openai = current_model.startswith("openai/")
        payload = {
            "model": current_model,
            "input": text,
            "voice": preferred_openai_voice if is_openai else preferred_mai_voice,
            "response_format": audio_format,
        }
        if is_openai:
            payload["speed"] = speed
        for attempt in range(retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=180
                )
            except requests.RequestException as e:
                print(
                    f"Pollinations TTS request error (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue

            if response.status_code == 200 and response.content:
                print(
                    f"Pollinations synthesized speech with {current_model} "
                    f"({len(response.content)} bytes)."
                )
                return response.content

            if response.status_code in (401, 402, 403):
                print(
                    "Pollinations TTS is unavailable for this key: "
                    f"{response.status_code} {response.text[:300]}"
                )
                return None

            if response.status_code == 400:
                # Usually an unknown voice for that model; try the next model.
                print(
                    f"Pollinations rejected {current_model} "
                    f"(400): {response.text[:200]}"
                )
                break

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < retries - 1:
                    print(
                        f"Pollinations TTS {response.status_code}, retrying "
                        f"(attempt {attempt + 2}/{retries})"
                    )
                    time.sleep(2 ** attempt)
                    continue
                print(f"Pollinations TTS failed: {response.status_code}")
                break

            print(
                f"Pollinations TTS error {response.status_code}: {response.text[:300]}"
            )
            break

    return None


# --------------------------------------------------------------------------- #
# edge-tts (Microsoft Edge read aloud, no key required)
# --------------------------------------------------------------------------- #

EDGE_TTS_VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
]


def _edge_rate_string(speech_rate: Optional[float]) -> str:
    """Turn 1.1 into "+10%", the format edge-tts expects."""
    if not speech_rate:
        return "+0%"
    percent = int(round((float(speech_rate) - 1.0) * 100))
    return f"{percent:+d}%"


async def _edge_speak(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            buffer.extend(chunk["data"])
    return bytes(buffer)


def edge_tts_bytes(
    text: str,
    voice: str,
    rate: Optional[float] = None,
    pitch: Optional[str] = None,
) -> Optional[bytes]:
    """Synthesize speech with edge-tts, which needs no API key.

    This is the "always works" narration path: it only needs a network
    connection and the `edge-tts` package. It is an unofficial client for the
    Edge read aloud service, so it is used as a fallback rather than the
    primary engine.
    """
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("edge-tts is not installed; skipping it (pip install edge-tts).")
        return None

    text = (text or "").strip()
    if not text:
        return None

    config = load_config()
    e_config = config.get("edge", {})

    if rate is None:
        rate = (config.get("tts") or {}).get("speech_rate")
    if pitch is None:
        pitch = e_config.get("pitch", "+0Hz")

    retries = int(e_config.get("max_retries", 2))
    for attempt in range(retries):
        try:
            audio = asyncio.run(
                _edge_speak(text, voice, _edge_rate_string(rate), pitch)
            )
            if audio:
                print(f"edge-tts synthesized speech with {voice} ({len(audio)} bytes).")
                return audio
            print("edge-tts returned no audio data.")
        except Exception as e:
            print(f"edge-tts error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# --------------------------------------------------------------------------- #
# Speech to text helpers
# --------------------------------------------------------------------------- #

def _to_shortcap_segments(transcript: Any) -> List[Dict[str, Any]]:
    """Convert a verbose_json transcript into shortcap's segment format.

    shortcap expects a list of `{"start", "end", "words": [...]}` where every
    word starts with a space - it concatenates them into caption text.
    """
    raw_words = getattr(transcript, "words", None)
    if raw_words is None and isinstance(transcript, dict):
        raw_words = transcript.get("words")

    words: List[Dict[str, Any]] = []
    for word in raw_words or []:
        if isinstance(word, dict):
            text = word.get("word", "")
            start = word.get("start")
            end = word.get("end")
        else:
            text = getattr(word, "word", "")
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
        if not text or start is None:
            continue
        words.append({"word": " " + text.strip(), "start": start, "end": end})

    if not words:
        return []

    raw_segments = getattr(transcript, "segments", None)
    if raw_segments is None and isinstance(transcript, dict):
        raw_segments = transcript.get("segments")

    start = words[0]["start"]
    end = words[-1]["end"]
    if raw_segments:
        first = raw_segments[0]
        last = raw_segments[-1]
        start = (
            first.get("start", start)
            if isinstance(first, dict)
            else getattr(first, "start", start)
        )
        end = (
            last.get("end", end)
            if isinstance(last, dict)
            else getattr(last, "end", end)
        )

    return [{"start": start, "end": end, "words": words}]


def transcribe_audio(audio_file: str, client: Optional[OpenAI] = None) -> List[Dict[str, Any]]:
    """Transcribe an audio file with the active STT provider."""
    config = load_config()
    provider = config.get("providers", {}).get("stt", "openai")
    client = client or get_stt_client()

    if provider == "groq":
        model = config["groq"]["whisper_model"]
    else:
        model = config["openai"].get("whisper_model", "whisper-1")

    with open(audio_file, "rb") as handle:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=handle,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )
    return _to_shortcap_segments(transcript)


def transcribe_video(video_file: str, client: Optional[OpenAI] = None) -> List[Dict[str, Any]]:
    """Extract the audio track of a video and transcribe it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_file = os.path.join(tmp_dir, "audio.wav")
        extract_audio(video_file, audio_file)
        return transcribe_audio(audio_file, client=client)


# --------------------------------------------------------------------------- #
# Image providers
# --------------------------------------------------------------------------- #

def get_image_generator(only: Optional[str] = None):
    """Return a callable that produces image bytes for a prompt.

    `providers.image` may be a single provider name or an ordered list; each
    provider is tried in turn so a blocked free tier does not break the run.
    Pass `only` to pin a single provider, which is what the manual comparison
    entry point does.
    """
    from api import (
        fal_flux_api,
        felo_image_api,
        gemini_image_api,
        pollinations_image_api,
        replicate_flux_api,
    )
    from local_image import local_image_api

    config = load_config()
    chain = config.get("providers", {}).get("image", "replicate")
    if isinstance(chain, str):
        chain = [chain]

    registry = {
        "pollinations": pollinations_image_api,
        "gemini": gemini_image_api,
        "felo": felo_image_api,
        "replicate": replicate_flux_api,
        "fal": fal_flux_api,
        "local": local_image_api,
    }
    if only:
        if only not in registry:
            raise ValueError(
                f"Unknown image provider '{only}' (choose from {', '.join(registry)})"
            )
        generators = [registry[only]]
    else:
        generators = [registry[name] for name in chain if name in registry]

    def generate(prompt: str) -> Optional[bytes]:
        for generator in generators:
            try:
                data = generator(prompt)
                if data:
                    return data
            except Exception as e:
                print(f"Image provider {getattr(generator, '__name__', generator)} failed: {e}")
        return None

    return generate


if __name__ == "__main__":
    config = load_config()

    print(describe_providers())
    for name in (
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "FELO_API_KEY",
        "POLLINATIONS_API_KEY",
        "REPLICATE_API_TOKEN",
        "FAL_KEY",
        "OPENAI_API_KEY",
    ):
        print(f"{name}: {'set' if os.getenv(name) else 'missing'}")

    print("pollinations image:", config["pollinations"]["model"])
    print("pollinations tts:  ", config["pollinations_tts"]["model"])
    print("edge voices:       ", ", ".join(EDGE_TTS_VOICES))
