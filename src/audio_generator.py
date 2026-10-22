import os
import tempfile
import time
from typing import List, Optional

from utils import load_config, retry_delay
from providers import (
    chunk_text,
    concat_audio_files,
    edge_tts_bytes,
    gemini_tts_bytes,
    pollinations_tts_bytes,
)


def get_tts_chain(config=None) -> List[str]:
    """TTS providers to try, in order.

    `providers.tts` is always tried first so a `--tts-provider` override wins,
    then `providers.tts_fallbacks` supplies the degradation path.
    """
    config = config or load_config()
    providers = config.get("providers", {})
    primary = providers.get("tts", "openai")
    fallbacks = providers.get("tts_fallbacks") or []
    chain = [primary] + [name for name in fallbacks if name != primary]
    return chain


def generate_audio(client, text, output_file, voice_name) -> bool:
    """Synthesize narration, degrading to the next provider when one fails."""
    config = load_config()
    chain = get_tts_chain(config)

    for position, provider in enumerate(chain):
        try:
            succeeded = _synthesize_with(
                client, text, output_file, voice_name, provider, config
            )
        except Exception as e:
            print(f"TTS provider '{provider}' raised: {e}")
            succeeded = False

        if succeeded and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return True

        if position < len(chain) - 1:
            print(f"TTS provider '{provider}' produced no audio; trying '{chain[position + 1]}'.")

    return False


def _synthesize_with(client, text, output_file, voice_name, provider, config,
                     model: Optional[str] = None) -> bool:
    if provider == "gemini":
        return _generate_audio_gemini(text, output_file, voice_name, config, model)
    if provider == "pollinations":
        return _generate_audio_pollinations(text, output_file, voice_name, config, model)
    if provider == "edge":
        return _generate_audio_edge(text, output_file, voice_name, config)
    if provider == "groq":
        return _generate_audio_groq(client, text, output_file, voice_name, config, model=model)
    return _generate_audio_openai(client, text, output_file, voice_name, config)


def _save_bytes(audio: bytes, output_file: str) -> bool:
    try:
        with open(output_file, "wb") as handle:
            handle.write(audio)
        return True
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return False


def _generate_audio_pollinations(text, output_file, voice_name, config,
                                 model: Optional[str] = None) -> bool:
    """Pollinations TTS: free tier models, no daily character cap."""
    if not model:
        model = config.get("pollinations_tts", {}).get("model")
    try:
        audio = pollinations_tts_bytes(text, voice_name, model=model)
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

    if not audio or not _save_bytes(audio, output_file):
        if not audio:
            print("Error generating audio: Pollinations returned no audio data.")
        return False

    print(
        f"Speech synthesized for text [{text[:80]}], and the audio was saved to [{output_file}]"
    )
    return True


def _generate_audio_edge(text, output_file, voice_name, config) -> bool:
    """edge-tts: no API key, used as the always-available fallback."""
    try:
        audio = edge_tts_bytes(text, voice_name)
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

    if not audio or not _save_bytes(audio, output_file):
        if not audio:
            print("Error generating audio: edge-tts returned no audio data.")
        return False

    print(
        f"Speech synthesized for text [{text[:80]}], and the audio was saved to [{output_file}]"
    )
    return True


def _generate_audio_gemini(text, output_file, voice_name, config,
                           model: Optional[str] = None) -> bool:
    """Gemini TTS.

    The Gemini TTS models are available on the free tier, and there is no
    length limit comparable to Groq's Orpheus, so the text is sent as one
    request.
    """
    try:
        audio = gemini_tts_bytes(text, voice_name, model=model)
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

    if not audio:
        print("Error generating audio: Gemini returned no audio data.")
        return False

    try:
        with open(output_file, "wb") as handle:
            handle.write(audio)
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return False

    print(f"Speech synthesized for text [{text[:80]}], and the audio was saved to [{output_file}]")
    return True


def _generate_audio_openai(client, text, output_file, voice_name, config) -> bool:
    speech_rate = config["tts"]["speech_rate"]
    openai_config = config.get("openai", {})
    model = openai_config.get("tts_model", "tts-1")
    audio_format = openai_config.get("tts_format", "mp3")

    try:
        result = client.audio.speech.create(
            model=model,
            voice=voice_name,
            input=text,
            speed=speech_rate,
            response_format=audio_format,
        )

        with open(output_file, "wb") as audio_file:
            audio_file.write(result.content)

        print(f"Speech synthesized for text [{text[:80]}], and the audio was saved to [{output_file}]")
        return True

    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False


def _generate_audio_groq(client, text, output_file, voice_name, config,
                         max_retries: int = 3,
                         model: Optional[str] = None) -> bool:
    """Groq (Orpheus) TTS.

    Orpheus only emits wav and has no `speed` parameter, so the configured
    speech rate is applied as a tempo change on the merged audio. Groq's token
    budget refills continuously, so a rate limited chunk is retried after the
    header driven delay instead of being dropped.
    """
    groq_config = config["groq"]
    model = model or groq_config["tts_model"]
    max_chars = groq_config.get("tts_max_chars", 420)
    speech_rate = config.get("tts", {}).get("speech_rate")

    chunks = chunk_text(text, max_chars)
    if not chunks:
        print("Nothing to synthesize for this narration chunk.")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = []
        for index, chunk in enumerate(chunks, 1):
            content = None
            for attempt in range(max_retries):
                try:
                    result = client.audio.speech.create(
                        model=model,
                        voice=voice_name,
                        input=chunk,
                        response_format=groq_config.get("tts_format", "wav"),
                    )
                    content = result.content
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = retry_delay(e, attempt)
                        print(
                            f"Groq TTS rate limited on chunk {index}/{len(chunks)}; "
                            f"waiting {delay:.0f}s..."
                        )
                        time.sleep(delay)
                    else:
                        print(f"Groq TTS failed on chunk {index}/{len(chunks)}: {e}")

            if not content:
                return False

            chunk_path = os.path.join(temp_dir, f"chunk_{index}.wav")
            with open(chunk_path, "wb") as handle:
                handle.write(content)
            chunk_files.append(chunk_path)

        tempo = None
        if speech_rate and abs(float(speech_rate) - 1.0) > 0.01:
            tempo = float(speech_rate)

        try:
            concat_audio_files(chunk_files, output_file, tempo=tempo)
        except Exception as e:
            print(f"Failed to merge audio chunks: {e}")
            return False

    print(
        f"Speech synthesized from {len(chunks)} chunk(s) [{text[:60]}], saved to [{output_file}]"
    )
    return True


if __name__ == "__main__":
    from providers import _build_client, get_audio_extension

    sentence = (
        'Eliot Finch, the archivist, had one rule: never trust a locked door. '
        '"It\'s only a journal," he whispered - but the pages were warm. '
        "At 3:17 a.m., the clock stopped."
    )

    voices = {
        "groq": "autumn",
        "pollinations": "en-US-Olivia",
        "edge": "en-US-AriaNeural",
        "gemini": "Kore",
    }
    config = load_config()
    models = {
        "groq": config["groq"]["tts_model"],
        "pollinations": config["pollinations_tts"]["model"],
        "gemini": config["gemini"]["tts_model"],
    }

    os.makedirs("data/_test/tts", exist_ok=True)
    for provider, voice in voices.items():
        client = _build_client(provider) if provider == "groq" else None
        extension = get_audio_extension(provider)
        output_file = f"data/_test/tts/{provider}.{extension}"
        print(provider, voice, "->", output_file)
        _synthesize_with(
            client,
            sentence,
            output_file,
            voice,
            provider,
            config,
            model=models.get(provider),
        )
