import os
import tempfile
from typing import Optional

from utils import load_config
from providers import chunk_text, concat_audio_files, gemini_tts_bytes


def generate_audio(client, text, output_file, voice_name) -> bool:
    """Synthesize narration for one scene using the active TTS provider."""
    config = load_config()
    provider = config.get("providers", {}).get("tts", "openai")

    if provider == "gemini":
        return _generate_audio_gemini(text, output_file, voice_name, config)
    if provider == "groq":
        return _generate_audio_groq(client, text, output_file, voice_name, config)
    return _generate_audio_openai(client, text, output_file, voice_name, config)


def _generate_audio_gemini(text, output_file, voice_name, config) -> bool:
    """Gemini TTS.

    The Gemini TTS models are available on the free tier, and there is no
    length limit comparable to Groq's Orpheus, so the text is sent as one
    request.
    """
    try:
        audio = gemini_tts_bytes(text, voice_name)
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

    print(f"Speech synthesized for text [{text}], and the audio was saved to [{output_file}]")
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

        print(f"Speech synthesized for text [{text}], and the audio was saved to [{output_file}]")
        return True

    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False


def _generate_audio_groq(client, text, output_file, voice_name, config) -> bool:
    """Groq (Orpheus) TTS.

    Orpheus only accepts short inputs and only emits wav, so long subtitles are
    split into chunks that are synthesized separately and merged afterwards.
    It also has no `speed` parameter, so the configured speech rate is applied
    as a tempo change on the merged audio.
    """
    groq_config = config["groq"]
    model = groq_config["tts_model"]
    max_chars = groq_config.get("tts_max_chars", 200)
    speech_rate = config.get("tts", {}).get("speech_rate")

    chunks = chunk_text(text, max_chars)
    if not chunks:
        print("Nothing to synthesize for this scene.")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_files = []
        for index, chunk in enumerate(chunks, 1):
            try:
                result = client.audio.speech.create(
                    model=model,
                    voice=voice_name,
                    input=chunk,
                    response_format=groq_config.get("tts_format", "wav"),
                )
            except Exception as e:
                print(f"Groq TTS failed on chunk {index}/{len(chunks)}: {e}")
                return False

            chunk_path = os.path.join(temp_dir, f"chunk_{index}.wav")
            with open(chunk_path, "wb") as handle:
                handle.write(result.content)
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
        f"Speech synthesized from {len(chunks)} chunk(s), saved to [{output_file}]"
    )
    return True
