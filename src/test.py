import argparse
import json
import os
import tempfile

from providers import (
    describe_providers,
    get_image_generator,
    get_llm_client,
    get_stt_client,
    get_tts_client,
    transcribe_audio,
)
from utils import (
    load_config,
    pick_image_style,
    pick_story_type,
    pick_voice_name,
)
from story_generator import (
    generate_general_storyboard,
    generate_life_pro_tips_storyboard,
    generate_characters,
    generate_philosophy_storyboard,
    generate_fun_facts_storyboard,
)
from utils import call_openai_api


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def smoke_test(with_image: bool = False):
    """Exercise every provider leg individually."""
    print(f"Active providers: {describe_providers()}\n")
    failures = []

    # 1. LLM -----------------------------------------------------------------
    print("[1/4] LLM ...")
    try:
        client = get_llm_client()
        response = call_openai_api(
            client,
            [{"role": "user", "content": "Reply with exactly: llm ok"}],
            max_retries=2,
        )
        if response:
            print(f"      LLM OK -> {response.strip()[:80]}")
        else:
            failures.append("LLM")
            print("      LLM FAILED")
    except Exception as e:
        failures.append("LLM")
        print(f"      LLM FAILED -> {e}")

    # 2. TTS -----------------------------------------------------------------
    print("[2/4] TTS ...")
    audio_path = None
    try:
        tts_client = get_tts_client()
        config = load_config()
        provider = config.get("providers", {}).get("tts", "openai")
        voice = config.get(provider, {}).get("tts_voices", ["alloy"])[0]
        extension = "wav" if config.get(provider, {}).get("tts_format") == "wav" else "mp3"
        audio_path = os.path.join(
            tempfile.gettempdir(), f"fvg_smoke.{extension}"
        )
        from audio_generator import generate_audio

        ok = generate_audio(
            tts_client,
            "A lighthouse keeper hears a bell that no one else can hear.",
            audio_path,
            voice,
        )
        if ok and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"      TTS OK -> {audio_path} ({os.path.getsize(audio_path)} bytes)")
        else:
            failures.append("TTS")
            print("      TTS FAILED")
    except Exception as e:
        failures.append("TTS")
        print(f"      TTS FAILED -> {e}")

    # 3. STT -----------------------------------------------------------------
    print("[3/4] STT ...")
    try:
        if audio_path and os.path.exists(audio_path):
            segments = transcribe_audio(audio_path)
            word_count = len(segments[0]["words"]) if segments else 0
            if word_count:
                print(f"      STT OK -> {word_count} words")
            else:
                failures.append("STT")
                print("      STT returned no words")
        else:
            failures.append("STT")
            print("      STT SKIPPED (no audio produced by the TTS step)")
    except Exception as e:
        failures.append("STT")
        print(f"      STT FAILED -> {e}")

    # 4. Image ---------------------------------------------------------------
    if with_image:
        print("[4/4] Image ...")
        try:
            generator = get_image_generator()
            data = generator("A lone lighthouse at night, cinematic, vertical")
            if data:
                image_path = os.path.join(
                    tempfile.gettempdir(), "fvg_smoke.png"
                )
                with open(image_path, "wb") as handle:
                    handle.write(data)
                print(f"      Image OK -> {image_path} ({len(data)} bytes)")
            else:
                failures.append("Image")
                chain = describe_providers()["image"]
                print(f"      Image FAILED - none of {chain} produced an image.")
                print(
                    "      Each provider printed its reason above: a missing API "
                    "key, an exhausted quota, or billing that is not enabled."
                )
        except Exception as e:
            failures.append("Image")
            print(f"      Image FAILED -> {e}")
    else:
        print("[4/4] Image ... skipped (use --with-image)")

    print()
    if failures:
        print(f"SMOKE TEST FAILED for: {', '.join(failures)}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


# --------------------------------------------------------------------------- #
# Partial pipeline runs
# --------------------------------------------------------------------------- #

def test_text_to_image(script_dir):
    """Regenerate images for an existing story."""
    story_type = pick_story_type()
    image_style = pick_image_style()

    # story dir is in the data/Samples
    story_dir = os.path.join(script_dir, "../data/Images")

    title, story = None, None
    # read the story from file
    with open(
        os.path.join(story_dir, "story_english.txt"), "r", encoding="utf-8"
    ) as f:
        content = f.read()
    # The file is written as title / description / story
    parts = content.split("\n\n", 2)
    if len(parts) == 3:
        title, _description, story = parts
    else:
        title, story = parts[0], parts[1]

    client = get_llm_client()

    characters = []
    if story_type.lower() != "life pro tips" and story_type.lower() != "fun facts":
        print("\nGenerating character descriptions...")
        characters = generate_characters(client, story)
        if not characters:
            print("Failed to generate characters. Please try again later.")
            return
        print("characters: ", characters)

    character_names = (
        [character["name"] for character in characters] if characters else []
    )

    # generate storyboard
    print("\nGenerating storyboard...")
    if story_type.lower() == "life pro tips":
        storyboard_project = generate_life_pro_tips_storyboard(client, title, story)
    elif story_type.lower() == "philosophy":
        storyboard_project = generate_philosophy_storyboard(
            client, title, story, character_names
        )
    elif story_type.lower() == "fun facts":
        storyboard_project = generate_fun_facts_storyboard(client, title, story)
    else:
        storyboard_project = generate_general_storyboard(
            client, title, story, character_names
        )

    if len(storyboard_project.get("storyboards")) == 0:
        print("Failed to generate storyboard. Please try again later.")
        return
    storyboard_project["characters"] = characters

    # generate images and download images
    from image_generator import generate_and_download_images

    image_files = generate_and_download_images(
        storyboard_project, story_dir, image_style, get_image_generator()
    )
    print(image_files)

    # Save the storyboard_project to a json file
    print("\nSaving storyboard project...")
    with open(
        os.path.join(story_dir, "storyboard_project.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(storyboard_project, f, ensure_ascii=False, indent=4)


def test_video_from_images(script_dir, story_dir):
    """Rebuild the video from an existing storyboard project."""
    voice_name = pick_voice_name()
    # Set up the directory paths
    audio_dir = os.path.join(story_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    video_path = os.path.join(story_dir, "story_video.mp4")

    # Load the storyboard project
    with open(
        os.path.join(story_dir, "storyboard_project.json"), "r", encoding="utf-8"
    ) as f:
        storyboard_project = json.load(f)

    print("\nCreating video from images...")
    create_video(
        get_tts_client(), storyboard_project, video_path, audio_dir, voice_name
    )

    print(f"Video created: {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Faceless video generator helpers.")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a provider smoke test."
    )
    parser.add_argument(
        "--with-image", action="store_true",
        help="Include an image generation attempt in the smoke test.",
    )
    parser.add_argument(
        "--text-to-image", action="store_true",
        help="Regenerate images for an existing story.",
    )
    parser.add_argument(
        "--story-dir", default=None,
        help="Directory of an existing storyboard project (used with --video).",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Rebuild the video from an existing storyboard project.",
    )
    args, _ = parser.parse_known_args()

    if args.smoke:
        raise SystemExit(smoke_test(with_image=args.with_image))

    if args.text_to_image:
        test_text_to_image(SCRIPT_DIR)

    if args.video:
        story_dir = args.story_dir or os.path.join(SCRIPT_DIR, "../data/Images")
        test_video_from_images(SCRIPT_DIR, os.path.abspath(story_dir))


if __name__ == "__main__":
    main()
