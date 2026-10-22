import argparse
import json
import os

from dotenv import load_dotenv

from providers import (
    describe_providers,
    get_audio_extension,
    get_image_generator,
    get_llm_client,
    get_tts_client,
)
from utils import (
    create_resource_dir,
    load_config,
    pick_image_style,
    pick_story_type,
    pick_voice_name,
    set_config_overrides,
)
from story_generator import (
    generate_story_and_title,
    generate_general_storyboard,
    generate_life_pro_tips_storyboard,
    generate_characters,
    generate_philosophy_storyboard,
    generate_fun_facts_storyboard,
)
from image_generator import generate_and_download_images
from video_creator import create_video

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(os.path.dirname(script_dir), ".env")
# Load the .env file
load_dotenv(dotenv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a faceless video from a story."
    )
    parser.add_argument(
        "--scenes", type=int, default=None,
        help="Override the maximum number of storyboard scenes."
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip image generation and use the blank image fallback."
    )
    parser.add_argument("--llm-provider", default=None, help="Override the LLM provider.")
    parser.add_argument("--llm-model", default=None, help="Override the LLM model id.")
    parser.add_argument("--tts-provider", default=None, help="Override the TTS provider.")
    parser.add_argument("--stt-provider", default=None, help="Override the STT provider.")
    parser.add_argument(
        "--image-provider", default=None,
        help="Comma separated image provider chain, e.g. 'gemini,replicate'."
    )
    parser.add_argument("--story-type", default=None, help="Skip the story type prompt.")
    parser.add_argument("--image-style", default=None, help="Skip the image style prompt.")
    parser.add_argument("--voice", default=None, help="Skip the voice prompt.")
    args, _ = parser.parse_known_args()
    return args


def apply_overrides(args):
    """Turn CLI flags into runtime config overrides."""
    overrides = {}
    provider_overrides = {}

    if args.llm_provider:
        provider_overrides["llm"] = args.llm_provider
    if args.llm_model:
        # The model lives in the provider specific block (groq / openai).
        llm_provider = args.llm_provider or load_config().get(
            "providers", {}
        ).get("llm", "openai")
        overrides.setdefault(llm_provider, {})["model"] = args.llm_model
    if args.tts_provider:
        provider_overrides["tts"] = args.tts_provider
    if args.stt_provider:
        provider_overrides["stt"] = args.stt_provider
    if args.image_provider:
        chain = [name.strip() for name in args.image_provider.split(",") if name.strip()]
        if chain:
            provider_overrides["image"] = chain
    if provider_overrides:
        overrides["providers"] = provider_overrides

    if args.scenes:
        overrides.setdefault("storyboard", {})["max_scenes"] = args.scenes

    if overrides:
        set_config_overrides(overrides)
    return overrides


def main():
    args = parse_args()
    apply_overrides(args)
    config = load_config()

    print(f"Active providers: {describe_providers()}")

    # 1. pick story type, image style and voice name
    story_type = args.story_type or pick_story_type()
    image_style = args.image_style or pick_image_style()
    voice_name = args.voice or pick_voice_name()

    client = get_llm_client()
    tts_client = get_tts_client()
    audio_extension = get_audio_extension()

    # 2. generate story and title
    title, description, story = generate_story_and_title(client, story_type)
    if story is None or title is None:
        print("Failed to generate a story and title. Please try again later.")
        return

    story_dir = create_resource_dir(script_dir, story_type, title)

    # 3. generate characters
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

    # 4. generate storyboard
    print("\nGenerating storyboard...")
    if story_type.lower() == "life pro tips":
        storyboard_project = generate_life_pro_tips_storyboard(
            client, title, story)
    elif story_type.lower() == "philosophy":
        storyboard_project = generate_philosophy_storyboard(
            client, title, story, character_names
        )
    elif story_type.lower() == "fun facts":
        storyboard_project = generate_fun_facts_storyboard(
            client, title, story)
    else:
        storyboard_project = generate_general_storyboard(
            client, title, story, character_names
        )

    if len(storyboard_project.get("storyboards")) == 0:
        print("Failed to generate storyboard. Please try again later.")
        return

    # Remove scenes with empty subtitles
    storyboard_project["storyboards"] = [
        scene
        for scene in storyboard_project["storyboards"]
        if scene["subtitles"].strip()
    ]

    storyboard_project["characters"] = characters

    # 5. save the story and storyboard to files
    with open(
        os.path.join(story_dir, "story_english.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(f"{title}\n\n{description}\n\n{story}")

    # 6. generate images
    print("\nGenerating images for each scene...")
    if args.no_images:
        print("Image generation skipped (--no-images).")
        image_generator = lambda prompt: None  # noqa: E731
    else:
        image_generator = get_image_generator()

    image_files = generate_and_download_images(
        storyboard_project,
        story_dir,
        image_style,
        image_generator,
    )

    # Update storyboard_project with image and audio paths
    for i, storyboard in enumerate(storyboard_project['storyboards']):
        # image_files can be shorter than the scene list (for example when image
        # generation was skipped); in that case keep the fallback the image
        # generator already assigned.
        if i < len(image_files):
            storyboard['image'] = image_files[i]
        storyboard['audio'] = os.path.join(
            story_dir, "audio", f"scene_{storyboard['scene_number']}.{audio_extension}"
        )

    # Save the storyboard_project to a json file
    print("\nSaving storyboard project...")
    with open(
        os.path.join(story_dir, "storyboard_project.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(storyboard_project, f, ensure_ascii=False, indent=4)

    # 7. create video
    if image_files:
        print("\nCreating video from images...")
        audio_dir = os.path.join(story_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        video_path = os.path.join(story_dir, "story_video.mp4")
        create_video(tts_client, storyboard_project, video_path, audio_dir, voice_name)
        print(f"Video created: {video_path}")
    else:
        print("No images were generated. Cannot create video.")


if __name__ == "__main__":
    main()
