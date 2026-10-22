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
    deep_merge,
    get_preset,
    get_video_config,
    get_voice_list,
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
from narration import run_narration
from video_creator import create_video

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(script_dir), ".env"))


if __name__ == "__main__":
    preset = "youtube"
    story_type = "Mystery"
    image_style = "cinematic"
    max_scenes = 24
    no_images = False

    set_config_overrides(
        deep_merge(get_preset(preset), {"storyboard": {"max_scenes": max_scenes}})
    )

    print(f"Active providers: {describe_providers()}")
    video = get_video_config()
    print(
        f"Output: {video['width']}x{video['height']} @ {video['fps']}fps, "
        f"up to {max_scenes} scenes"
    )

    voice_name = get_voice_list()[0]
    client = get_llm_client()
    tts_client = get_tts_client()
    audio_extension = get_audio_extension()

    title, description, story = generate_story_and_title(client, story_type)
    if story is None or title is None:
        raise SystemExit("Failed to generate a story and title.")

    story_dir = create_resource_dir(script_dir, story_type, title)

    characters = []
    if story_type.lower() not in ("life pro tips", "fun facts"):
        print("\nGenerating character descriptions...")
        characters = generate_characters(client, story)
        if not characters:
            raise SystemExit("Failed to generate characters.")
        print("characters:", characters)

    character_names = (
        [character["name"] for character in characters] if characters else []
    )

    print("\nGenerating storyboard...")
    if story_type.lower() == "life pro tips":
        storyboard_project = generate_life_pro_tips_storyboard(
            client, title, story
        )
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

    if not storyboard_project.get("storyboards"):
        raise SystemExit("Failed to generate storyboard.")

    storyboard_project["storyboards"] = [
        scene
        for scene in storyboard_project["storyboards"]
        if scene["subtitles"].strip()
    ]
    storyboard_project["characters"] = characters

    with open(
        os.path.join(story_dir, "story_english.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.write(f"{title}\n\n{description}\n\n{story}")

    audio_dir = os.path.join(story_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print("\nSynthesizing narration...")
    kept_storyboards, timeline, narration = run_narration(
        storyboard_project["storyboards"],
        story,
        tts_client,
        voice_name,
        audio_dir,
        audio_extension,
    )

    if timeline and narration:
        storyboard_project["storyboards"] = kept_storyboards
    else:
        timeline = None
        for storyboard in storyboard_project["storyboards"]:
            storyboard["audio"] = os.path.join(
                audio_dir,
                f"scene_{storyboard['scene_number']}.{audio_extension}",
            )

    storyboard_project["narration"] = narration

    print("\nGenerating images for each scene...")
    if no_images:
        print("Image generation skipped.")
        image_generator = lambda prompt: None  # noqa: E731
    else:
        image_generator = get_image_generator()

    image_files = generate_and_download_images(
        storyboard_project, story_dir, image_style, image_generator
    )

    for index, storyboard in enumerate(storyboard_project["storyboards"]):
        if index < len(image_files):
            storyboard["image"] = image_files[index]

    print("\nSaving storyboard project...")
    with open(
        os.path.join(story_dir, "storyboard_project.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(storyboard_project, handle, ensure_ascii=False, indent=4)

    if not image_files:
        raise SystemExit("No images were generated; cannot create the video.")

    print("\nCreating video from images...")
    video_path = os.path.join(story_dir, "story_video.mp4")
    create_video(
        tts_client,
        storyboard_project,
        video_path,
        audio_dir,
        voice_name,
        timeline=timeline,
        narration_track=narration.get("track") if narration else None,
    )
    print(f"Video created: {video_path}")
