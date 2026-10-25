from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from styles import apply_palette, style_prompt
from utils import create_blank_image, load_config, normalize_image_file


def generate_image(
    storyboard: Dict[str, Any],
    characters: List[Dict[str, Any]],
    style: str,
    image_generator_func: Callable[[str], Optional[bytes]]
) -> Optional[bytes]:
    raw_prompt = storyboard.get('description', '')
    clean_scene = re.sub(r'\{\{.*?\}\}', '', raw_prompt).strip()

    # Identify characters explicitly or implicitly referenced in this scene
    active_characters = []
    for character in characters:
        name_forms = [
            character['name'].split()[0].lower(),
            character['name'].lower(),
            f"{character['name'].split()[0].lower()}'s",
            f"{character['name'].lower()}'s",
        ]
        if any(
            form in clean_scene.lower()
            for form in name_forms
        ):
            active_characters.append(character)

    # If the story has only 1 main character and pronouns like "he/she" appear, anchor that character
    if not active_characters and len(characters) == 1:
        words = set(clean_scene.lower().split())
        if words.intersection({"he", "him", "his", "she", "her", "man", "woman", "person", "character"}):
            active_characters.append(characters[0])

    # Construct character visual continuity tokens
    character_blocks = []
    for character in active_characters:
        if character.get("visual_anchor"):
            character_blocks.append(f"Character {character['name']}: {character['visual_anchor']}")
        else:
            outfit = character.get("signature_outfit", "")
            outfit_desc = f", wearing {outfit}" if outfit else ""
            character_blocks.append(
                f"Character {character['name']}: {character.get('ethnicity', '')} {character.get('gender', '')} "
                f"{character.get('age', '')}, face: {character.get('facial_features', '')}, "
                f"hair: {character.get('hair_style', '')}, physique: {character.get('body_type', '')}"
                f"{outfit_desc}"
            )

    # Prompt ordering prioritized for diffusion text encoders:
    # 1. Art style anchor (medium, linework, shading, lighting)
    # 2. Character visual anchors (locks facial identity, hair, skin, outfit)
    # 3. Specific scene action, props, and background setting
    # 4. Consistency reinforcement tokens
    prompt_segments = [style_prompt(style)]

    if character_blocks:
        prompt_segments.append(" | ".join(character_blocks))

    prompt_segments.append(f"Action and setting: {clean_scene}")
    prompt_segments.append(
        "consistent character design, identical face across scenes, same signature outfit, "
        "crisp clean outlines, 2D animation still frame, coherent art direction"
    )

    final_prompt = " | ".join(prompt_segments)
    return image_generator_func(final_prompt)


def generate_and_download_images(
    storyboard_project: Dict[str, Any],
    story_dir: str,
    image_style: str,
    image_generator_func: Callable[[str], Optional[bytes]],
    max_workers: Optional[int] = None,
) -> List[str]:
    start_time = time.time()

    image_files = []
    characters = storyboard_project.get('characters', [])
    storyboards = storyboard_project.get('storyboards', [])

    if not storyboards:
        return []

    # Determine worker count: CPU local model shouldn't be oversubscribed.
    if max_workers is None:
        try:
            cfg = load_config()
            primary_img = cfg.get("providers", {}).get("image", "replicate")
            if isinstance(primary_img, list):
                primary_img = primary_img[0] if primary_img else "replicate"
            max_workers = 1 if primary_img == "local" else 4
        except Exception:
            max_workers = 4

    raw_images: Dict[int, Optional[bytes]] = {}

    if max_workers > 1 and len(storyboards) > 1:
        print(f"Generating {len(storyboards)} scene images with {max_workers} concurrent workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    generate_image, sb, characters, image_style, image_generator_func
                ): i
                for i, sb in enumerate(storyboards)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    raw_images[index] = future.result()
                except Exception as e:
                    print(f"Error generating scene {storyboards[index].get('scene_number', index + 1)}: {e}")
                    raw_images[index] = None
    else:
        for i, storyboard in enumerate(storyboards):
            raw_images[i] = generate_image(storyboard, characters, image_style, image_generator_func)

    for i, storyboard in enumerate(storyboards):
        image_content = raw_images.get(i)
        image_filename = os.path.join(story_dir, f"scene_{storyboard['scene_number']}.png")
        if image_content:
            storyboard['image'] = image_filename
            try:
                with open(image_filename, 'wb') as f:
                    f.write(image_content)
                # Providers return different resolutions (and often JPEG), while
                # moviepy needs every clip to have identical dimensions.
                if not normalize_image_file(image_filename):
                    print(
                        f"Could not normalize image for scene {storyboard['scene_number']}"
                    )
                # Lock the palette in: diffusion output drifts between shots,
                # the grade is what makes 24 images read as one video.
                apply_palette(image_filename, image_style)
                image_files.append(image_filename)
                print(f"Image saved for scene {storyboard['scene_number']}")
            except IOError as e:
                print(f"Failed to save image for scene {storyboard['scene_number']}: {e}")
                if i > 0 and image_files:
                    storyboard['image'] = image_files[-1]
                else:
                    create_blank_image(image_filename)
                    storyboard['image'] = image_filename
                    image_files.append(image_filename)
        else:
            print(f"Failed to generate image for scene {storyboard['scene_number']}")
            if i > 0 and image_files:
                storyboard['image'] = image_files[-1]
            else:
                create_blank_image(image_filename)
                storyboard['image'] = image_filename
                image_files.append(image_filename)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"generate_and_download_images execution time: {execution_time:.2f} seconds")

    return image_files


if __name__ == "__main__":
    from providers import get_image_generator

    prompt = (
        "a crumbling gothic hall at night, moonlight through a tall arched "
        "window, dust motes in the air, wide cinematic shot"
    )

    os.makedirs("data/_test/images", exist_ok=True)
    for provider in ("pollinations", "gemini", "local"):
        data = get_image_generator(only=provider)(f"{prompt} | {style_prompt('storyboard')}")
        if not data:
            print(provider, "failed")
            continue

        path = f"data/_test/images/{provider}.png"
        with open(path, "wb") as handle:
            handle.write(data)
        normalize_image_file(path)
        apply_palette(path, "storyboard")
        print(provider, "->", path)
