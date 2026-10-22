from utils import (
    call_openai_api,
    create_empty_storyboard,
    load_config,
    STORY_TYPE_HASHTAGS
)
import re
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


CHARACTER_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ethnicity": {"type": "string"},
                    "gender": {"type": "string"},
                    "age": {"type": "string"},
                    "facial_features": {"type": "string"},
                    "body_type": {"type": "string"},
                    "hair_style": {"type": "string"},
                    "accessories": {"type": "string"},
                },
                "required": [
                    "name",
                    "ethnicity",
                    "gender",
                    "age",
                    "facial_features",
                    "body_type",
                    "hair_style",
                    "accessories",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["characters"],
    "additionalProperties": False,
}

STORYBOARD_SCHEMA = {
    "type": "object",
    "properties": {
        "storyboards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_number": {"type": "integer"},
                    "description": {"type": "string"},
                    "subtitles": {"type": "string"},
                    "transition_type": {
                        "type": "string",
                        "enum": ["zoom-in", "zoom-out"],
                    },
                },
                "required": [
                    "scene_number",
                    "description",
                    "subtitles",
                    "transition_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["storyboards"],
    "additionalProperties": False,
}


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object out of a model response."""
    if not response:
        return None

    cleaned = _strip_code_fences(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
    else:
        logging.error("No JSON object found in the response")
    return None


def call_json_api(
    client, messages, schema: Dict[str, Any], name: str
) -> Optional[Dict[str, Any]]:
    """Call the model and parse a JSON object, preferring schema mode.

    Providers such as Groq support strict JSON schema output, which removes the
    need to dig JSON out of free form text. If the provider rejects it, the
    call is retried with plain prompting.
    """
    response = call_openai_api(
        client,
        messages,
        max_retries=1,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    )
    parsed = parse_json_response(response) if response else None
    if parsed is not None:
        return parsed

    print("Falling back to plain JSON parsing for this response.")
    response = call_openai_api(client, messages, max_retries=2)
    return parse_json_response(response) if response else None


def get_story_type_guidelines(story_type: str) -> str:
    story_type_guidelines = {
        "philosophy": '''
        1. Central philosophical question: Present a fundamental philosophical inquiry or dilemma.
        2. Characters: Introduce one or more characters who explore or embody different perspectives on the central question.
        3. Setting: Establish a context that enhances the philosophical exploration (e.g., a specific time period, a metaphorical landscape, or a thought experiment scenario).
        4. Dialogue or inner monologue: Use conversation or introspection to explore the philosophical concepts.
        5. Analogies or thought experiments: Incorporate illustrative examples or hypothetical scenarios to clarify complex ideas.
        6. Logical progression: Develop the philosophical argument or exploration in a clear, step-by-step manner.
        7. Open-ended conclusion: End with a thought-provoking statement or question that encourages further reflection.

        Ensure the story or dialogue is accessible to a general audience while maintaining philosophical depth and rigor.
        Aim to stimulate critical thinking and encourage readers to question their assumptions about the topic.
        ''',
        "life pro tips": '''
        Guidelines for creating the life pro tip:
        1. Practicality: Ensure the tip is applicable to everyday situations within the chosen category.
        2. Clarity: Present the tip in clear, concise language.
        3. Uniqueness: Provide a tip that is not commonly known or offers a new perspective in the chosen category.
        4. Actionable advice: Give specific steps or methods to implement the tip.
        5. Impact: Focus on a tip that can make a significant difference in people's lives within the chosen category.
        6. Explanation: Provide a detailed explanation of why the tip works and how it can be beneficial.
        7. Examples: Include one or two examples of how the tip can be applied in real-life situations.
        ''',
        "fun facts": '''
        Guidelines for creating the fun fact:
        1. Topic: Choose an interesting subject from areas such as science, history, culture, nature, or technology.
        2. Surprise factor: Focus on a fact that is likely to surprise or amaze the reader.
        3. Relevance: Select a fact that has some connection to everyday life or current events.
        4. Clarity: Present the fact in clear, concise language that's easy to understand.
        5. Depth: Provide a detailed explanation or additional context for the fact.
        6. Engagement: Frame the fact in a way that sparks curiosity and encourages further exploration.
        7. Accuracy: Ensure the fact is true and verifiable.
        ''',
        "long form jokes": '''
        Guidelines for creating the long form joke:
        1. Narrative structure: Compose a lengthy joke that incorporates an engaging narrative structure as a build-up, followed by a clever and uproarious punchline.
        2. Detailed storyline: Craft a well-developed and detailed storyline that leads up to the punchline, ensuring it is both witty and side-splittingly funny.
        3. Character development: Introduce one or more interesting characters and develop them throughout the joke.
        4. Setting: Establish a clear and vivid setting that enhances the humor of the situation.
        5. Pacing: Maintain a good rhythm throughout the joke, building tension and anticipation as you approach the punchline.
        6. Misdirection: Use clever misdirection to set up unexpected twists and surprises.
        7. Callbacks: If possible, incorporate callbacks to earlier elements of the joke for added humor.
        8. Punchline: Deliver a strong, unexpected, and hilarious punchline that ties everything together.
        9. Appropriate content: Ensure the joke is suitable for a general audience, avoiding offensive or overly controversial topics.
        ''',
        "bedtime": '''
        1. Main character: Introduce a relatable protagonist that children can easily connect with.
        2. Setting: Establish a cozy, magical, or dreamy setting appropriate for a bedtime story.
        3. Gentle plot: Present a mild adventure or problem that is resolved peacefully.
        4. Emotional journey: Develop the character's emotions in a way that promotes feelings of comfort and security.
        5. Story structure: Follow a clear beginning, middle, and end, with a calming resolution.
        6. Language and style: Use simple, soothing language suitable for children, with some repetition or rhythmic elements.
        7. Moral or lesson: Include a subtle, positive message or gentle life lesson.

        Ensure the story is calming and conducive to sleep, avoiding any scary elements or overly exciting adventures.
        The story should help children wind down and prepare for sleep.
        ''',
        "urban legends": '''
        1. Setting: Establish a specific region or culture where the legend takes place.
        2. Origins: Describe the supposed origins of the legend, including any historical context.
        3. Main narrative: Tell the core story of the urban legend in vivid detail.
        4. Variations: Mention any notable variations of the legend, if applicable.
        5. Real events: If relevant, reference any real events that might have inspired the legend.
        6. Atmospheric elements: Include details that create a sense of unease or fear.
        7. Cultural impact: Briefly discuss how the legend has affected local culture or beliefs.
        8. Ambiguity: Maintain some level of uncertainty about the truth of the legend.

        Tell the story of an urban legend from a specific region or culture. Describe the legend in detail, including its origins, variations, and any real events that might have inspired it. Your goal is to create a chilling and captivating narrative that draws viewers into the world of urban folklore.
        ''',
    }

    # Default guidelines for unspecified story types
    default_guidelines = '''
    1. Main character: Introduce a compelling protagonist with clear goals or desires.
    2. Setting: Establish a vivid and appropriate setting for the story.
    3. Conflict: Present a central conflict or challenge for the main character to overcome.
    4. Emotional journey: Develop the character's emotions throughout the story.
    5. Story structure: Follow a clear beginning (setup), middle (confrontation), and end (resolution).
    6. Language and style: Adapt the language to suit the {story_type}.

    Ensure the story captures the reader's attention and imagination while staying true to the conventions of {story_type}.
    Be creative and surprising in your storytelling approach within the boundaries of the chosen story type.
    '''

    guidelines = story_type_guidelines.get(story_type.lower(), default_guidelines)

    # Special case for 'Interesting History'
    if story_type.lower() == 'interesting history':
        guidelines += '''
        Important: Ensure the story is based on real historical events and figures.
        Provide accurate historical details while maintaining an engaging narrative.
        '''

    return guidelines.format(story_type=story_type)


def create_story_prompt(story_type: str, char_limit: Tuple[int, int]) -> str:
    base_prompt = f'''
    Create content based on the following guidelines:

    1. Title: Create an engaging and relevant title. Ensure they are unique and avoid repetition.
    2. Description: Write a 100-150 character description that provides an overview of the content. Include 2-3 relevant hashtags, followed by #facelessvideos.app as the last hashtag.
    3. Content: Generate the main content according to the specific guidelines below.

    {get_story_type_guidelines(story_type)}

    Important: Please ensure the total character count is between {char_limit[0]} and {char_limit[1]} characters.

    Format your response as follows, with no markdown code fences and no extra commentary:
    Title: [Your generated title]

    Description: [Your generated description]

    [Your generated content]
    '''
    return base_prompt


def parse_story_response(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Split a story response into title, description and body.

    Reasoning models like gpt-oss sometimes narrate their plan before the
    actual answer, so the labelled sections are located first and the plain
    blank-line split is only a fallback.
    """
    if not response:
        return None, None, None

    text = _strip_code_fences(response)

    # Reasoning models often narrate the requested format before answering,
    # which produces decoy "Title:" / "Description:" labels. The real answer
    # comes last, so try the labels from the end backwards.
    for marker in reversed(list(re.finditer(r"Title:", text, re.IGNORECASE))):
        remainder = text[marker.end():]
        match = re.match(
            r"\s*([^\n]+?)\s*\n+\s*Description:\s*([^\n]+?)\s*\n+(.+)",
            remainder,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return (
                match.group(1).strip(),
                match.group(2).strip(),
                match.group(3).strip(),
            )

    parts = text.split("\n\n", 2)
    if len(parts) == 3:
        title = parts[0].replace("Title:", "").strip()
        description = parts[1].replace("Description:", "").strip()
        content = parts[2].strip()
        if title and description and content:
            return title, description, content

    return None, None, None


def normalize_hashtags(description: str, story_type: str) -> str:
    """Put the mapped hashtag first and #facelessvideos.app last.

    The model sometimes repeats a tag or leaves an empty one behind, so the
    list is cleaned up on the way through.
    """
    first_hashtag = STORY_TYPE_HASHTAGS[story_type].lstrip("#")

    desc_parts = description.split("#", 1)
    if len(desc_parts) <= 1:
        return f"{description} #{first_hashtag} #facelessvideos.app"

    desc_text = desc_parts[0].strip()
    tags = [tag.strip() for tag in desc_parts[1].split("#")]
    tags = [tag for tag in tags if tag and tag != "facelessvideos.app"]
    ordered = [first_hashtag] + [
        tag for tag in tags if tag.lower() != first_hashtag.lower()
    ]

    seen = set()
    unique = []
    for tag in ordered:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            unique.append(tag)

    hashtags = " ".join(f"#{tag}" for tag in unique)
    return f"{desc_text} {hashtags} #facelessvideos.app"


def generate_story_and_title(client, story_type: str) -> Tuple[str, str, str]:
    config = load_config()
    char_limit = (config['story_generation']['char_limit_min'], config['story_generation']['char_limit_max'])

    prompt = create_story_prompt(story_type, char_limit)

    messages = [
        {
            "role": "system",
            "content": '''You are a versatile content creator skilled in storytelling, and creating content for various audiences. Your content is known for:
            1. Engaging and informative writing across various genres
            2. Clear and concise communication of complex ideas
            3. Adapting language and style to suit different content types and audiences
            4. Creating compelling narratives or informative lists as required
            5. Crafting catchy, relevant titles that capture the essence of the content

            You excel at following specific guidelines while maintaining creativity and relevance in your content creation.

            When creating titles:
            - Make them attractive and attention-grabbing
            - Ensure they are unique and avoid repetition
            - Capture the essence of the content while piquing curiosity'''
        },
        {"role": "user", "content": prompt},
    ]

    response = call_openai_api(client, messages)
    title, description, content = parse_story_response(response)
    if not title or not description or not content:
        return None, None, None

    # Ensure the first hashtag is from the STORY_TYPE_HASHTAGS mapping
    description = normalize_hashtags(description, story_type)

    return title, description, content


def generate_characters(client, story: str) -> List[Dict[str, str]]:
    prompt = f"""Based on the following story, create detailed descriptions for each character, including their name, ethnicity, gender, age, facial features, body type, hair style, and accessories. Focus on permanent or long-term attributes.

        Story:
        {story}

        Output format:
        {{
            "characters": [
                {{
                    "name": "Character Name",
                    "ethnicity": "Character's Ethnicity",
                    "gender": "Character's Gender",
                    "age": "Character's Age",
                    "facial_features": "Description of Character's facial features",
                    "body_type": "Description of Character's body type",
                    "hair_style": "Description of Character's hair style",
                    "accessories": "Description of Character's accessories"
                }},
                ...
            ]
        }}

        Guidelines:
        - Include the character's name as it appears in the story.
        - Specify the character's ethnicity if it's relevant and discernible from the story.
        - State the character's gender.
        - Specify the character's age or apparent age range.
        - For facial features, include details about eyes, nose, mouth, chin, forehead, cheekbones, and overall face shape. Include any notable unique features like scars, birthmarks, or facial hair.
        - Describe the character's body type, including height and build.
        - For hair style, describe the color, length, style, and texture.
        - For accessories, include only non-clothing items such as jewelry, glasses and watches that are consistently associated with the character.
        - Aim for concise but descriptive entries for each attribute.
        - Focus on permanent or long-term features, not on changeable expressions or temporary states.
        - Do not include any descriptions of clothing or attire.

        Please provide only the JSON object, without any additional text.
        """

    messages = [
        {
            "role": "system",
            "content": """You are an expert at analyzing stories and creating detailed, vivid character descriptions, focusing on overall appearance. Your skills include:
                1. Extracting subtle character details from narrative context
                2. Creating consistent and believable descriptions of characters
                3. Focusing on permanent features and distinguishing attributes
                4. Adapting descriptions to fit the story's genre and tone
                5. Balancing physical features with character essence
                6. Translating character personalities into comprehensive physical attributes
                7. Accurately estimating and describing characters' attributes based on story context
                8. Avoiding any mention of clothing or attire in character descriptions"""
        },
        {"role": "user", "content": prompt},
    ]

    parsed = call_json_api(client, messages, CHARACTER_SCHEMA, "characters")
    if parsed is None:
        print("API returned empty response")
        return []

    characters = parsed.get("characters")
    if isinstance(characters, list):
        return characters

    # Tolerate a bare array response.
    if isinstance(parsed, list):
        return parsed
    return []


def generate_storyboard(client, title: str, story: str, story_type: str, character_names: List[str] = None) -> Dict[str, Any]:
    config = load_config()
    max_scenes = config['storyboard']['max_scenes']
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    # define type-specific guidelines
    type_guidelines = {
        "general": "Focus on visually representing key moments and significant changes in the story.",
        "philosophy": "Focus on visually representing abstract philosophical concepts through concrete imagery, metaphors, or thought experiments.",
        "fun facts": "Focus on visually representing the information, statistics, or concepts mentioned in the fun fact.",
        "life pro tips": "Focus on visually representing the implementation of the tip, its benefits, and potential scenarios where it can be applied.",
    }

    # define type-specific opening scene instructions
    opening_scene_instructions = {
        "general": "A vivid description (60-70 words) that sets up an engaging question related to the overall theme of the story.",
        "philosophy": "A vivid description (60-70 words) that sets up the central philosophical question or dilemma.",
        "fun facts": "A vivid description (60-70 words) that sets up an engaging question related to the fun fact.",
        "life pro tips": "A vivid description (60-70 words) that sets up a common problem or situation related to the life pro tip.",
    }

    # define type-specific additional guidelines
    additional_guidelines = {
        "general": """
        - Describe characters' clothing in detail, ensuring consistency within scenes.
        - Cover the entire story without omitting any significant parts.
        - Use the provided character full names in the descriptions.
        """,
        "philosophy": """
        - Include scenes that show characters engaged in deep thought, dialogue, or experiencing realizations.
        - Incorporate visual elements that symbolize the philosophical themes being explored.
        """,
        "fun facts": """
        - Use visual metaphors or analogies to help explain complex ideas if necessary.
        - Include scenes that show the real-world application or implications of the fun fact, if applicable.
        """,
        "life pro tips": """
        - Use before-and-after style scenes to show the impact of applying the tip, if applicable.
        - Include scenes that show both the process of implementing the tip and its positive outcomes.
        """,
    }

    prompt = f"""Based on the following {story_type}, create a detailed storyboard with {max_scenes} scenes.

        Title: {title}
        {"Character full Names: " + ', '.join(character_names) if character_names else ""}

        First, create an opening scene:
        1. Scene Number: 1
        2. Description: {opening_scene_instructions.get(story_type, opening_scene_instructions["general"])}
        3. Subtitles: An engaging question that captures the essence of the {story_type} and piques the viewer's interest.

        Then, for each subsequent scene, provide the following details:
        1. Scene Number
        2. Description: A vivid description (60-70 words) focusing on key visual elements. {type_guidelines.get(story_type, type_guidelines["general"])}
        3. Subtitles: Use EXACT quotes from the original text.
        4. Transition: Specify the type of transition to the current scene.

        Guidelines:
        - Subtitles MUST contain only exact text from the original text, without any additions, omissions, or modifications.
        - Include every sentence from the original text in the subtitles, maintaining the correct order across all scenes.
        - Each subtitle must be unique; do not repeat content in multiple scenes.
        - For partial sentences at scene boundaries, include the fragment and continue it in the next scene's subtitles.
        - EVERY SCENE MUST HAVE NON-EMPTY SUBTITLES.
        - Return exactly {max_scenes} scenes, numbered 1 to {max_scenes}.
        - The narration is read continuously, so scene boundaries only decide where the picture changes. To reach {max_scenes}, split longer sentences across several scenes at clause boundaries - never in the middle of a clause or a word.
        - Keep each scene's subtitles to a short phrase (a few words) so the picture keeps changing.
        - Ensure that the scenes flow logically and capture the essence of the {story_type}.
        - When using a character's name in possessive form (e.g., "Character's") in the description,
          surround it with double curly braces {{{{ }}}} if it's not referring to the character's appearance.
          For example: "{{{{Giovanni's}}}} workshop" or "The mysterious figure stood at the entrance of {{{{Giovanni's}}}} laboratory."
        {additional_guidelines.get(story_type, "")}

        Use only the following options for transition details:
        - Transition types: zoom-in, zoom-out

        Guidelines for using zoom-in, and zoom-out transitions:
        - Zoom-in: Use to focus on important details, build tension, or show a character's point of view. Examples: revealing a clue, emphasizing a character's reaction, or creating suspense.
        - Zoom-out: Use to reveal context, end a scene, or show isolation. Examples: showing a character in a larger environment, concluding a sequence, or transitioning from a detail to a wider view.

        Important rules:
        1. Do not use zoom-in transitions when the current scene's shot size is close-up or extreme close-up.
        2. For transitions, use ONLY the following types:
            - zoom-in
            - zoom-out
        3. DO NOT use any other transition types, including fade, dissolve, or cut.

        Output the result as a JSON object with the following structure:
        {{
            "storyboards": [
                {{
                    "scene_number": 1,
                    "description": "Scene Description",
                    "subtitles": "Subtitles or Dialogue",
                    "transition_type": "zoom-in"
                }},
                ...
            ]
        }}

        Field rules:
        - scene_number must be a whole number (1, 2, 3, ...), never text.
        - transition_type must be exactly "zoom-in" or "zoom-out".
        - description and subtitles must be plain strings.

        Here's the story:

    {story}"""

    messages = [
        {
            "role": "system",
            "content": '''You are a highly skilled storyboard artist specializing in visualizing various types of content. You excel at:
                1. Creating vivid, engaging scene descriptions that translate different types of content into compelling visuals
                2. Developing visual metaphors and analogies to represent complex ideas or concepts
                3. Crafting scenes that show the real-world application or implications of the information
                4. Incorporating basic cinematographic techniques to enhance visual storytelling
                5. Faithfully representing the original text using exact quotes for subtitles
                6. Ensuring the visual narrative accurately captures key points and their development
                7. Balancing informative content with visually interesting and engaging scenes
                8. Maintaining logical consistency between scenes while providing a variety of visual representations

                Your storyboards effectively bridge the gap between various types of content and visual representation,
                paying close attention to both the overall flow of information and specific details that enhance
                the communication of the content.'''
        },
        {"role": "user", "content": prompt},
    ]

    parsed = call_json_api(client, messages, STORYBOARD_SCHEMA, "storyboard")
    if parsed is None:
        print("API returned empty response")
        return create_empty_storyboard(title)

    storyboards = parsed.get("storyboards")
    if not isinstance(storyboards, list):
        print("No storyboards found in the response")
        return create_empty_storyboard(title)

    result = {
        "project_info": {
            "title": title,
            "user": "AI Generated",
            "timestamp": timestamp,
        },
        "storyboards": storyboards,
    }
    return result


def generate_general_storyboard(client, title: str, story: str, character_names: List[str]) -> Dict[str, Any]:
    return generate_storyboard(client, title, story, "general", character_names)

def generate_philosophy_storyboard(client, title: str, story: str, character_names: List[str]) -> Dict[str, Any]:
    return generate_storyboard(client, title, story, "philosophy", character_names)

def generate_fun_facts_storyboard(client, title: str, story: str) -> Dict[str, Any]:
    return generate_storyboard(client, title, story, "fun facts")

def generate_life_pro_tips_storyboard(client, title: str, story: str) -> Dict[str, Any]:
    return generate_storyboard(client, title, story, "life pro tips")


if __name__ == "__main__":
    from providers import get_llm_client

    client = get_llm_client()

    title, description, story = generate_story_and_title(client, "Mystery")
    print(title)
    print(description)
    print(story)

    characters = generate_characters(client, story)
    project = generate_general_storyboard(
        client, title, story, [character["name"] for character in characters]
    )
    print(project)
