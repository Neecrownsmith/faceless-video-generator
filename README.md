# Faceless Video Generator 

---
> ## 🎬 Special Offer from FacelessVideos.app!
> 
> ### Create Professional Faceless Videos with AI - In One Click!
>
> ✨ **One-stop automated video creation platform:**
> - 🤖 Generate complete faceless videos from just text input
> - 🎨 Choose between Flux Schnell and Flux Dev AI models
> - 🎁 **Limited Time**: New users get 1000 FREE credits!
>
> [🚀 Start Creating Now →](https://facelessvideos.app/)
---

## Project Overview

Faceless Video Generator is a comprehensive multimedia content creation tool that streamlines the process of story generation, image generation, and video production. Users can generate various types of stories, create corresponding images, and compile these images and subtitles into a video.

## Demo

![Demo](demo/demo.gif)

## Key Features

- **Story Generation**: Create complete stories based on user-specified story types, including Scary, Mystery, Bedtime, Interesting History, Urban Legends, Motivational, Fun Facts, Long Form Jokes, Life Pro Tips, Philosophy, Love, and Custom Topics.
- **Image Generation**: Produce images based on story scenes using AI image generation models.
- **Video Production**: Combine generated images and subtitles into a video with selectable voice options.
- **Multiple Story Types**: Support for various story types, each with tailored generation processes.
- **Character Generation**: For applicable story types, the system generates detailed character descriptions.
- **Customizable Look Presets**: Five named styles (`storyboard`, `noir`, `flat`, `chibi`, `doodle`) that set the image prompt, the palette grade, the camera move and the caption treatment. Older names like `cinematic` still work.
- **Voice Selection**: Multiple voice options for audio generation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SmartClipAI/faceless-video-generator.git
   cd faceless-video-generator
   ```

2. Create and activate a virtual environment:

   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the project root directory with the following content:

    ```plaintext
    # Groq: LLM, speech to text, and (after accepting the model terms) TTS
    GROQ_API_KEY=your_groq_api_key

    # Gemini: free tier text to speech. Image generation needs billing.
    GEMINI_API_KEY=your_gemini_api_key

    # Optional: image generation fallbacks
    REPLICATE_API_TOKEN=your_replicate_api_token
    FAL_KEY=your_fal_api_key

    # Optional: the original OpenAI setup still works
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=your_openai_base_url
    ```

    Note: the system uses Replicate for image generation by default. If you prefer to use FAL's image generation service, you can modify `src/main.py` accordingly.

### Free tier limits to be aware of

| Service | Free tier |
| --- | --- |
| Groq | 200k tokens per day per model, 1k requests per day. Each model has its own budget, and the configured `model_fallbacks` are used when one is exhausted. |
| Groq TTS | The Orpheus models need the terms accepted once in the console before they can be called. |
| Gemini TTS | 10 requests per day **per model**. Longer videos rotate through `tts_models`. |
| Gemini images | **0 requests.** Image generation requires billing. |
| Felo | The free plan does not admit generation calls; the account balance has to be topped up. |
| Local (SD-Turbo) | Unlimited, offline, no key. Slow on a CPU. |

## Usage

1. Activate the virtual environment (if not already activated):

   ```bash
   # On Windows
   .\venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

2. Run the main script via CLI:

   ```bash
   # Default run (YouTube preset, Mystery, storyboard style)
   python src/main.py

   # Custom topic and style for vertical video (Shorts/TikTok)
   python src/main.py --preset shorts --story-type "Scary" --style noir --scenes 14

   # Interactive wizard
   python src/main.py --interactive

   # Audio-only dry run (skip image generation)
   python src/main.py --no-images

   # With custom background music
   python src/main.py --bgm "path/to/music.mp3"
   ```

   CLI Options:
   - `--preset`: `youtube` (1280x720) or `shorts` (720x1280)
   - `--story-type`, `-t`: Story genre (e.g. `Mystery`, `Scary`, `Bedtime`, `Philosophy`, `Fun Facts`)
   - `--style`, `-s`: Visual style (`storyboard`, `noir`, `flat`, `chibi`, `doodle`, etc.)
   - `--scenes`, `-n`: Max scenes to generate (overrides preset default)
   - `--voice`, `-v`: Override TTS voice persona
   - `--no-images`: Dry run without generating images
   - `--bgm`: Path to background audio/music
   - `--interactive`, `-i`: Interactively pick story type, style, and voice from terminal prompts

### Look presets

`src/main.py` picks one of five looks from `config.json` → `styles.profiles`.
A preset sets the prompt suffix, the palette post-process, the camera move and
the caption treatment:

| style | look | camera | captions |
| --- | --- | --- | --- |
| `afro-cartoon` | 2D modern animation skit (clean cel shading, warm sunlight, viral comedy style) | pan / zoom / pop | word captions |
| `stickman` | minimalist clean line art stickman with oval bean body | pop / bob | word captions |
| `storybook` | children's book cartoon illustration, painterly colors | bob / zoom | word captions |
| `storyboard` | flat cartoon, thick outlines, pastel fills | pop / pan | speech bubbles |
| `noir` | sepia ink, spot blacks, hatching | pan / zoom | word captions |
| `flat` | bright vector, no outlines | zoom / pan | word captions |
| `chibi` | big head sticker art, soft shading | bob / pop | word captions |
| `doodle` | thin black line art on white | wipe / pop | word captions |

The palette pass in `src/styles.py` re-grades every frame after generation, so
24 separate images still read as one video. Unknown names such as `cinematic`
keep working: the name is passed to the image model as the style.

### Checking the providers

```bash
python src/test.py              # LLM, TTS and speech to text
python src/test.py              # set with_image=True in the block to add one image
```

## Project Structure

- `src/main.py`: Main script controlling the overall workflow.
- `src/providers.py`: Provider clients, speech chunking, audio merging, transcription and the image provider chain.
- `src/local_image.py`: Local SD-Turbo image generation (no API key required).
- `src/pil_text.py`: Pillow based caption rendering (shortcap's own renderer needs ImageMagick).
- `src/story_generator.py`: Functions for generating stories, translations, and storyboards.
- `src/image_generator.py`: Functions for generating images.
- `src/video_creator.py`: Functions for creating videos and burning in captions.
- `src/audio_generator.py`: Functions for generating audio using the configured TTS provider.
- `src/utils.py`: Utility functions for various tasks.
- `src/transitions.py`: Video transition effects.
- `src/parse_json.py`: JSON parsing utilities.
- `config.json`: Configuration file for various settings.

## Configuration

The `config.json` file contains various settings for the project, including:

### Provider Selection

Every outbound call goes through a provider, and the `providers` block decides which service handles each step:

```json
"providers": {
  "llm": "groq",
  "tts": "gemini",
  "stt": "groq",
  "image": ["gemini", "replicate", "fal"]
}
```

| Provider | Used for | Notes |
| --- | --- | --- |
| `groq` | LLM, TTS, speech to text | OpenAI compatible. `openai/gpt-oss-*` supports strict JSON schema output. |
| `gemini` | TTS, image generation | TTS is free tier; **image generation is not** (the free tier allows 0 image requests). |
| `local` | image generation | Runs SD-Turbo on the CPU through diffusers. No API key, no credits, no network. |
| `openai` | LLM, TTS, speech to text | The original provider, kept as a fallback. |
| `replicate` / `fal` | image generation | Only used through the `image` chain. |

The `image` setting accepts a single provider or an ordered list. Each provider is tried in turn, so a blocked free tier does not break a run.

### Local image generation

The `local` provider generates images on this machine with [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), a distilled Stable Diffusion model that produces good images in a single step. It needs no API key and no credits, which makes it the provider that always works.

```bash
pip install diffusers==0.31.0 transformers==4.46.3 accelerate==0.34.2 safetensors==0.8.0 hf_transfer==0.1.9
```

The weights (~2.4 GB) are downloaded into `models/sd-turbo` on first use and then reused. The `local` block in `config.json` controls the model, the number of steps, the output size and the thread count:

| Setting | Default | Notes |
| --- | --- | --- |
| `model` | `stabilityai/sd-turbo` | Any diffusers text to image model works. |
| `local_dir` | `models/sd-turbo` | Where the weights live. |
| `variant` | `fp16` | Half the download size, loaded as float32 on the CPU. |
| `steps` | `1` | SD-Turbo is distilled for single step generation. More steps means a slower render. |
| `width` / `height` | `576` / `1024` | 9:16. The frames are resized to 720x1280 afterwards. |

Expect roughly **2.5 minutes per image on a 4 core laptop CPU**, so a 14 scene video needs about 40 minutes of rendering. A GPU cuts that down to seconds.

### Story Generation Settings
- `char_limit_min`: Minimum character count for generated stories (700)
- `char_limit_max`: Maximum character count for generated stories (800)

### Storyboard Settings
- `max_scenes`: Maximum number of scenes to generate for each story (14)

### OpenAI Settings
- `model`: The GPT model to use for story generation ("gpt-4")
- `temperature`: Creativity level for story generation (0.9, higher means more creative)

### Groq Settings
- `model`: Default chat model, with `model_fallbacks` used when a model is rate limited
- `reasoning_effort`: `"low"` keeps the reasoning models from leaking markup into the JSON output
- `tts_model`, `tts_voices`, `tts_max_chars`: Orpheus speech settings (200 character input limit, wav output only)
- `whisper_model`: Model used for the caption transcript

### Gemini Settings
- `models`: Image models to try, in order
- `tts_models`: Speech models to try, in order. The free tier allows **10 requests per day per model**, so a longer video rotates through them.
- `tts_voices`: Prebuilt voice names

### Image Generation Settings
The project supports two image generation APIs:

#### Replicate Flux API Settings
- `model`: Image generation model ("black-forest-labs/flux-schnell")
- `aspect_ratio`: Output image aspect ratio ("9:16" for vertical videos)
- `num_inference_steps`: Number of denoising steps (4)
- `disable_safety_checker`: Safety filter toggle (false)
- `guidance`: How closely to follow the prompt (3.0)
- `output_quality`: Image quality setting (100)

#### FAL Flux API Settings
- `model`: Image generation model ("fal-ai/flux/schnell")
- `image_size`: Output image size ("portrait_16_9")
- `num_inference_steps`: Number of denoising steps (4)
- `guidance_scale`: How closely to follow the prompt (3.5)
- `enable_safety_checker`: Safety filter toggle (false)
- `num_images`: Number of images to generate per prompt (1)

### Text-to-Speech Settings
- `speech_rate`: Speed multiplier for generated speech (1.1)

You can modify these settings in the `config.json` file to customize the behavior of the application according to your needs.

## Supported Fonts

![Available Font Styles](assets/font.png)

The application supports the following font styles for text rendering:

- Titan One
- Ranchers
- Rampart One
- Permanent Marker
- Open Sans
- Noto Sans
- Montserrat
- Luckiest Guy
- Knewave
- Jua
- Creepster
- Caveat
- Bungee
- Bebas Neue
- Bangers
- Bakbak One

These fonts can be used for customizing the text appearance in generated videos and images.

