import base64
import os
import time
from typing import Any, Dict, List, Optional

import requests
import replicate
import fal_client

from utils import load_config


# --------------------------------------------------------------------------- #
# Replicate
# --------------------------------------------------------------------------- #

def _is_permanent_error(error: Exception) -> bool:
    """True for errors that will not get better by retrying."""
    status = getattr(error, "status", None)
    if status in (400, 401, 402, 403, 404):
        return True
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "unauthenticated",
            "status: 401",
            "status: 403",
            "invalid api key",
            "please set the fal_key",
            "authentication",
        )
    )


# --------------------------------------------------------------------------- #
# Replicate
# --------------------------------------------------------------------------- #

def replicate_flux_api(prompt: str, max_retries: int = 3) -> Optional[bytes]:
    config = load_config()
    replicate_config = config["replicate_flux_api"]

    if not os.getenv("REPLICATE_API_TOKEN"):
        print("REPLICATE_API_TOKEN is not set; skipping Replicate image generation.")
        return None

    payload = {
        "prompt": prompt,
        "aspect_ratio": replicate_config["aspect_ratio"],
        "num_inference_steps": replicate_config["num_inference_steps"],
        "disable_safety_checker": replicate_config["disable_safety_checker"],
        "guidance": replicate_config["guidance"],
        "output_quality": replicate_config["output_quality"],
    }

    for attempt in range(max_retries):
        try:
            image_urls = replicate.run(
                config["replicate_flux_api"]["model"], input=payload
            )
            if image_urls and isinstance(image_urls, list) and len(image_urls) > 0:
                image_url = image_urls[0]
                response = requests.get(image_url)
                response.raise_for_status()
                return response.content
            else:
                raise ValueError("No image URL returned from Replicate API")
        except Exception as e:
            if _is_permanent_error(e):
                print(f"Replicate image generation is unavailable: {e}")
                return None
            if attempt < max_retries - 1:
                print(
                    f"Error in Flux Schnell generation (attempt {attempt + 1}/{max_retries}): {e}"
                )
                print("Retrying...")
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                print(
                    f"Error in Flux Schnell generation after {max_retries} attempts: {e}"
                )
    return None


# --------------------------------------------------------------------------- #
# fal.ai
# --------------------------------------------------------------------------- #

def submit_fal_request(prompt: str, config: dict) -> Optional[str]:
    try:
        handler = fal_client.submit(
            config["model"],
            arguments={
                "prompt": prompt,
                "image_size": config["image_size"],
                "num_images": config["num_images"],
                "num_inference_steps": config["num_inference_steps"],
                "enable_safety_checker": config["enable_safety_checker"],
            },
        )

        result = handler.get()
        if result and isinstance(result, dict) and "images" in result:
            images = result["images"]
            if isinstance(images, list) and images:
                return images[0].get("url")

        return None
    except Exception as e:
        print(f"Error in FAL AI API request: {e}")
        return None


def fal_flux_api(prompt: str, max_retries: int = 3) -> Optional[bytes]:
    config = load_config()
    fal_config = config["fal_flux_api"]

    if not (os.getenv("FAL_KEY") or os.getenv("FAL_KEY_ID")):
        print("FAL_KEY is not set; skipping fal.ai image generation.")
        return None

    for attempt in range(max_retries):
        try:
            image_url = submit_fal_request(prompt, fal_config)
            if image_url:
                response = requests.get(image_url)
                response.raise_for_status()
                return response.content
            else:
                raise ValueError("No image URL returned from FAL AI API")
        except Exception as e:
            if _is_permanent_error(e):
                print(f"fal.ai image generation is unavailable: {e}")
                return None
            if attempt < max_retries - 1:
                print(
                    f"Error in FAL AI API request (attempt {attempt + 1}/{max_retries}): {e}"
                )
                print("Retrying...")
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                print(f"Error in FAL AI API request after {max_retries} attempts: {e}")
    return None


# --------------------------------------------------------------------------- #
# Felo
# --------------------------------------------------------------------------- #

_FELO_IMAGE_MODELS: Optional[List[str]] = None


def discover_felo_image_models(refresh: bool = False) -> List[str]:
    """List the image generation models this Felo account can actually use."""
    global _FELO_IMAGE_MODELS
    if _FELO_IMAGE_MODELS is not None and not refresh:
        return _FELO_IMAGE_MODELS

    _FELO_IMAGE_MODELS = []
    config = load_config()
    base_url = config["felo"]["base_url"].rstrip("/")
    api_key = os.getenv("FELO_API_KEY")
    if not api_key:
        print("FELO_API_KEY is not set; skipping Felo image models.")
        return _FELO_IMAGE_MODELS

    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        _FELO_IMAGE_MODELS = [
            model["id"]
            for model in models
            if model.get("model_kind") == "image_generation" and model.get("id")
        ]
        if _FELO_IMAGE_MODELS:
            print(f"Felo image models available: {', '.join(_FELO_IMAGE_MODELS)}")
        else:
            print("No Felo image generation models are enabled on this account.")
    except Exception as e:
        print(f"Failed to list Felo image models: {e}")

    return _FELO_IMAGE_MODELS


def felo_image_api(prompt: str, max_retries: Optional[int] = None) -> Optional[bytes]:
    """Generate an image through Felo's OpenAI compatible images endpoint."""
    config = load_config()
    felo_config = config["felo"]
    base_url = felo_config["base_url"].rstrip("/")
    api_key = os.getenv("FELO_API_KEY")
    if not api_key:
        print("FELO_API_KEY is not set; skipping Felo image generation.")
        return None

    model = felo_config.get("model", "auto")
    if model == "auto":
        models = discover_felo_image_models()
        if not models:
            return None
        model = models[0]

    retries = max_retries or felo_config.get("max_retries", 3)
    payloads = [
        {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "aspect_ratio": felo_config.get("aspect_ratio"),
            "size": felo_config.get("size"),
        },
        {"model": model, "prompt": prompt, "n": 1},
    ]

    for attempt in range(retries):
        payload = payloads[0] if attempt == 0 else payloads[-1]
        try:
            response = requests.post(
                f"{base_url}/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            if response.status_code >= 400:
                print(f"Felo image error {response.status_code}: {response.text[:300]}")
                if response.status_code in (400, 401, 402, 403, 404):
                    # The free plan does not admit image generation calls.
                    return None
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

            data = response.json()
            entries = data.get("data") or []
            if not entries:
                print("Felo returned no image data.")
                return None

            entry = entries[0]
            if entry.get("b64_json"):
                return base64.b64decode(entry["b64_json"])
            if entry.get("url"):
                image_response = requests.get(entry["url"], timeout=120)
                image_response.raise_for_status()
                return image_response.content
            print("Felo response contained neither url nor b64_json.")
            return None
        except Exception as e:
            print(f"Error in Felo image generation (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

def _extract_gemini_image(payload: Dict[str, Any]) -> Optional[bytes]:
    """Pull the generated image out of a Gemini generateContent response."""
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            # Nano Banana emits intermediate "thought" images; skip them.
            if part.get("thought"):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline:
                continue
            data = inline.get("data")
            if data:
                try:
                    return base64.b64decode(data)
                except Exception as e:
                    print(f"Failed to decode Gemini image payload: {e}")
                    return None
    return None


def _gemini_image_quota_blocked(response: Any) -> bool:
    """True when Gemini refuses image generation outright (free tier, billing)."""
    try:
        message = response.json().get("error", {}).get("message", "") or ""
    except Exception:
        return False
    if "limit: 0" in message:
        return True
    if response.status_code == 403:
        return True
    return False


def gemini_image_api(prompt: str, max_retries: Optional[int] = None) -> Optional[bytes]:
    """Generate an image with a Gemini (Nano Banana) image model."""
    config = load_config()
    gemini_config = config["gemini"]
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set; skipping Gemini image generation.")
        return None

    base_url = gemini_config["base_url"].rstrip("/")
    retries = max_retries or gemini_config.get("max_retries", 3)
    versions = gemini_config.get("api_versions", ["v1beta", "v1"])

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": gemini_config.get("aspect_ratio", "9:16"),
                "imageSize": gemini_config.get("image_size", "1K"),
            },
        },
    }

    for model in gemini_config.get("models", []):
        for version in versions:
            url = f"{base_url}/{version}/models/{model}:generateContent"
            for attempt in range(retries):
                try:
                    response = requests.post(
                        url,
                        headers={
                            "x-goog-api-key": api_key,
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=240,
                    )
                    if response.status_code == 404:
                        # Wrong API version or unknown model: try the next one.
                        break
                    if _gemini_image_quota_blocked(response):
                        print(
                            "Gemini image generation is not available on this account: "
                            "the free tier grants 0 image requests. To enable it, link "
                            "a billing account to the Google Cloud project that owns "
                            "the API key (Google Cloud console > Billing > Link "
                            "account), or use another image provider."
                        )
                        return None
                    if response.status_code in (429, 500, 502, 503, 504):
                        if attempt < retries - 1:
                            print(
                                f"Gemini image request {response.status_code}, retrying "
                                f"(attempt {attempt + 2}/{retries})"
                            )
                            time.sleep(2 ** attempt)
                            continue
                        print(f"Gemini image request failed: {response.status_code}")
                        break
                    if response.status_code >= 400:
                        print(
                            f"Gemini image error {response.status_code}: {response.text[:400]}"
                        )
                        break

                    payload = response.json()
                    image = _extract_gemini_image(payload)
                    if image:
                        print(f"Gemini generated an image with {model}.")
                        return image
                    print(f"Gemini returned no image data for {model}.")
                    break
                except requests.RequestException as e:
                    print(f"Gemini image request error (attempt {attempt + 1}/{retries}): {e}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
    return None
