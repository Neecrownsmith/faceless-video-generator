"""JSON extraction utilities for LLM responses."""

import json
import logging
import re
from typing import Any, Dict, Optional


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences such as ```json ... ```."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_and_parse_json(response: str) -> Optional[Dict[str, Any]]:
    """Extract and parse a JSON object or array out of a model response string.

    Tries standard json.loads on stripped content first, then searches for the
    outermost JSON object or array if commentary/fences are present.
    """
    if not response:
        return None

    cleaned = strip_code_fences(response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find outermost object { ... }
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    # Find outermost array [ ... ]
    arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON array: {e}")

    logging.error("No valid JSON found in response.")
    return None
