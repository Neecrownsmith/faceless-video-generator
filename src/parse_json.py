import json
import re


def extract_and_parse_json2(response: str) -> dict:
    # Remove leading and trailing backticks and "json" tag if present
    cleaned_response = response.strip().strip("`").strip()
    if cleaned_response.startswith("json"):
        cleaned_response = cleaned_response[4:].strip()

    try:
        storyboard_data = json.loads(cleaned_response)
        return storyboard_data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")


def extract_and_parse_json(response: str) -> dict:
    """
    Extracts JSON from a string response and attempts to parse it.

    Args:
    response (str): The full response string, potentially containing non-JSON text.

    Returns:
    dict: Parsed JSON data if successful, empty dict if failed.
    """
    # Find the JSON part of the response
    json_match = re.search(r"\{.*\}", response, re.DOTALL)

    if json_match:
        json_str = json_match.group()
        try:
            parsed_data = json.loads(json_str)
            print("Successfully parsed JSON data.")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Failed to parse JSON: {json_str}")
    else:
        print("No JSON object found in the response")

    return {}


if __name__ == "__main__":
    print(extract_and_parse_json('Here: {"title": "A {curly} affair", "scenes": []}'))
    print(extract_and_parse_json('```json\n{"title": "Fenced"}\n```'))
    print(extract_and_parse_json("no json here"))
