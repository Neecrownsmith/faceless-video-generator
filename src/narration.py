"""Narration planning: decouple the spoken text from the visual scenes.

The storyboard used to control both at once - one scene meant one TTS request,
one image, and a clip whose length equaled that request. Asking for more scenes
therefore cut the narration into shorter utterances, and every fragment was
synthesized in isolation, which is what made the delivery sound flat.

Here the story is reconstructed from the scene subtitles (they are exact quotes
from the story), packed into sentence aware chunks, synthesized once per chunk,
and only then sliced into images using the measured audio duration. The visual
count and the spoken script stop being the same number.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from audio_generator import generate_audio
from providers import concat_audio_files
from utils import get_narration_config

# Sentence terminators followed by whitespace. Ellipses are treated as a
# terminator so a pause stays with the clause it ends.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str) -> List[str]:
    """Split text on sentence terminators, keeping the punctuation."""
    if not text or not text.strip():
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip())]
    return [part for part in parts if part]


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """Break a sentence that alone exceeds the chunk budget, on clause edges."""
    pieces: List[str] = []
    buffer = ""

    for clause in _CLAUSE_SPLIT.split(sentence):
        if not clause:
            continue
        candidate = f"{buffer} {clause}".strip() if buffer else clause
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer.strip())
        while len(clause) > max_chars:
            pieces.append(clause[:max_chars].strip())
            clause = clause[max_chars:].strip()
        buffer = clause

    if buffer:
        pieces.append(buffer.strip())
    return pieces


def pack_sentences(text: str, max_chars: int) -> List[str]:
    """Group sentences into chunks of at most `max_chars`.

    A sentence is never split unless it alone exceeds the budget, because a
    request that spans complete sentences is what keeps the prosody natural.
    """
    chunks: List[str] = []
    buffer = ""

    for sentence in split_sentences(text):
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split_long_sentence(sentence, max_chars))
            continue

        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            chunks.append(buffer.strip())
            buffer = sentence

    if buffer:
        chunks.append(buffer.strip())
    return chunks


_BOUNDARY_PUNCT = ",;:.!?\u2026"

# Punctuation that must stay attached to the word before it, and characters
# that must not be separated from the word after them.
_NO_SPACE_BEFORE = set(".,;:!?)]}%\u2026\u201d\u2019")
_NO_SPACE_AFTER = set("([{\u201c\u2018")


def _normalize(text: str) -> str:
    """Collapse whitespace so two renderings of the same words compare equal."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def _needs_separator(left: str, right: str) -> bool:
    """True when two adjacent pieces of text need a space between them."""
    if not left or not right:
        return False
    if left[-1].isspace() or right[0].isspace():
        # The piece already carries its own whitespace.
        return False
    if right[0] in _NO_SPACE_BEFORE or left[-1] in _NO_SPACE_AFTER:
        return False
    return True


def build_narration_and_offsets(
    scene_texts: List[str], story: str
) -> Tuple[str, List[Tuple[int, int]]]:
    """Rebuild the spoken text from the scene subtitles.

    Subtitles are meant to be exact quotes, so each one is located inside the
    story and the story's own characters are emitted for it. That keeps the
    original punctuation intact even when the model invents a comma at the
    point where it decided to cut the text. Fragments that cannot be found - a
    garbled word, or the opening hook that is not part of the story at all - are
    emitted verbatim so the words still survive.

    Returns the narration and each scene's character span inside it.
    """
    story_text = _normalize(story)
    parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0        # position inside the narration being assembled
    story_cursor = 0  # how much of the story has already been emitted

    def emit(piece: str) -> Tuple[int, int]:
        nonlocal cursor
        # A separator is only wanted between two words; punctuation that sits
        # on a fragment boundary must stay glued to the word before it.
        if parts and _needs_separator(parts[-1][-1], piece[0]):
            parts.append(" ")
            cursor += 1
        parts.append(piece)
        start = cursor
        cursor += len(piece)
        return start, cursor

    for text in scene_texts:
        probe = _normalize(text)
        if not probe:
            offsets.append((cursor, cursor))
            continue

        matched = None
        span = story_text.find(probe, story_cursor) if story_text else -1
        if span >= 0:
            matched = probe
        else:
            stripped = probe.rstrip(_BOUNDARY_PUNCT)
            span = story_text.find(stripped, story_cursor) if story_text else -1
            if span >= 0 and stripped:
                matched = stripped

        if matched is None:
            # Unrecognised fragment: keep the model's wording.
            offsets.append(emit(probe))
            continue

        if span > story_cursor:
            # The model skipped part of the story; carry it across so nothing
            # is left unsaid.
            emit(story_text[story_cursor:span])
            story_cursor = span

        offsets.append(emit(matched))
        story_cursor = span + len(matched)

    if story_cursor < len(story_text):
        emit(story_text[story_cursor:])

    return "".join(parts).strip(), offsets


def build_plan(
    storyboards: List[Dict[str, Any]], story: str, max_chars: int
) -> Dict[str, Any]:
    """Rebuild the spoken text and split it into narration chunks.

    Every chunk keeps the scenes it covers so the images can be timed against
    the real audio afterwards.
    """
    scene_texts = [(scene.get("subtitles") or "").strip() for scene in storyboards]
    narration, offsets = build_narration_and_offsets(scene_texts, story)
    # `emit` never writes leading whitespace, so the spans already start at 0.
    offsets = _clamp_offsets(offsets, len(narration))

    chunks: List[Dict[str, Any]] = []
    cursor = 0
    for piece in pack_sentences(narration, max_chars):
        span = narration.find(piece, cursor)
        if span < 0:
            span = cursor
        start, end = span, span + len(piece)
        cursor = end

        scenes = [
            {
                "index": index,
                "scene_number": storyboards[index].get("scene_number", index + 1),
                # Weight decides how much of the chunk's audio a scene gets.
                "weight": max(1, len(scene_texts[index])),
            }
            for index, (scene_start, scene_end) in enumerate(offsets)
            if scene_end > scene_start and scene_start < end and scene_end > start
        ]
        if not scenes:
            continue
        chunks.append(
            {
                "text": piece,
                "text_start": start,
                "text_end": end,
                "scenes": scenes,
            }
        )

    return {"text": narration, "chunks": chunks}


def _clamp_offsets(offsets: List[Tuple[int, int]], length: int) -> List[Tuple[int, int]]:
    """Keep spans ordered and inside the narration."""
    clamped: List[Tuple[int, int]] = []
    cursor = 0
    for start, end in offsets:
        start = min(max(start, cursor), length)
        end = min(max(end, start), length)
        clamped.append((start, end))
        cursor = end
    return clamped


def _measure(path: str) -> Optional[float]:
    """Duration of an audio file in seconds, or None when unreadable."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        from moviepy.editor import AudioFileClip

        clip = AudioFileClip(path)
        try:
            return float(clip.duration)
        finally:
            clip.close()
    except Exception as e:
        print(f"Could not read the duration of {path}: {e}")
        return None


def _synthesize_chunk(
    text: str,
    output_file: str,
    tts_client,
    voice_name: str,
    retries: int,
) -> Optional[float]:
    """Synthesize one narration chunk, falling back to single sentences.

    Losing narration is worse than slightly clipped prosody, so a chunk that
    will not synthesize as a whole is split into sentences and merged.
    """
    for attempt in range(max(1, retries)):
        if generate_audio(tts_client, text, output_file, voice_name):
            duration = _measure(output_file)
            if duration:
                return duration
            print(f"Narration for {os.path.basename(output_file)} produced no audio.")
        elif attempt < retries - 1:
            print(f"Retrying narration chunk {os.path.basename(output_file)}...")

        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return None

    print(
        f"Falling back to sentence level synthesis for {os.path.basename(output_file)}."
    )
    stem, extension = os.path.splitext(output_file)
    parts: List[str] = []
    for position, sentence in enumerate(sentences, 1):
        part = f"{stem}_part{position}{extension}"
        if not generate_audio(tts_client, sentence, part, voice_name):
            return None
        parts.append(part)

    try:
        concat_audio_files(parts, output_file)
        return _measure(output_file)
    finally:
        for part in parts:
            try:
                os.remove(part)
            except OSError:
                pass


def synthesize_plan(
    plan: Dict[str, Any],
    tts_client,
    voice_name: str,
    audio_dir: str,
    extension: str,
    retries: int,
) -> List[Dict[str, Any]]:
    """Synthesize every chunk and stamp its measured duration onto the plan."""
    os.makedirs(audio_dir, exist_ok=True)
    finished: List[Dict[str, Any]] = []

    for position, chunk in enumerate(plan["chunks"], 1):
        output_file = os.path.join(audio_dir, f"narration_{position}.{extension}")
        print(
            f"Narration chunk {position}/{len(plan['chunks'])} "
            f"({len(chunk['text'])} chars)..."
        )
        duration = _synthesize_chunk(
            chunk["text"], output_file, tts_client, voice_name, retries
        )
        if not duration:
            print(f"Skipping narration chunk {position}; no audio could be produced.")
            continue
        finished.append({**chunk, "file": output_file, "duration": duration})

    return finished


def _fit_chunk(indices: List[int], span: float, min_shot: float) -> List[int]:
    """Drop scenes from the end of a chunk when the audio cannot hold them.

    The narration length decides how many images are affordable; rendering an
    image that can only be on screen for a fraction of a second wastes the
    generation time and makes the video strobe.
    """
    fitted = list(indices)
    while len(fitted) > 1 and len(fitted) * min_shot > span:
        dropped = fitted.pop()
        print(
            f"Scene {dropped + 1} dropped: only {span:.1f}s of narration is left."
        )
    return fitted


def _distribute(span: float, weights: List[int], min_shot: float, max_shot: float) -> List[float]:
    """Split `span` across `weights`, keeping every shot inside the bounds."""
    count = len(weights)
    if count == 0:
        return []

    total = float(sum(weights)) or 1.0
    durations = [span * weight / total for weight in weights]

    for _ in range(4):
        durations = [max(min_shot, min(max_shot, d)) for d in durations]
        current = sum(durations)
        if abs(current - span) < 1e-6 or current <= 0:
            break
        durations = [d * (span / current) for d in durations]

    durations = [max(min_shot, d) for d in durations]
    # Absorb the rounding remainder in the last shot so the timeline and the
    # narration track end together.
    durations[-1] = max(min_shot, durations[-1] + (span - sum(durations)))
    return durations


def build_timeline(
    chunks: List[Dict[str, Any]], min_shot: float, max_shot: float
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Build the absolute timeline and report which scene indices survive."""
    cursor = 0.0
    for chunk in chunks:
        chunk["start"] = cursor
        cursor += float(chunk["duration"])

    timeline: List[Dict[str, Any]] = []
    for chunk in chunks:
        span = float(chunk["duration"])
        scenes = chunk["scenes"]
        fitted = _fit_chunk([scene["index"] for scene in scenes], span, min_shot)
        if not fitted:
            continue

        weights = [scene["weight"] for scene in scenes if scene["index"] in fitted]
        durations = _distribute(span, weights, min_shot, max_shot)

        inner = 0.0
        for index, duration in zip(fitted, durations):
            timeline.append(
                {
                    "scene_index": index,
                    "scene_number": next(
                        scene["scene_number"] for scene in scenes if scene["index"] == index
                    ),
                    "start": float(chunk["start"]) + inner,
                    "duration": duration,
                }
            )
            inner += duration

    kept = sorted({entry["scene_index"] for entry in timeline})
    return timeline, kept


def run_narration(
    storyboards: List[Dict[str, Any]],
    story: str,
    tts_client,
    voice_name: str,
    audio_dir: str,
    extension: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Synthesize the narration and decide which scenes get an image.

    Returns the kept storyboards, the timeline the renderer follows, and the
    narration manifest. The storyboard list can come back shorter than it went
    in when the audio is too short to justify every image.
    """
    if not storyboards:
        return [], [], None

    # Scene numbers become positions so image file names stay stable and unique.
    for position, scene in enumerate(storyboards, 1):
        scene["scene_number"] = position

    config = get_narration_config()
    plan = build_plan(storyboards, story, config["chunk_chars"])

    if not plan["text"]:
        print("No narration text was produced by the storyboard.")
        return storyboards, [], None

    print(
        f"Narration: {len(plan['text'])} characters across "
        f"{len(plan['chunks'])} chunk(s)."
    )

    chunks = synthesize_plan(
        plan,
        tts_client,
        voice_name,
        audio_dir,
        extension,
        config["max_retries"],
    )
    if not chunks:
        print("Narration could not be synthesized; falling back to per scene audio.")
        return storyboards, [], None

    total = sum(chunk["duration"] for chunk in chunks)
    print(f"Narration duration: {total:.1f}s.")

    timeline, kept_indices = build_timeline(
        chunks, config["min_shot_seconds"], config["max_shot_seconds"]
    )

    index_map = {old: new for new, old in enumerate(kept_indices)}
    for entry in timeline:
        entry["scene_index"] = index_map[entry["scene_index"]]

    kept_storyboards = [storyboards[index] for index in kept_indices]
    for position, scene in enumerate(kept_storyboards, 1):
        scene["narration_chunk"] = position

    if not kept_storyboards:
        return storyboards, [], None

    if len(kept_storyboards) < len(storyboards):
        print(
            f"{len(kept_storyboards)} of {len(storyboards)} scenes will get an image."
        )

    track = os.path.join(audio_dir, f"narration.{extension}")
    try:
        concat_audio_files([chunk["file"] for chunk in chunks], track)
    except Exception as e:
        print(f"Failed to merge the narration track: {e}")
        track = chunks[0]["file"]

    manifest = {
        "text": plan["text"],
        "total_duration": total,
        "chunks": [
            {
                "index": position,
                "text": chunk["text"],
                "duration": chunk["duration"],
                "file": chunk["file"],
            }
            for position, chunk in enumerate(chunks, 1)
        ],
        "track": track,
    }
    return kept_storyboards, timeline, manifest


if __name__ == "__main__":
    from utils import get_narration_config

    storyboards = [
        {"scene_number": 1, "subtitles": "Eliot Finch, a bookish archivist,"},
        {"scene_number": 2, "subtitles": "dreams of unearthing the forgotten truth."},
        {"scene_number": 3, "subtitles": "The moonlit library's attic holds a sealed journal."},
    ]
    story = (
        "Eliot Finch, a bookish archivist, dreams of unearthing the forgotten "
        "truth. The moonlit library's attic holds a sealed journal, its pages "
        "inked with an impossible cipher."
    )

    narration, offsets = build_narration_and_offsets(
        [scene["subtitles"] for scene in storyboards], story
    )
    print(narration)
    for start, end in offsets:
        print(start, end, narration[start:end])

    plan = build_plan(storyboards, story, get_narration_config()["chunk_chars"])
    for chunk in plan["chunks"]:
        print(chunk["text"], [scene["index"] for scene in chunk["scenes"]])
