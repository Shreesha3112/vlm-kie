"""Repair and parse JSON from LLM outputs.

LLMs often wrap JSON in markdown fences or produce minor formatting errors.
This module extracts and repairs JSON robustly.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract a JSON object or array from a string.

    Handles:
    - ```json ... ``` markdown fences
    - Leading/trailing text around the JSON
    - Single JSON object or array
    """
    # Strip markdown fences
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if fenced:
        return fenced.group(1).strip()

    # Find the first { or [ (whichever comes first) and match to its closing pair
    positions = {'{': text.find('{'), '[': text.find('[')}
    # Remove chars not present
    positions = {ch: pos for ch, pos in positions.items() if pos != -1}
    if not positions:
        return text

    # Pick the pair whose opening char appears earliest
    start_char = min(positions, key=lambda ch: positions[ch])
    end_char = '}' if start_char == '{' else ']'
    pairs = [(start_char, end_char)]

    for start_char, end_char in pairs:
        start = positions[start_char]
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return text  # Return as-is; let json.loads surface the error


def parse_llm_json(text: str) -> dict | list | None:
    """Parse JSON from LLM output, attempting several repair strategies.

    Returns parsed object or None on failure.
    """
    raw = extract_json(text)

    # Attempt 1: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: remove trailing commas (common LLM error)
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: replace single quotes with double quotes (LLM shortcut)
    try:
        cleaned2 = re.sub(r"'([^']*)'", r'"\1"', cleaned)
        return json.loads(cleaned2)
    except json.JSONDecodeError:
        pass

    logger.warning("JSON repair failed. Raw output: %.200s", text)
    return None
