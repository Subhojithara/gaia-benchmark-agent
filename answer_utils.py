"""GAIA prompt rules, answer normalization, and attachment retrieval utilities."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
GAIA_DATASET_REPO = "gaia-benchmark/GAIA"

# Official GAIA answer-formatting rules adapted for smolagents CodeAgent:
# the agent passes ONLY the final answer value to `final_answer(...)` (without
# emitting the literal "FINAL ANSWER:" wrapper in the submitted string).
GAIA_SYSTEM_INSTRUCTIONS = """
You are a general AI assistant solving GAIA benchmark questions.
Always verify facts using your tools (web_search, visit_webpage, wikipedia_lookup,
youtube_inspect, read_file_content, analyze_image, transcribe_audio, or Python code)
rather than guessing from memory.
If an attached file path is provided in the task, inspect or read that file first before
searching the web.
If a tool returns an error, truncated text, or no keyword matches on a relevant page,
adjust your query, read the page sequentially (start_char=0), try synonyms/broader words,
or follow linked sources until you have verified the answer.
For YouTube URLs, call youtube_inspect(url) first, and pass extract_frames=True if captions
are unavailable or visual inspection is required.
When a question specifies a historical year or compilation date (e.g., 2022 Wikipedia or an
archived page date), use wikipedia_lookup(..., as_of_year=...) or check archived snapshots
(such as web.archive.org) if the live page changed.

When you call `final_answer(answer)`, pass ONLY the final answer string/number itself —
do NOT include "FINAL ANSWER:" or any explanation around it.
Follow these strict GAIA formatting rules for the value passed to `final_answer`:
1. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list
   of numbers and/or strings.
2. If you are asked for a number, don't use comma to write your number neither use units such
   as $ or percent sign unless specified otherwise. If an exact number of decimal places is
   requested (e.g., two decimal places), pass a formatted string such as "89706.00".
3. If you are asked for a string, don't use articles (the, a, an), neither abbreviations
   (e.g. for cities), and write the digits in plain text unless specified otherwise.
4. If you are asked for a comma separated list, apply the rules above depending on whether each
   element is a number or a string, and join elements with ", " (comma + space), following any
   requested ordering (such as alphabetical or ascending).
5. Do not add trailing periods or surrounding quotes to your final answer.
""".strip()

_PRESERVED_DOT_SUFFIXES = (
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "jr.",
    "sr.",
    "st.",
    "inc.",
    "ltd.",
    "co.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
)

_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*(?:(?:\*\*|__|\*|`)\s*)?:\s*", re.IGNORECASE)
_THOUSANDS_NUMBER_RE = re.compile(r"^[+-]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?$")
_CURRENCY_THOUSANDS_RE = re.compile(r"^[$€£¥][+-]?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?$")
_NUMERIC_TOKEN_RE = re.compile(r"[+-]?(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:\.\d+)?")
_NUMERIC_LIST_RE = re.compile(
    r"^[+-]?(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:\.\d+)?(?:,[+-]?(?:[1-9]\d{0,2}(?:,\d{3})+|\d+)(?:\.\d+)?)+$"
)
_DECIMAL_TRAILING_DOT_RE = re.compile(r"^[+-]?\d[\d,]*\.\d+\.$")


def _has_preserved_trailing_dot(text: str) -> bool:
    last_tok = text.rsplit(None, 1)[-1].lower()
    if last_tok in _PRESERVED_DOT_SUFFIXES:
        return True
    if _DECIMAL_TRAILING_DOT_RE.match(text):
        return False
    if last_tok.count(".") > 1:
        return True
    if len(text) > 2 and re.fullmatch(r"[a-z]\.", last_tok):
        return True
    return False


def _normalize_comma_segment(seg: str) -> list[str]:
    seg = seg.strip()
    if not seg:
        return []
    if _THOUSANDS_NUMBER_RE.match(seg):
        return [seg.replace(",", "")]
    if _CURRENCY_THOUSANDS_RE.match(seg):
        return [seg]
    if _NUMERIC_LIST_RE.match(seg):
        return [
            tok.replace(",", "") if _THOUSANDS_NUMBER_RE.match(tok) else tok
            for tok in _NUMERIC_TOKEN_RE.findall(seg)
        ]
    subparts = [p.strip() for p in re.split(r"(?<!\d)\s*,\s*|\s*,\s*(?!\d)", seg) if p.strip()]
    out: list[str] = []
    for part in subparts:
        if part == seg:
            out.append(re.sub(r"\s*,\s*(?=\S)", ", ", part))
        else:
            out.extend(_normalize_comma_segment(part))
    return out


def normalize_answer(raw: Any) -> str:
    """Normalize an agent output into a clean string compatible with GAIA exact-match scoring."""
    if raw is None:
        return ""
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float):
        return str(raw) if math.isfinite(raw) else ""
    if isinstance(raw, (list, tuple, set)):
        parts = [normalize_answer(item) for item in raw]
        return ", ".join(p for p in parts if p != "")
    if isinstance(raw, dict):
        for key in ("final_answer", "answer", "submitted_answer", "result", "value"):
            if key in raw and raw[key] is not None:
                return normalize_answer(raw[key])
        return json.dumps(raw, ensure_ascii=False)

    text = str(raw).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    # If the model literally echoed "FINAL ANSWER: ...", extract from the last marker.
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    had_marker = bool(matches)
    if matches:
        prefix = text[: matches[-1].start()].rstrip()
        after = text[matches[-1].end() :]
        for md in ("```", "**", "__", "`", "*"):
            if (
                prefix.endswith(md)
                and after.startswith(md)
                and (len(after) == len(md) or after[len(md)].isspace())
            ):
                after = after[len(md) :]
                break
        non_empty_lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
        text = non_empty_lines[0] if non_empty_lines else ""
        for md in ("```", "**", "__", "`", "*"):
            if prefix.endswith(md) and text.endswith(md) and not text.startswith(md):
                text = text[: len(text) - len(md)].strip()

    # Strip matching outer markdown/quote/bracket wrappers repeatedly until stable.
    wrappers: list[tuple[str, str]] = [
        ("```", "```"),
        ("**", "**"),
        ("__", "__"),
        ("`", "`"),
        ("*", "*"),
        ('"', '"'),
        ("'", "'"),
    ]
    if had_marker:
        wrappers.append(("[", "]"))

    prev = None
    while text != prev:
        prev = text
        for left, right in wrappers:
            if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
                text = text[len(left) : len(text) - len(right)].strip()

        # Remove a stray sentence-ending period unless it looks like an abbreviation or ellipsis.
        if len(text) > 1 and text.endswith(".") and not text.endswith("..") and not _has_preserved_trailing_dot(text):
            text = text[:-1].rstrip()

    # Collapse internal whitespace/newlines to a single space.
    text = re.sub(r"\s+", " ", text).strip()

    # Strip thousands-separator commas from numbers and ensure ", " spacing in lists.
    if "," in text:
        if _THOUSANDS_NUMBER_RE.match(text):
            text = text.replace(",", "")
        elif _CURRENCY_THOUSANDS_RE.match(text):
            return text
        else:
            segments = re.split(r"\s*,\s+", text)
            normalized_segments: list[str] = []
            for seg in segments:
                normalized_segments.extend(_normalize_comma_segment(seg))
            text = ", ".join(normalized_segments)

    return text


def fetch_task_file(
    task_id: str,
    file_name: str,
    api_url: str = DEFAULT_API_URL,
    cache_dir: str | Path = ".cache/gaia_files",
    hf_token: str | None = None,
) -> str | None:
    """Download a GAIA task attachment via the scoring API, falling back to Hugging Face Hub.

    Returns the local file path if available, or None if unavailable.
    """
    if not file_name or not str(file_name).strip():
        return None

    safe_name = Path(str(file_name)).name
    if not safe_name:
        return None

    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / safe_name

    if local_path.is_file() and local_path.stat().st_size > 0:
        return str(local_path.resolve())

    # 1. Try the course scoring API `/files/{task_id}`.
    if task_id:
        file_url = f"{api_url.rstrip('/')}/files/{task_id}"
        try:
            resp = requests.get(file_url, timeout=20)
            if resp.status_code == 200 and resp.content:
                local_path.write_bytes(resp.content)
                return str(local_path.resolve())
        except requests.RequestException:
            pass

    # 2. Fall back to the gated gaia-benchmark/GAIA dataset repo when HF_TOKEN is available.
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    if token:
        try:
            from huggingface_hub import hf_hub_download

            for split_dir in ("2023/validation", "2023/test"):
                try:
                    cached = hf_hub_download(
                        repo_id=GAIA_DATASET_REPO,
                        repo_type="dataset",
                        filename=f"{split_dir}/{safe_name}",
                        token=token,
                    )
                    shutil.copy2(cached, local_path)
                    return str(local_path.resolve())
                except Exception:
                    continue
        except ImportError:
            pass

    return None
