"""Tools for the GAIA Benchmark Agent (search, web/PDF/Wikipedia, files, VLM, audio, YouTube)."""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown
from PIL import Image, ImageOps
from pypdf import PdfReader
from smolagents import tool

logger = logging.getLogger(__name__)

DEFAULT_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
WIKIMEDIA_UA = "GaiaUnit4ResearchAgent/1.0 (https://huggingface.co/spaces; python-requests)"

_DEFAULT_NIM_BASE = "https://integrate.api.nvidia.com/v1"
_DEFAULT_VLM_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def _paginate_text(
    text: str,
    start_char: int,
    max_chars: int,
    label: str = "content",
    supports_search: bool = True,
) -> str:
    """Slice text cleanly with tail-offset support (`start_char < 0`) and continuation hint."""
    total = len(text)
    if total == 0:
        return ""
    max_chars = max(500, min(int(max_chars), 40000))
    if start_char < 0:
        start = max(0, total - max_chars)
    else:
        start = max(0, min(int(start_char), total))
    end = min(total, start + max_chars)
    chunk = text[start:end]
    if start > 0 or end < total:
        search_hint = ", or search_query='...' to search" if supports_search else ""
        return (
            f"{chunk}\n\n"
            f"[Showing chars {start}..{end} of {total} in {label}. "
            f"Pass start_char={end} to read next chunk, start_char=-1 for the end{search_hint}.]"
        )
    return chunk


def _resolve_local_file(file_path: str) -> Path:
    """Resolve `file_path`, falling back to `.cache/gaia_files/<basename>` if only a filename was passed."""
    path = Path(file_path)
    if path.is_file():
        return path
    cached = Path(".cache/gaia_files") / path.name
    if cached.is_file():
        return cached
    return path


def _search_in_text(text: str, query: str, max_hits: int = 8, radius: int = 700) -> str:
    """Return merged context windows matching `query` (case-insensitive) inside `text`."""
    q = query.strip()
    if not q:
        return text
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        # Try individual words if full phrase didn't match.
        words = [w for w in re.split(r"\s+", q) if len(w) >= 3]
        if words:
            word_pat = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
            matches = list(word_pat.finditer(text))
    if not matches:
        return f"No matches found for '{query}' inside document of {len(text)} chars."

    windows: list[tuple[int, int]] = []
    for m in matches[: max_hits * 2]:
        s = max(0, m.start() - radius)
        e = min(len(text), m.end() + radius)
        if windows and s <= windows[-1][1] + 100:
            windows[-1] = (windows[-1][0], max(windows[-1][1], e))
        else:
            windows.append((s, e))
        if len(windows) >= max_hits:
            break

    parts = [f"Found {len(matches)} match(es) for '{query}' (total length {len(text)} chars):"]
    for idx, (s, e) in enumerate(windows, 1):
        parts.append(f"--- Excerpt {idx} (chars {s}..{e}) ---\n{text[s:e]}")
    return "\n\n".join(parts)


def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for idx, page in enumerate(reader.pages, 1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            page_text = f"[Error extracting page {idx}: {exc}]"
        pages.append(f"--- Page {idx} ---\n{page_text.strip()}")
    return "\n\n".join(pages).strip()


def _extract_html_markdown(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if href and not href.startswith(("#", "javascript:", "mailto:")):
            a["href"] = urljoin(base_url, href)
    md = html_to_markdown(str(soup), heading_style="ATX")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _search_brave(query: str, max_results: int, api_key: str) -> list[dict[str, str]]:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("web", {}).get("results", [])
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "snippet": str(item.get("description", "")),
        }
        for item in items[:max_results]
    ]


def _search_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        client = DDGS(timeout=6)
    except TypeError:
        client = DDGS()

    last_exc: Exception | None = None
    for backend in ("bing", "yahoo", "auto"):
        try:
            try:
                raw = list(client.text(query, max_results=max_results, backend=backend))
            except TypeError:
                raw = list(client.text(query, max_results=max_results))
            if raw:
                return [
                    {
                        "title": str(item.get("title", "")),
                        "url": str(item.get("href") or item.get("url") or ""),
                        "snippet": str(item.get("body") or item.get("snippet") or ""),
                    }
                    for item in raw[:max_results]
                ]
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc
    return []


def _search_ddg_lite(query: str, max_results: int) -> list[dict[str, str]]:
    resp = requests.post(
        "https://lite.duckduckgo.com/lite/",
        data={"q": query},
        headers={"User-Agent": DEFAULT_BROWSER_UA, "Referer": "https://lite.duckduckgo.com/"},
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DDG Lite returned HTTP {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for node in soup.select("a.result-link, td.result-snippet"):
        classes = node.get("class") or []
        if "result-link" in classes:
            if len(results) >= max_results:
                break
            href = (node.get("href") or "").strip()
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = urljoin("https://lite.duckduckgo.com", href)
            query_params = parse_qs(urlparse(href).query)
            if "uddg" in query_params:
                href = query_params["uddg"][0]
            if not href or "duckduckgo.com/y.js" in href:
                current = None
                continue
            current = {
                "title": node.get_text(" ", strip=True),
                "url": href,
                "snippet": "",
            }
            results.append(current)
        elif "result-snippet" in classes and current is not None and not current["snippet"]:
            current["snippet"] = node.get_text(" ", strip=True)
            if len(results) >= max_results:
                break

    return results[:max_results]


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """Search the web for a query and return titled URLs and snippets.

    Uses Brave Search when BRAVE_SEARCH_API_KEY is configured, otherwise DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to return (1-20).
    """
    if not query or not query.strip():
        return "Error: search query must be non-empty."
    limit = max(1, min(int(max_results), 20))

    results: list[dict[str, str]] = []
    errors: list[str] = []

    brave_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if brave_key:
        try:
            results = _search_brave(query.strip(), limit, brave_key)
        except Exception as exc:
            errors.append(f"Brave search failed: {exc}")

    if not results:
        try:
            results = _search_ddg(query.strip(), limit)
        except Exception as exc:
            errors.append(f"DuckDuckGo search failed: {exc}")

    if not results:
        try:
            results = _search_ddg_lite(query.strip(), limit)
        except Exception as exc:
            errors.append(f"DuckDuckGo Lite search failed: {exc}")

    if not results:
        if errors:
            return "Web search error: " + "; ".join(errors)
        return f"No web results found for: {query}"

    formatted: list[str] = []
    for idx, item in enumerate(results, 1):
        formatted.append(f"{idx}. [{item['title']}]({item['url']})\n   {item['snippet']}")
    return "\n\n".join(formatted)


@tool
def visit_webpage(
    url: str,
    start_char: int = 0,
    max_chars: int = 20000,
    search_query: str = "",
) -> str:
    """Fetch a webpage or PDF URL and extract readable markdown/text with resolved links.

    Handles HTML pages, PDF documents (including URLs without .pdf extension),
    and plain text files. Supports pagination and in-document keyword search.

    Args:
        url: HTTP or HTTPS URL to fetch.
        start_char: Starting character offset (pass -1 to read the tail of the document).
        max_chars: Maximum characters to return per call (default 20000).
        search_query: Optional keyword or phrase to locate inside the fetched page/PDF.
    """
    clean_url = (url or "").strip()
    parsed = urlparse(clean_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"Error: invalid URL '{url}'. Must start with http:// or https://."

    headers = {
        "User-Agent": DEFAULT_BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(clean_url, headers=headers, timeout=25, allow_redirects=True)
        if resp.status_code == 403 and any(h in parsed.netloc.lower() for h in ("wikipedia.org", "wikimedia.org")):
            resp = requests.get(
                clean_url,
                headers={"User-Agent": WIKIMEDIA_UA, "Accept-Language": "en-US,en;q=0.9"},
                timeout=25,
                allow_redirects=True,
            )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Error fetching URL '{clean_url}': {exc}"

    content_type = resp.headers.get("Content-Type", "").lower()
    raw_bytes = resp.content

    is_pdf = (
        "application/pdf" in content_type
        or resp.url.lower().split("?")[0].endswith(".pdf")
        or raw_bytes[:5] == b"%PDF-"
    )

    if is_pdf:
        try:
            text = _extract_pdf_text(raw_bytes)
        except Exception as exc:
            return f"Error parsing PDF from '{resp.url}': {exc}"
    elif any(content_type.startswith(p) for p in ("image/", "audio/", "video/")) or (
        ("application/zip" in content_type or "application/octet-stream" in content_type)
        and not raw_bytes[:5] == b"%PDF-"
    ):
        return f"Non-text binary content at '{resp.url}' (Content-Type: {content_type}, {len(raw_bytes)} bytes)."
    else:
        if "charset=" not in content_type and (resp.encoding or "").lower() in ("", "iso-8859-1", "latin-1"):
            apparent = getattr(resp, "apparent_encoding", None)
            if isinstance(apparent, str) and apparent:
                resp.encoding = apparent
            else:
                resp.encoding = "utf-8"
        page_text = resp.text
        if "html" in content_type or "<html" in page_text[:500].lower() or "<!doctype" in page_text[:500].lower():
            text = _extract_html_markdown(page_text, resp.url)
        else:
            text = page_text

    if search_query and search_query.strip():
        result = _search_in_text(text, search_query.strip())
        if result.startswith("No matches found"):
            result += (
                " Tip: the page may use different wording or synonyms; call "
                "visit_webpage(url, start_char=0, max_chars=25000) to read directly or try broader keywords."
            )
        return result

    return _paginate_text(text, start_char=start_char, max_chars=max_chars, label=resp.url)


@tool
def wikipedia_lookup(
    query: str,
    title: str = "",
    section: str = "",
    as_of_year: int = 0,
    start_char: int = 0,
    max_chars: int = 20000,
    search_query: str = "",
) -> str:
    """Search English Wikipedia and retrieve full article text/tables or a specific section.

    Preserves Markdown tables (useful for discographies, rosters, medal counts, etc.) and
    supports retrieving historical page revisions via `as_of_year` (e.g. 2022).

    Args:
        query: Search query if exact article title is unknown.
        title: Exact English Wikipedia article title (overrides search when non-empty).
        section: Optional section heading substring to fetch only that section.
        as_of_year: Optional cutoff year (e.g. 2022) to fetch the latest revision on or before Dec 31;
            0 uses current.
        start_char: Starting character offset for pagination (-1 for tail).
        max_chars: Maximum characters to return (default 20000).
        search_query: Optional keyword or phrase to search within the retrieved article/section.
    """
    api_endpoint = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": WIKIMEDIA_UA}

    explicit_title = (title or "").strip()
    page_title = explicit_title
    search_summary = ""

    def _search_first_title(search_term: str) -> tuple[str, str]:
        r = requests.get(
            api_endpoint,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_term.strip(),
                "srlimit": 5,
                "format": "json",
            },
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return "", ""
        lines = ["Top Wikipedia matches:"]
        for i, h in enumerate(hits, 1):
            snippet = BeautifulSoup(h.get("snippet", ""), "html.parser").get_text(" ", strip=True)
            lines.append(f"{i}. {h['title']} — {snippet}")
        return hits[0]["title"], "\n".join(lines) + "\n\n"

    if not page_title:
        if not query or not query.strip():
            return "Error: provide either query or title."
        try:
            page_title, search_summary = _search_first_title(query)
        except Exception as exc:
            return f"Wikipedia search error: {exc}"
        if not page_title:
            return f"No English Wikipedia results found for '{query}'."

    rev_note = ""
    parse_params: dict[str, Any] = {
        "action": "parse",
        "prop": "text|sections",
        "format": "json",
        "disableeditsection": 1,
    }

    if as_of_year and int(as_of_year) > 0:
        cutoff = f"{int(as_of_year)}-12-31T23:59:59Z"
        try:
            rev_resp = requests.get(
                api_endpoint,
                params={
                    "action": "query",
                    "prop": "revisions",
                    "titles": page_title,
                    "redirects": 1,
                    "rvstart": cutoff,
                    "rvlimit": 1,
                    "rvprop": "ids|timestamp",
                    "format": "json",
                },
                headers=headers,
                timeout=15,
            )
            rev_resp.raise_for_status()
            pages = rev_resp.json().get("query", {}).get("pages", {})
            page_obj = next(iter(pages.values()), {})
            if "missing" in page_obj:
                if explicit_title:
                    fallback_title, search_summary = _search_first_title(explicit_title)
                    if not fallback_title:
                        return f"No English Wikipedia results found for '{explicit_title}'."
                    page_title = fallback_title
                    explicit_title = ""
                    rev_resp = requests.get(
                        api_endpoint,
                        params={
                            "action": "query",
                            "prop": "revisions",
                            "titles": page_title,
                            "redirects": 1,
                            "rvstart": cutoff,
                            "rvlimit": 1,
                            "rvprop": "ids|timestamp",
                            "format": "json",
                        },
                        headers=headers,
                        timeout=15,
                    )
                    rev_resp.raise_for_status()
                    pages = rev_resp.json().get("query", {}).get("pages", {})
                    page_obj = next(iter(pages.values()), {})
                if "missing" in page_obj:
                    return f"Wikipedia page '{page_title}' not found."
            revs = page_obj.get("revisions", [])
            if not revs:
                return f"No Wikipedia revision found for '{page_title}' on or before {cutoff}."
            revid = revs[0]["revid"]
            rev_note = f" (revision {revid} at {revs[0]['timestamp']})"
            parse_params["oldid"] = revid
            page_title = page_obj.get("title", page_title)
        except Exception as exc:
            return f"Wikipedia revision lookup error for '{page_title}': {exc}"
    else:
        parse_params["page"] = page_title
        parse_params["redirects"] = 1

    try:
        parse_resp = requests.get(api_endpoint, params=parse_params, headers=headers, timeout=20)
        parse_resp.raise_for_status()
        pdata = parse_resp.json()
        if "error" in pdata:
            if explicit_title and pdata["error"].get("code") == "missingtitle":
                fallback_title, search_summary = _search_first_title(explicit_title)
                if fallback_title:
                    parse_params["page"] = fallback_title
                    page_title = fallback_title
                    parse_resp = requests.get(api_endpoint, params=parse_params, headers=headers, timeout=20)
                    parse_resp.raise_for_status()
                    pdata = parse_resp.json()
            if "error" in pdata:
                return f"Wikipedia parse error: {pdata['error']}"
        parsed = pdata["parse"]
    except Exception as exc:
        return f"Wikipedia fetch error for '{page_title}': {exc}"

    resolved_title = parsed.get("title", page_title)
    sections_meta = parsed.get("sections", [])
    clean_sections = [
        {
            "index": s.get("index"),
            "name": BeautifulSoup(str(s.get("line", "")), "html.parser").get_text(" ", strip=True),
        }
        for s in sections_meta
        if s.get("line")
    ]
    section_names = [s["name"] for s in clean_sections]

    html_content = parsed.get("text", {}).get("*", "")
    if section and section.strip():
        sec_query = section.strip().lower()
        matched_sec = next((s for s in clean_sections if s["name"].lower() == sec_query), None)
        if matched_sec is None:
            matched_sec = next((s for s in clean_sections if sec_query in s["name"].lower()), None)
        if matched_sec is not None:
            sec_params = dict(parse_params)
            sec_params["section"] = matched_sec["index"]
            sec_params["prop"] = "text"
            try:
                sresp = requests.get(api_endpoint, params=sec_params, headers=headers, timeout=20)
                sresp.raise_for_status()
                sdata = sresp.json()
                if "parse" in sdata:
                    html_content = sdata["parse"].get("text", {}).get("*", html_content)
                    rev_note += f" [Section: {matched_sec['name']}]"
            except requests.RequestException as exc:
                logger.debug("Section parse fallback failed for %s: %s", page_title, exc)
        else:
            rev_note += f" [Note: section '{section}' not found; showing full article]"

    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if href.startswith("/"):
            a["href"] = "https://en.wikipedia.org" + href

    md = html_to_markdown(str(soup), heading_style="ATX")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    header = f"# {resolved_title}{rev_note}\n"
    if section_names:
        header += "Sections: " + ", ".join(section_names[:35]) + "\n\n"

    full_body = search_summary + header + md
    if search_query and search_query.strip():
        return search_summary + header + _search_in_text(md, search_query.strip())
    return _paginate_text(full_body, start_char=start_char, max_chars=max_chars, label=resolved_title)


@tool
def read_file_content(
    file_path: str,
    sheet_name: str = "",
    start_char: int = 0,
    max_chars: int = 25000,
    search_query: str = "",
) -> str:
    """Read local spreadsheet (.xlsx/.csv), PDF, Python, JSON, or text files.

    For images (.png/.jpg) or audio (.mp3/.wav), directs you to `analyze_image` or `transcribe_audio`.

    Args:
        file_path: Local path to the attached file.
        sheet_name: Optional Excel sheet name to read (empty reads all sheets).
        start_char: Starting character offset for pagination (-1 for tail).
        max_chars: Maximum characters to return (default 25000).
        search_query: Optional keyword or phrase to search within the file content.
    """
    path = _resolve_local_file(file_path)
    if not path.is_file():
        return f"Error: file not found at '{file_path}'."

    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return (
            f"'{path}' is an image file ({ext}). "
            f"Call analyze_image(image_path='{path}', question=...) to inspect it."
        )
    if ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"):
        return (
            f"'{path}' is an audio file ({ext}). "
            f"Call transcribe_audio(audio_path='{path}') to transcribe it."
        )

    try:
        if ext in (".xlsx", ".xlsm", ".xls"):
            with pd.ExcelFile(path) as excel:
                note = ""
                if sheet_name:
                    exact_sheet = sheet_name if sheet_name in excel.sheet_names else None
                    if exact_sheet is None:
                        ci_map = {s.lower(): s for s in excel.sheet_names}
                        exact_sheet = ci_map.get(sheet_name.strip().lower())
                    if exact_sheet is not None:
                        sheets = [exact_sheet]
                    else:
                        note = f" [Note: sheet '{sheet_name}' not found; reading all sheets]"
                        sheets = excel.sheet_names
                else:
                    sheets = excel.sheet_names
                blocks = [f"Excel file '{path.name}' sheets: {excel.sheet_names}{note}"]
                for s in sheets:
                    df = pd.read_excel(excel, sheet_name=s)
                    blocks.append(f"## Sheet: {s} (shape={df.shape})\n{df.to_csv(index=False)}")
            text = "\n\n".join(blocks)
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df = None
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                df = pd.read_csv(path, sep=sep, encoding="utf-8", encoding_errors="replace")
            text = f"Table '{path.name}' (shape={df.shape})\n{df.to_csv(index=False)}"
        elif ext == ".pdf":
            text = _extract_pdf_text(path.read_bytes())
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading '{file_path}': {exc}"

    if search_query and search_query.strip():
        return _search_in_text(text, search_query.strip())
    return _paginate_text(text, start_char=start_char, max_chars=max_chars, label=path.name)


def _pil_to_b64(
    img: Image.Image,
    min_side: int = 192,
    max_side: int = 2048,
    image_format: str = "PNG",
    quality: int = 85,
) -> str:
    """Normalize PIL image mode & dimensions (>= min_side to avoid NIM 422) and encode base64."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if min(w, h) < min_side:
        # Use bounded 2x NEAREST upscaling so small bitmap glyphs stay crisp (no LANCZOS ringing),
        # then pad below if either dimension is still under min_side.
        max_int_scale = max(1, min(2, max_side // max(1, max(w, h))))
        int_scale = min(max_int_scale, max(1, math.ceil(min_side / max(1, min(w, h)))))
        if int_scale > 1:
            img = img.resize((w * int_scale, h * int_scale), Image.Resampling.NEAREST)
            w, h = img.size

    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        w, h = img.size

    if min(w, h) < min_side:
        nw, nh = max(w, min_side), max(h, min_side)
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((max(0, w - 1), 0)),
            img.getpixel((0, max(0, h - 1))),
            img.getpixel((max(0, w - 1), max(0, h - 1))),
        ]
        rgb_corners = [c for c in corners if isinstance(c, tuple) and len(c) == 3]
        pad_color = max(set(rgb_corners), key=rgb_corners.count) if rgb_corners else (255, 255, 255)
        canvas = Image.new("RGB", (nw, nh), pad_color)
        canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
        img = canvas

    buf = io.BytesIO()
    if image_format.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _prepare_image_b64(image_path: Path, min_side: int = 192, max_side: int = 2048) -> str:
    """Load an image, normalize orientation & small dimensions (>= min_side to avoid NIM 422), and encode PNG base64."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return _pil_to_b64(img, min_side=min_side, max_side=max_side, image_format="PNG")


def _call_vlm(images_b64: list[str], prompt: str) -> str:
    """Call an OpenAI-compatible Vision Language Model (NIM primary, OpenAI fallback)."""
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    candidates: list[tuple[str, str, str]] = []
    if nvidia_key:
        vlm_model = os.getenv("GAIA_VLM_MODEL_ID", _DEFAULT_VLM_MODEL).strip() or _DEFAULT_VLM_MODEL
        api_base = os.getenv("GAIA_API_BASE", _DEFAULT_NIM_BASE).strip().rstrip("/")
        candidates.append((api_base, nvidia_key, vlm_model))
        if vlm_model != _DEFAULT_VLM_MODEL:
            candidates.append((_DEFAULT_NIM_BASE, nvidia_key, _DEFAULT_VLM_MODEL))
    if openai_key:
        candidates.append(("https://api.openai.com/v1", openai_key, "gpt-4o"))

    if not candidates:
        return "Error: no VLM API key configured (set NVIDIA_API_KEY or OPENAI_API_KEY)."

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        mime = "image/jpeg" if b64.startswith("/9j/") else "image/png"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    errors: list[str] = []
    for base_url, api_key, model_id in candidates:
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 8192,
                    "temperature": 0.0,
                },
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            answer = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            if answer:
                return answer
            errors.append(f"{model_id}: empty response")
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    return "VLM error: " + "; ".join(errors)


@tool
def analyze_image(
    image_path: str,
    question: str = (
        "Describe everything visible in this image in full detail, including all text, numbers, labels, and positions."
    ),
) -> str:
    """Analyze an image file using a Vision Language Model.

    Args:
        image_path: Local path to the image file (.png, .jpg, .jpeg, .webp).
        question: Specific question or analysis instruction for the image.
    """
    path = _resolve_local_file(image_path)
    if not path.is_file():
        return f"Error: image file not found at '{image_path}'."
    try:
        b64 = _prepare_image_b64(path)
    except Exception as exc:
        return f"Error loading image '{image_path}': {exc}"
    return _call_vlm([b64], question)


_WHISPER_MODEL: Any = None


def _get_whisper_model() -> Any:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel  # type: ignore

        _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _WHISPER_MODEL


@tool
def transcribe_audio(audio_path: str) -> str:
    """Transcribe spoken audio from an audio file (.mp3, .wav, .m4a, .ogg, .flac).

    Args:
        audio_path: Local path to the audio file.
    """
    path = _resolve_local_file(audio_path)
    if not path.is_file():
        return f"Error: audio file not found at '{audio_path}'."

    errors: list[str] = []

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            with path.open("rb") as f:
                resp = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    data={"model": "whisper-1"},
                    files={"file": (path.name, f)},
                    timeout=90,
                )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if text:
                return text
        except Exception as exc:
            errors.append(f"OpenAI Whisper failed: {exc}")

    ran_local_whisper = False
    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(str(path))
        text = " ".join(seg.text.strip() for seg in segments).strip()
        ran_local_whisper = True
        if text:
            return text
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"faster-whisper failed: {exc}")

    hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
    if hf_token:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=hf_token)
            out = client.automatic_speech_recognition(str(path), model="openai/whisper-large-v3")
            text = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", str(out))
            if str(text).strip():
                return str(text).strip()
        except Exception as exc:
            errors.append(f"HF ASR failed: {exc}")

    if ran_local_whisper:
        return "No speech detected in audio."
    if not errors:
        return (
            "Audio transcription unavailable: install 'faster-whisper' or configure "
            "HF_TOKEN (for whisper-large-v3) or OPENAI_API_KEY."
        )
    return "Audio transcription failed: " + "; ".join(errors)


def _extract_youtube_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]
    qs = parse_qs(parsed.query)
    if qs.get("v"):
        return qs["v"][0]
    m = re.search(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{6,})", parsed.path)
    return m.group(1) if m else ""


def _fetch_youtube_transcript(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    if hasattr(api, "fetch"):
        try:
            transcript = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            tlist = api.list(video_id)
            first = next(iter(tlist))
            transcript = first.fetch()
        lines = []
        for entry in transcript:
            start = getattr(entry, "start", None) if not isinstance(entry, dict) else entry.get("start")
            text = getattr(entry, "text", None) if not isinstance(entry, dict) else entry.get("text")
            if text:
                ts = f"[{float(start):.1f}s] " if start is not None else ""
                lines.append(f"{ts}{text}")
        return "\n".join(lines)

    # Fallback for youtube-transcript-api < 1.0 static API.
    entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    return "\n".join(f"[{e['start']:.1f}s] {e['text']}" for e in entries)


def _fetch_youtube_metadata(video_id: str) -> dict[str, Any]:
    """Fetch title, author, duration, and description via yt-dlp with oEmbed fallback."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    meta: dict[str, Any] = {}
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL(
            {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 15}
        ) as ydl:
            info = ydl.extract_info(video_url, download=False)
        if isinstance(info, dict):
            meta["title"] = str(info.get("title") or "")
            meta["author"] = str(info.get("uploader") or info.get("channel") or "")
            meta["duration"] = info.get("duration")
            meta["description"] = str(info.get("description") or "")
            if meta["title"] or meta["description"]:
                return meta
    except Exception as exc:
        logger.debug("yt-dlp metadata lookup failed for %s: %s", video_id, exc)

    try:
        oembed = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": video_url, "format": "json"},
            timeout=10,
        )
        if oembed.status_code == 200:
            data = oembed.json()
            meta["title"] = str(data.get("title") or "")
            meta["author"] = str(data.get("author_name") or "")
    except requests.RequestException as exc:
        logger.debug("YouTube oEmbed failed for %s: %s", video_id, exc)
    return meta


def _sample_video_frames(
    url: str,
    num_frames: int,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
) -> list[str]:
    """Download a low-res stream via yt-dlp into a temp dir and sample `num_frames` JPEG base64 strings."""
    import yt_dlp

    tmp_dir = Path(tempfile.mkdtemp(prefix="gaia_yt_"))
    try:
        out_tpl = str(tmp_dir / "video.%(ext)s")
        opts = {
            "format": (
                "bestvideo[height<=360][vcodec^=avc1]/"
                "bestvideo[height<=360][ext=mp4]/"
                "best[height<=360]/"
                "bestvideo[height<=480][vcodec^=avc1]/"
                "best[height<=480]/bestvideo/best"
            ),
            "outtmpl": out_tpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "socket_timeout": 20,
            "retries": 2,
            "fragment_retries": 2,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        files = [p for p in tmp_dir.iterdir() if p.is_file()]
        if not files:
            raise RuntimeError("yt-dlp produced no video file.")
        video_path = files[0]

        frames_b64: list[str] = []
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(video_path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                if total <= 0:
                    raise RuntimeError("Video has zero readable frames.")
                fps = float(cap.get(getattr(cv2, "CAP_PROP_FPS", 5)) or 0.0)
                start_idx = 0
                end_idx = total - 1
                if fps > 0 and start_sec > 0:
                    start_idx = max(0, min(total - 1, int(start_sec * fps)))
                if fps > 0 and end_sec > start_sec:
                    end_idx = max(start_idx, min(total - 1, int(end_sec * fps)))

                span = max(1, end_idx - start_idx + 1)
                count = max(1, min(num_frames, span))
                indices = [
                    min(end_idx, start_idx + int((i + 0.5) * span / count))
                    for i in range(count)
                ]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    frames_b64.append(_pil_to_b64(pil_img, min_side=192, max_side=960, image_format="JPEG"))
            finally:
                cap.release()
        except ImportError:
            # Fallback to ffmpeg CLI if cv2 is unavailable.
            if not shutil.which("ffmpeg"):
                raise RuntimeError(
                    "Neither opencv-python-headless nor ffmpeg is installed for frame extraction."
                ) from None
            cmd = ["ffmpeg"]
            if start_sec > 0:
                cmd += ["-ss", f"{start_sec:.2f}"]
            if end_sec > start_sec:
                cmd += ["-to", f"{end_sec:.2f}"]
            cmd += [
                "-i",
                str(video_path),
                "-vf",
                f"fps=1/{max(1, 5)}",
                "-frames:v",
                str(num_frames),
                str(tmp_dir / "frame_%03d.png"),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            for fp in sorted(tmp_dir.glob("frame_*.png"))[:num_frames]:
                with Image.open(fp) as pil_img:
                    frames_b64.append(_pil_to_b64(pil_img, min_side=192, max_side=960, image_format="JPEG"))

        return frames_b64
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@tool
def youtube_inspect(
    url: str,
    extract_frames: bool = False,
    num_frames: int = 12,
    question: str = "",
    start_sec: float = 0.0,
    end_sec: float = 0.0,
) -> str:
    """Inspect a YouTube video: fetch metadata/description, transcript/captions, and optionally analyze keyframes.

    Args:
        url: Full YouTube video URL.
        extract_frames: Set True to sample video frames and run VLM analysis
            (useful when captions are missing or visual counting is required).
        num_frames: Number of evenly-spaced frames to sample when extract_frames is True (1-12, default 12).
        question: Visual question to answer over sampled frames when extract_frames is True.
        start_sec: Optional start time in seconds to zoom frame sampling into a segment.
        end_sec: Optional end time in seconds (0 = end of video).
    """
    video_id = _extract_youtube_id(url)
    if not video_id:
        return f"Error: could not extract YouTube video ID from '{url}'."

    sections: list[str] = [f"YouTube Video ID: {video_id}"]

    # 1. Metadata (title, author, duration, description).
    meta = _fetch_youtube_metadata(video_id)
    meta_lines = []
    if meta.get("title"):
        meta_lines.append(f"Title: {meta['title']}")
    if meta.get("author"):
        meta_lines.append(f"Author: {meta['author']}")
    if meta.get("duration") is not None:
        meta_lines.append(f"Duration: {meta['duration']}s")
    if meta.get("description"):
        meta_lines.append(f"Description:\n{meta['description'][:2000]}")
    if meta_lines:
        sections.append("\n".join(meta_lines))

    # 2. Captions / transcript.
    try:
        transcript_text = _fetch_youtube_transcript(video_id)
        sections.append(
            "Transcript:\n" + _paginate_text(transcript_text, 0, 15000, "transcript", supports_search=False)
        )
    except Exception as exc:
        sections.append(f"Transcript unavailable: {exc}")

    # 3. Optional visual frame sampling + VLM.
    if extract_frames:
        try:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            count = max(1, min(int(num_frames), 12))
            frames = _sample_video_frames(video_url, count, start_sec=start_sec, end_sec=end_sec)
            if frames:
                vlm_q = (question or "").strip() or "Describe what is visible across these video frames."
                ctx_parts = []
                if meta.get("title"):
                    ctx_parts.append(f"Video title: {meta['title']}")
                if meta.get("description"):
                    ctx_parts.append(f"Video description: {meta['description'][:1200]}")
                ctx_block = ("\n".join(ctx_parts) + "\n\n") if ctx_parts else ""
                frame_prompt = (
                    f"{ctx_block}"
                    f"These are {len(frames)} sequential frames (Frame 1 to Frame {len(frames)}, "
                    f"ordered start to end) from the video.\n"
                    f"First, write one brief line for EACH frame (Frame 1 through Frame {len(frames)}) listing "
                    f"every distinct entity, species, person, object, action, or on-screen text visible in that frame "
                    f"(check small/background details carefully and distinguish juveniles/chicks from different "
                    f"adult species mentioned in the title or description).\n"
                    f"Then, synthesize your frame-by-frame observations (or identify the single frame with the most "
                    f"simultaneous elements if the question asks for a simultaneous count) to give a clear, "
                    f"well-supported answer to: {vlm_q}"
                )
                vlm_ans = _call_vlm(frames, frame_prompt)
                sections.append(f"Visual frame analysis ({len(frames)} frames):\n{vlm_ans}")
            else:
                sections.append("Visual frame analysis: no frames extracted.")
        except Exception as exc:
            sections.append(f"Visual frame extraction failed: {exc}")

    return "\n\n".join(sections)


def build_tools() -> list[Any]:
    """Return the complete list of tools registered with the GAIA CodeAgent."""
    return [
        web_search,
        visit_webpage,
        wikipedia_lookup,
        read_file_content,
        analyze_image,
        transcribe_audio,
        youtube_inspect,
    ]
