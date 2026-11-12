---
title: GAIA Benchmark Agent - Unit 4 Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Benchmark Agent (Hugging Face AI Agents Course — Unit 4)

A general-purpose research agent built for the **Hugging Face AI Agents Course (Unit 4 Final Assignment)** to solve multi-step Level-1 questions from the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) (`>= 30%` exact-match target).

The repository is compatible with both **local CLI dry-run evaluation** and **Hugging Face Spaces** (`Gradio` + `hf_oauth: true`).

## Architecture & Key Features

- **Agent Brain (`agent.py`)**: Uses `smolagents.CodeAgent` with Python execution (`max_steps=18` default), per-question interpreter state isolation, markdown code-block handling (`code_block_tags="markdown"` / `stop=REMOVE_PARAMETER`), primary NVIDIA NIM reasoning model (`nvidia/nemotron-3-ultra-550b-a55b`) with automatic secondary fallback (`mistralai/mistral-large-4-1t-a52b`), plus optional OpenAI (`OPENAI_API_KEY`) and Hugging Face Inference (`HF_TOKEN`) support.
- **Exact-Match Answer Normalization (`answer_utils.py`)**: Adapts GAIA formatting rules for `CodeAgent` `final_answer(...)` calls and normalizes outputs (strips accidental `"FINAL ANSWER:"` markers, markdown wrappers, trailing periods outside abbreviations/initials, thousands-separator commas in numbers, and spacing in comma-separated lists).
- **Resilient Attachment Retrieval (`answer_utils.py`)**: Tries the Unit 4 scoring `/files/{task_id}` endpoint first and falls back to `gaia-benchmark/GAIA` (`2023/validation` / `2023/test`) via `huggingface_hub` when `HF_TOKEN` is available.
- **Streaming Gradio Runner (`app.py`)**: Streams per-question status updates over Gradio's queue, caches solved answers incrementally to `.cache/answers_cache.json` so interrupted runs resume cleanly, supports Hugging Face OAuth submission (`POST /submit`), and automatically disables the OAuth login widget during local unauthenticated previews.

## Tools (`tools.py`)

| Tool | Purpose |
|---|---|
| `web_search` | Web search via Brave Search API (when `BRAVE_SEARCH_API_KEY` is set) or DuckDuckGo (`ddgs` multi-backend + DDG Lite fallback). |
| `visit_webpage` | Fetches HTML (converted to clean Markdown with resolved links), PDFs (detected by `Content-Type` / `%PDF-` magic bytes, including `/pdf` URLs without `.pdf`), and text pages with pagination and keyword search. |
| `wikipedia_lookup` | MediaWiki search + full/section article retrieval preserving Markdown tables and supporting historical revisions (`as_of_year`). |
| `read_file_content` | Reads `.xlsx`/`.xlsm`/`.xls` (sheet-aware), `.csv`/`.tsv` (encoding fallback), `.pdf`, `.py`, `.json`, and text files with pagination and search. |
| `analyze_image` | Vision analysis via NVIDIA NIM VLM (`nvidia/nemotron-3.5-super-vl-120b-a12b`, OpenAI `gpt-4o` fallback) with EXIF transposition and small-image upscaling/padding (`>= 192px`). |
| `transcribe_audio` | Speech transcription via OpenAI Whisper API (if `OPENAI_API_KEY` is set), local `faster-whisper` (`base` CPU `int8`), or Hugging Face Inference `openai/whisper-large-v3` (if `HF_TOKEN` is set). |
| `youtube_inspect` | Extracts YouTube video metadata/description, captions/transcripts (`youtube-transcript-api`), and optional low-res keyframe sampling + VLM inspection (`yt-dlp` + `opencv-python-headless` / `ffmpeg`). |

## Repository Layout

- `app.py` — Gradio UI, OAuth flow, incremental answer cache, and Unit 4 `/submit` runner.
- `agent.py` — `GaiaAgent` (`BasicAgent` alias), model selection, retry/fallback, and `CodeAgent` configuration.
- `tools.py` — Web search, webpage/PDF reader, Wikipedia lookup, file readers, VLM, ASR, and YouTube inspection tools.
- `answer_utils.py` — GAIA system instructions, `normalize_answer`, and `fetch_task_file`.
- `eval_local.py` — Local CLI evaluation runner (`--skip-files`, `--limit`, `--task-ids`, `--resume`, never calls `POST /submit`).
- `tests/` — Unit and contract tests (`test_answer_utils.py`, `test_tools.py`, `test_agent_and_app.py`).
- `.github/workflows/ci.yml` — Automated Ruff lint and Pytest checks on pull requests and `main`.

## Local Setup

1. Create a Python 3.11 virtual environment and install dependencies:

```bash
uv venv .venv --python 3.11
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Configure environment variables (never commit `.env`):

```bash
cp .env.example .env
```

Set at least one LLM key (`NVIDIA_API_KEY`, `OPENAI_API_KEY`, or `HF_TOKEN`):

- `NVIDIA_API_KEY` — Primary NVIDIA NIM key (`nvidia/nemotron-3-ultra-550b-a55b`, `nvidia/nemotron-3.5-super-vl-120b-a12b`).
- `HF_TOKEN` — Needed to fetch gated `gaia-benchmark/GAIA` task attachments when `/files/{task_id}` is unavailable, and for HF Inference fallback.
- `OPENAI_API_KEY` — Optional (`gpt-4o` / `whisper-1`).
- `BRAVE_SEARCH_API_KEY` — Optional; upgrades `web_search` to Brave Search.

## Verification & Local Evaluation

Run lint and unit/contract tests:

```bash
ruff check .
pytest -v
```

Run a safe local dry-run against the Unit 4 `/questions` endpoint (does not submit to the leaderboard):

```bash
python eval_local.py --skip-files --output .cache/dry_run_results.json
# Or resume an interrupted run:
python eval_local.py --output .cache/dry_run_results.json --resume
```

Launch the local Gradio interface:

```bash
python app.py
```

## Deploying to Hugging Face Spaces

1. Create or duplicate a public Gradio Space (`sdk_version: 5.25.2`, `hf_oauth: true` in `README.md`) and push the repository files (excluding `.env` and `.cache/`).
2. In **Space Settings → Variables and secrets**, add `NVIDIA_API_KEY` (and `HF_TOKEN` with accepted terms on `gaia-benchmark/GAIA`, plus optional `BRAVE_SEARCH_API_KEY` / `OPENAI_API_KEY`).
3. Open the Space, sign in with **Login with Hugging Face**, and click **Run Evaluation & Submit All Answers**.

## License

Licensed under the [MIT License](LICENSE).
