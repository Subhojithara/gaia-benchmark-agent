"""Hugging Face Space Gradio UI for running and submitting the Unit 4 GAIA agent."""

import json
import os
import textwrap
from pathlib import Path

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv

from agent import GaiaAgent
from answer_utils import DEFAULT_API_URL, normalize_answer

try:
    import spaces
except ImportError:  # Local run without the `spaces` package (no ZeroGPU).
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    spaces = _SpacesFallback()

load_dotenv()

CACHE_FILE = Path(".cache/answers_cache.json")
RESULT_COLUMNS = ["Task ID", "File", "Question", "Submitted Answer", "Source"]


def _results_df(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _load_answer_cache(cache_path: Path | None = None) -> dict[str, str]:
    target = CACHE_FILE if cache_path is None else cache_path
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float)) and str(v) != ""}
    except Exception:
        pass
    return {}


def _save_answer_cache(cache: dict[str, str], cache_path: Path | None = None) -> None:
    target = CACHE_FILE if cache_path is None else cache_path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(target)


# ZeroGPU requires at least one @spaces.GPU function for the Space to start.
# The agent's models run remotely (NVIDIA NIM), so the long evaluation runs
# on CPU; this short utility carries the decorator so the Space can start.
@spaces.GPU(duration=60)
def clear_cache() -> str:
    if CACHE_FILE.is_file():
        CACHE_FILE.unlink()
    return "Answer cache cleared."


def run_and_submit_all(profile: gr.OAuthProfile | None, use_cache: bool = True):
    """Fetch Unit 4 questions, stream per-question agent progress with caching, and submit answers."""
    space_id = os.getenv("SPACE_ID")

    if profile and getattr(profile, "username", None):
        username = f"{profile.username}".strip()
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        yield "Please Login to Hugging Face with the button.", None
        return

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    yield f"Initializing GAIA agent for '{username}'...", None
    try:
        agent = GaiaAgent()
    except Exception as exc:
        print(f"Error instantiating agent: {exc}")
        yield f"Error initializing agent: {exc}", None
        return

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "https://huggingface.co/spaces/local-run/tree/main"
    print(f"Agent code URL: {agent_code}")

    print(f"Fetching questions from: {questions_url}")
    yield f"Fetching questions from {questions_url}...", None
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            yield "Fetched questions list is empty or invalid format.", None
            return
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as exc:
        yield f"Error fetching questions: {exc}", None
        return
    except requests.exceptions.JSONDecodeError as exc:
        yield f"Error decoding server response for questions: {exc}", None
        return

    valid_items = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        valid_items.append(item)

    if not valid_items:
        yield "Fetched questions list is empty or invalid format.", None
        return

    cache = _load_answer_cache() if use_cache else {}
    results_log: list[dict[str, str]] = []
    answers_payload: list[dict[str, str]] = []
    total = len(valid_items)

    print(f"Running agent on {total} questions...")
    yield f"Loaded {total} questions. Starting evaluation (0/{total})...", _results_df(results_log)

    for idx, item in enumerate(valid_items, 1):
        task_id = item["task_id"]
        question_text = item["question"]
        file_name = item.get("file_name", "") or ""

        cached_raw = cache.get(task_id, "") if use_cache else ""
        cached_answer = normalize_answer(cached_raw) if cached_raw else ""
        if use_cache and cached_answer:
            answers_payload.append({"task_id": task_id, "submitted_answer": cached_answer})
            results_log.append(
                {
                    "Task ID": task_id,
                    "File": file_name,
                    "Question": question_text,
                    "Submitted Answer": cached_answer,
                    "Source": "cache",
                }
            )
            yield (
                f"[{idx}/{total}] Loaded cached answer for {task_id} ({len(answers_payload)} ready).",
                _results_df(results_log),
            )
            continue

        file_note = f" (file: {file_name})" if file_name else ""
        yield (
            f"[{idx}/{total}] Running agent on {task_id}{file_note}...",
            _results_df(results_log),
        )
        try:
            submitted_answer = normalize_answer(agent(question_text, task_id=task_id, file_name=file_name))
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append(
                {
                    "Task ID": task_id,
                    "File": file_name,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                    "Source": "agent",
                }
            )
            if submitted_answer:
                cache[task_id] = submitted_answer
                _save_answer_cache(cache)
        except Exception as exc:
            print(f"Error running agent on task {task_id}: {exc}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "File": file_name,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT ERROR: {exc}",
                    "Source": "error",
                }
            )
        yield (
            f"[{idx}/{total}] Finished {task_id} ({len(answers_payload)}/{total} answered).",
            _results_df(results_log),
        )

    if not answers_payload:
        yield "Agent did not produce any answers to submit.", _results_df(results_log)
        return

    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers for user '{username}' to {submit_url}...")
    yield (
        f"Submitting {len(answers_payload)} answers for user '{username}' to {submit_url}...",
        _results_df(results_log),
    )

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        yield final_status, _results_df(results_log)
    except requests.exceptions.HTTPError as exc:
        error_detail = f"Server responded with status {exc.response.status_code}."
        try:
            error_json = exc.response.json()
            error_detail += f" Detail: {error_json.get('detail', exc.response.text)}"
        except Exception:
            error_detail += f" Response: {exc.response.text[:500]}"
        yield f"Submission Failed: {error_detail}", _results_df(results_log)
    except requests.exceptions.Timeout:
        yield "Submission Failed: The request timed out.", _results_df(results_log)
    except requests.exceptions.RequestException as exc:
        yield f"Submission Failed: Network error - {exc}", _results_df(results_log)


def _has_hf_oauth_context() -> bool:
    """Return True when running inside an HF Space or when a valid local HF user token exists."""
    if os.getenv("SYSTEM") == "spaces" and os.getenv("SPACE_ID"):
        return True
    try:
        from huggingface_hub import get_token, whoami

        if not get_token():
            return False
        info = whoami()
        return isinstance(info, dict) and info.get("type") == "user"
    except Exception:
        return False


def build_demo(enable_oauth: bool | None = None) -> gr.Blocks:
    """Construct the Gradio Blocks application."""
    use_oauth = _has_hf_oauth_context() if enable_oauth is None else enable_oauth

    with gr.Blocks() as blocks:
        gr.Markdown("# GAIA Benchmark Agent — Unit 4 Final Assignment")
        gr.Markdown(
            textwrap.dedent(
                """
                **Instructions:**

                1. Log in to your Hugging Face account using the button below (required for leaderboard submission).
                2. Ensure your Space Secrets (`NVIDIA_API_KEY` or `OPENAI_API_KEY`, plus `HF_TOKEN`
                   for gated GAIA attachments) are configured.
                3. Click **Run Evaluation & Submit All Answers** to run the `smolagents` `CodeAgent`
                   across all Unit 4 questions and submit your answers.
                """
            ).strip()
        )

        if use_oauth:
            gr.LoginButton()
        else:
            gr.Markdown(
                "_Local preview mode: `HF_TOKEN` / `SPACE_ID` not set, so HF OAuth login is disabled locally "
                "(it activates automatically on Hugging Face Spaces)._"
            )

        with gr.Row():
            use_cache_checkbox = gr.Checkbox(value=True, label="Use Answer Cache (resume partial runs)")
            clear_cache_btn = gr.Button("Clear Answer Cache")

        run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
        status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
        results_table = gr.DataFrame(label="Questions and Agent Answers", headers=RESULT_COLUMNS, wrap=True)

        clear_cache_btn.click(fn=clear_cache, outputs=[status_output])
        run_button.click(
            fn=run_and_submit_all,
            inputs=[use_cache_checkbox],
            outputs=[status_output, results_table],
            show_progress="minimal",
        )

    blocks.queue(default_concurrency_limit=1)
    return blocks


demo = build_demo()


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")
    if space_host_startup:
        print(f"SPACE_HOST: {space_host_startup} (https://{space_host_startup}.hf.space)")
    if space_id_startup:
        print(f"SPACE_ID: {space_id_startup} (https://huggingface.co/spaces/{space_id_startup}/tree/main)")
    print("-" * (60 + len(" App Starting ")) + "\n")
    demo.launch(debug=True, share=False)
