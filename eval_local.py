"""Local dry-run runner for Unit 4 GAIA questions (never calls POST /submit)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from agent import GaiaAgent
from answer_utils import DEFAULT_API_URL

load_dotenv()


def _save_results(out_path: Path, results: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)


def _load_existing_rows(out_path: Path) -> list[dict[str, Any]]:
    if not out_path.is_file():
        return []
    try:
        raw = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict) and item.get("task_id")]
    except Exception:
        pass
    return []


def _load_completed_results(out_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for item in _load_existing_rows(out_path):
        if str(item.get("answer") or "").strip() and not item.get("error"):
            completed[str(item["task_id"])] = item
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GaiaAgent locally on Unit 4 questions without submitting.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of questions to run (0 = all selected).")
    parser.add_argument("--skip-files", action="store_true", help="Skip questions requiring attached files.")
    parser.add_argument("--task-ids", nargs="*", default=[], help="Optional specific task_id values to run.")
    parser.add_argument("--output", default=".cache/dry_run_results.json", help="Path to write JSON results.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already-completed (non-error, non-empty) results from --output and run remaining questions.",
    )
    args = parser.parse_args(argv)

    questions = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15).json()
    if args.task_ids:
        wanted = set(args.task_ids)
        questions = [q for q in questions if q.get("task_id") in wanted]
    if args.skip_files:
        questions = [q for q in questions if not q.get("file_name")]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    out_path = Path(args.output)
    merged_by_id: dict[str, dict[str, Any]] = {}
    if args.resume:
        for row in _load_existing_rows(out_path):
            merged_by_id[str(row["task_id"])] = row
    completed = {
        tid: row
        for tid, row in merged_by_id.items()
        if str(row.get("answer") or "").strip() and not row.get("error")
    }

    agent: GaiaAgent | None = None
    for idx, item in enumerate(questions, 1):
        tid = item["task_id"]
        fn = item.get("file_name", "") or ""
        qtext = item["question"]

        if tid in completed:
            cached_row = completed[tid]
            merged_by_id[tid] = cached_row
            print(f"[{idx}/{len(questions)}] {tid} (resumed) file={fn!r} -> {cached_row['answer']!r}")
            _save_results(out_path, list(merged_by_id.values()))
            continue

        if agent is None:
            agent = GaiaAgent()

        t0 = time.time()
        try:
            ans = agent(qtext, task_id=tid, file_name=fn)
            err = ""
        except Exception as exc:
            ans = ""
            err = f"{type(exc).__name__}: {exc}"
        dt = time.time() - t0
        print(f"[{idx}/{len(questions)}] {tid} ({dt:.1f}s) file={fn!r} -> {ans!r}" + (f" ERR={err}" if err else ""))
        merged_by_id[tid] = {
            "task_id": tid,
            "file_name": fn,
            "question": qtext,
            "answer": ans,
            "error": err,
            "elapsed_s": round(dt, 2),
        }
        _save_results(out_path, list(merged_by_id.values()))

    final_rows = list(merged_by_id.values())
    _save_results(out_path, final_rows)
    print(f"Saved {len(final_rows)} results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
