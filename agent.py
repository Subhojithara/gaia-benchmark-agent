"""GAIA Benchmark Agent built on smolagents CodeAgent."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from smolagents import REMOVE_PARAMETER, CodeAgent, InferenceClientModel, OpenAIServerModel

from answer_utils import DEFAULT_API_URL, GAIA_SYSTEM_INSTRUCTIONS, fetch_task_file, normalize_answer
from tools import build_tools

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_NIM_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_NIM_FALLBACK_MODEL = "mistralai/mistral-large-4-1t-a52b"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

AUTHORIZED_IMPORTS = [
    "bisect",
    "bs4",
    "collections",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "io",
    "itertools",
    "json",
    "math",
    "numpy",
    "openpyxl",
    "operator",
    "pandas",
    "pathlib",
    "pypdf",
    "random",
    "re",
    "requests",
    "statistics",
    "string",
    "time",
    "urllib",
]


def create_model(model_id: str | None = None) -> Any:
    """Create the primary LLM model from environment configuration."""
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
    provider_pref = os.getenv("GAIA_PROVIDER", "").strip().lower()

    if (provider_pref == "openai" or not nvidia_key) and openai_key:
        chosen_model = model_id or os.getenv("GAIA_MODEL_ID", "gpt-4o").strip() or "gpt-4o"
        api_base = os.getenv("GAIA_API_BASE", "https://api.openai.com/v1").strip()
        return OpenAIServerModel(
            model_id=chosen_model,
            api_base=api_base,
            api_key=openai_key,
            client_kwargs={"timeout": 120.0, "max_retries": 2},
            temperature=0.0,
        )

    if nvidia_key:
        chosen_model = model_id or os.getenv("GAIA_MODEL_ID", DEFAULT_NIM_MODEL).strip() or DEFAULT_NIM_MODEL
        api_base = os.getenv("GAIA_API_BASE", DEFAULT_NIM_BASE).strip()
        return OpenAIServerModel(
            model_id=chosen_model,
            api_base=api_base,
            api_key=nvidia_key,
            client_kwargs={"timeout": 120.0, "max_retries": 2},
            max_tokens=8192,
            temperature=0.0,
            stop=REMOVE_PARAMETER,
        )

    chosen_model = model_id or os.getenv("GAIA_MODEL_ID", DEFAULT_HF_MODEL).strip() or DEFAULT_HF_MODEL
    return InferenceClientModel(
        model_id=chosen_model,
        token=hf_token or None,
        max_tokens=4096,
    )


def _create_fallback_model() -> Any | None:
    """Return a secondary NIM model when NVIDIA_API_KEY is active and primary != fallback."""
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    primary = os.getenv("GAIA_MODEL_ID", DEFAULT_NIM_MODEL).strip() or DEFAULT_NIM_MODEL
    if nvidia_key and primary != DEFAULT_NIM_FALLBACK_MODEL:
        return OpenAIServerModel(
            model_id=DEFAULT_NIM_FALLBACK_MODEL,
            api_base=os.getenv("GAIA_API_BASE", DEFAULT_NIM_BASE).strip(),
            api_key=nvidia_key,
            client_kwargs={"timeout": 120.0, "max_retries": 2},
            max_tokens=8192,
            temperature=0.0,
            stop=REMOVE_PARAMETER,
        )
    return None


class GaiaAgent:
    """General-purpose GAIA research agent using CodeAgent and custom tools."""

    def __init__(
        self,
        max_steps: int | None = None,
        cache_dir: str | Path = ".cache/gaia_files",
        api_url: str = DEFAULT_API_URL,
        verbosity_level: int = 1,
    ) -> None:
        self.max_steps = max_steps or int(os.getenv("GAIA_MAX_STEPS", "18"))
        self.cache_dir = Path(cache_dir)
        self.api_url = api_url
        self.verbosity_level = verbosity_level
        self.tools = build_tools()
        self.model = create_model()
        self._fallback_agent: CodeAgent | None = None
        self.agent = self._build_code_agent(self.model)

    def _build_code_agent(self, model: Any) -> CodeAgent:
        code_agent = CodeAgent(
            tools=self.tools,
            model=model,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            code_block_tags="markdown",
            executor_kwargs={"timeout_seconds": 120},
            max_steps=self.max_steps,
            max_print_outputs_length=20000,
            verbosity_level=self.verbosity_level,
            instructions=GAIA_SYSTEM_INSTRUCTIONS,
        )
        code_agent.prompt_templates = {
            **code_agent.prompt_templates,
            "final_answer": {
                "pre_messages": (
                    "The agent reached its step limit while solving a GAIA benchmark question. "
                    "Review the verified observations in memory and return ONLY the exact final answer."
                ),
                "post_messages": (
                    "Based strictly on the verified observations above, provide ONLY the final answer for:\n"
                    "{{task}}\n\n"
                    "OUTPUT RULES:\n"
                    "- Reply with ONLY the answer value (a number, as few words as possible, or comma-separated "
                    "list).\n"
                    "- Do NOT include explanations, sentences, Python code, 'Calling tools:', or "
                    "'FINAL ANSWER:' prefix.\n"
                    "- Numbers: plain digits, no commas or units unless explicitly asked.\n"
                    "- Strings: no articles (the, a, an), no abbreviations, no trailing period or quotes."
                ),
            },
        }
        return code_agent

    def _prepare_prompt(self, question: str, task_id: str = "", file_name: str = "") -> str:
        prompt = question.strip()
        if file_name and file_name.strip():
            local_path = fetch_task_file(
                task_id=task_id,
                file_name=file_name,
                api_url=self.api_url,
                cache_dir=self.cache_dir,
            )
            if local_path:
                prompt += (
                    f"\n\n[Attached file downloaded to local path: {local_path} "
                    f"(original file_name: {file_name}). Inspect this file first using "
                    "read_file_content, analyze_image, transcribe_audio, or Python code.]"
                )
            else:
                prompt += (
                    f"\n\n[Note: This question references attached file '{file_name}', "
                    "which could not be downloaded from the scoring server or Hugging Face Hub.]"
                )
        return prompt

    def _run_with_agent(self, agent: CodeAgent, prompt: str) -> str:
        executor = getattr(agent, "python_executor", None)
        if executor is not None:
            if hasattr(executor, "state") and isinstance(executor.state, dict):
                executor.state = {"__name__": "__main__"}
            if hasattr(executor, "custom_tools") and isinstance(executor.custom_tools, dict):
                executor.custom_tools = {}

        raw = agent.run(prompt)
        if isinstance(raw, str) and (
            "Calling tools:" in raw or re.match(r"^\s*```(?:python|py)\b", raw, re.IGNORECASE)
        ):
            logger.warning("Agent returned unparsed tool/code trace instead of final answer.")
            return ""
        return normalize_answer(raw)

    def __call__(self, question: str, task_id: str = "", file_name: str = "") -> str:
        prompt = self._prepare_prompt(question, task_id=task_id, file_name=file_name)
        last_exc: Exception | None = None

        for attempt in range(2):
            try:
                ans = self._run_with_agent(self.agent, prompt)
                if ans:
                    return ans
                last_exc = RuntimeError("Primary agent returned an empty answer.")
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("Primary agent attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    time.sleep(2.0)

        fallback_model = _create_fallback_model()
        if fallback_model is not None:
            if self._fallback_agent is None:
                self._fallback_agent = self._build_code_agent(fallback_model)
            try:
                ans = self._run_with_agent(self._fallback_agent, prompt)
                if ans:
                    return ans
            except Exception as exc:
                last_exc = exc
                logger.warning("Fallback agent failed: %s", exc)

        if last_exc is not None:
            raise last_exc
        return ""

    def ask_question(self, question: str, task_id: str = "", file_name: str = "") -> str:
        """Alias compatible with template forks calling `ask_question`."""
        return self(question, task_id=task_id, file_name=file_name)


# Drop-in alias for Final_Assignment_Template compatibility.
BasicAgent = GaiaAgent
