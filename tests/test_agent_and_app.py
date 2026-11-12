"""Unit and contract tests for agent.py and app.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

import app
import eval_local
from agent import BasicAgent, GaiaAgent, create_model


class TestModelSelection:
    def test_prefers_nvidia_nim_when_nvidia_key_set(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GAIA_PROVIDER", raising=False)
        model = create_model()
        assert model.model_id == "nvidia/nemotron-3-ultra-550b-a55b"
        assert model.client_kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"

    def test_uses_openai_when_only_openai_key_set(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        model = create_model()
        assert model.model_id == "gpt-4o"

    def test_falls_back_to_hf_inference_without_keys(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf-test")
        model = create_model()
        assert "Qwen" in model.model_id


class TestGaiaAgent:
    def test_basic_agent_alias(self):
        assert BasicAgent is GaiaAgent

    def test_attaches_downloaded_file_and_normalizes_answer(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        fake_file = tmp_path / "sample.py"
        fake_file.write_text("print(7)", encoding="utf-8")

        ga = GaiaAgent(cache_dir=tmp_path)
        with (
            patch("agent.fetch_task_file", return_value=str(fake_file)),
            patch.object(ga.agent, "run", return_value="FINAL ANSWER: 7.") as mock_run,
        ):
            ans = ga("What is output?", task_id="t1", file_name="sample.py")

        assert ans == "7"
        prompt_sent = mock_run.call_args.args[0]
        assert str(fake_file) in prompt_sent

    def test_retries_and_uses_fallback_model(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        ga = GaiaAgent(cache_dir=tmp_path)

        with (
            patch("agent.time.sleep"),
            patch.object(ga.agent, "run", side_effect=RuntimeError("rate limit")),
            patch.object(ga, "_build_code_agent") as mock_build_fb,
        ):
            fb_agent = Mock()
            fb_agent.run.return_value = "Saint Petersburg"
            mock_build_fb.return_value = fb_agent
            ans = ga("City?")

        assert ans == "Saint Petersburg"

    def test_configures_gaia_final_answer_template_and_executor_timeout(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        ga = GaiaAgent(cache_dir=tmp_path)
        assert ga.agent.python_executor.timeout_seconds == 120
        assert "OUTPUT RULES" in ga.agent.prompt_templates["final_answer"]["post_messages"]

    def test_rejects_tool_call_leak_and_uses_fallback(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        ga = GaiaAgent(cache_dir=tmp_path)
        leaked = "```python\nx = 1\n```\nCalling tools: [{'id': 'call_1'}]"

        with (
            patch("agent.time.sleep"),
            patch.object(ga.agent, "run", return_value=leaked) as mock_primary,
            patch.object(ga, "_build_code_agent") as mock_build_fb,
        ):
            fb_agent = Mock()
            fb_agent.run.return_value = "Louvrier"
            mock_build_fb.return_value = fb_agent
            ans = ga("Equine veterinarian?")

        assert ans == "Louvrier"
        # Deterministic empty/leak output should skip redundant second primary attempt.
        assert mock_primary.call_count == 1

    def test_resets_python_executor_state_and_allows_fenced_scalar_answer(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        ga = GaiaAgent(cache_dir=tmp_path)
        ga.agent.python_executor.state["leaked_var"] = 999
        ga.agent.python_executor.custom_tools["leaked_fn"] = lambda: 1

        def check_clean_and_return(_prompt):
            assert "leaked_var" not in ga.agent.python_executor.state
            assert ga.agent.python_executor.custom_tools == {}
            return "```42```"

        with patch.object(ga.agent, "run", side_effect=check_clean_and_return):
            ans = ga("What is 6*7?")
        assert ans == "42"

        with (
            patch("agent.time.sleep"),
            patch.object(ga.agent, "run", return_value="```python\nx = 1\n```"),
            patch.object(ga, "_build_code_agent") as mock_build_fb,
        ):
            fb = Mock()
            fb.run.return_value = "42"
            mock_build_fb.return_value = fb
            assert ga("What is 6*7?") == "42"


class TestGradioAppSubmission:
    def test_requires_oauth_login(self):
        updates = list(app.run_and_submit_all(profile=None, use_cache=False))
        assert len(updates) == 1
        status, df = updates[-1]
        assert "Login" in status
        assert df is None

    def test_runs_questions_streams_progress_caches_and_submits_clean_payload(self, tmp_path: Path, monkeypatch):
        cache_path = tmp_path / "answers_cache.json"
        monkeypatch.setattr(app, "CACHE_FILE", cache_path)
        monkeypatch.setenv("SPACE_ID", "harasubhojit/gaia-agent")

        questions = [
            {"task_id": "q1", "question": "Reverse left", "file_name": ""},
            {"task_id": "q2", "question": "Read file", "file_name": "data.xlsx"},
        ]

        q_resp = Mock(status_code=200)
        q_resp.raise_for_status = Mock()
        q_resp.json.return_value = questions

        submit_resp = Mock(status_code=200)
        submit_resp.raise_for_status = Mock()
        submit_resp.json.return_value = {
            "username": "harasubhojit",
            "score": 100.0,
            "correct_count": 2,
            "total_attempted": 2,
            "message": "ok",
        }

        fake_agent = Mock()
        fake_agent.side_effect = ["FINAL ANSWER: right", "89706.00"]

        with (
            patch("app.GaiaAgent", return_value=fake_agent),
            patch("app.requests.get", return_value=q_resp),
            patch("app.requests.post", return_value=submit_resp) as mock_post,
        ):
            profile = SimpleNamespace(username="harasubhojit")
            updates = list(app.run_and_submit_all(profile=profile, use_cache=True))

        assert len(updates) >= 7
        assert any("[1/2] Running agent" in st for st, _ in updates)
        assert any("[2/2] Running agent" in st for st, _ in updates)
        assert any(df is not None and len(df) == 1 for _, df in updates)

        status, df = updates[-1]
        assert "100.0%" in status
        assert len(df) == 2
        payload = mock_post.call_args.kwargs["json"]
        assert payload["username"] == "harasubhojit"
        assert payload["agent_code"] == "https://huggingface.co/spaces/harasubhojit/gaia-agent/tree/main"
        assert payload["answers"] == [
            {"task_id": "q1", "submitted_answer": "right"},
            {"task_id": "q2", "submitted_answer": "89706.00"},
        ]

        # Second run with cache enabled should NOT invoke agent again.
        fake_agent.reset_mock()
        with (
            patch("app.GaiaAgent", return_value=fake_agent),
            patch("app.requests.get", return_value=q_resp),
            patch("app.requests.post", return_value=submit_resp),
        ):
            updates2 = list(app.run_and_submit_all(profile=profile, use_cache=True))
        status2, df2 = updates2[-1]
        assert "100.0%" in status2
        assert list(df2["Source"]) == ["cache", "cache"]
        fake_agent.assert_not_called()
        assert cache_path.is_file()
        assert "clear" in app.clear_cache().lower()
        assert not cache_path.exists()

    def test_incremental_cache_persists_partial_run_before_crash(self, tmp_path: Path, monkeypatch):
        cache_path = tmp_path / "answers_cache.json"
        monkeypatch.setattr(app, "CACHE_FILE", cache_path)
        monkeypatch.setenv("SPACE_ID", "harasubhojit/gaia-agent")

        questions = [
            {"task_id": "q1", "question": "Q1", "file_name": ""},
            {"task_id": "q2", "question": "Q2", "file_name": ""},
        ]
        q_resp = Mock(status_code=200)
        q_resp.raise_for_status = Mock()
        q_resp.json.return_value = questions

        submit_resp = Mock(status_code=200)
        submit_resp.raise_for_status = Mock()
        submit_resp.json.return_value = {
            "username": "harasubhojit",
            "score": 50.0,
            "correct_count": 1,
            "total_attempted": 1,
            "message": "ok",
        }

        fake_agent = Mock(side_effect=["42", RuntimeError("boom")])
        with (
            patch("app.GaiaAgent", return_value=fake_agent),
            patch("app.requests.get", return_value=q_resp),
            patch("app.requests.post", return_value=submit_resp),
        ):
            profile = SimpleNamespace(username="harasubhojit")
            updates = list(app.run_and_submit_all(profile=profile, use_cache=True))

        assert app._load_answer_cache(cache_path) == {"q1": "42"}
        _, final_df = updates[-1]
        assert list(final_df["Source"]) == ["agent", "error"]

    def test_build_demo_local_and_space_oauth_modes_with_queue(self, monkeypatch):
        demo_local = app.build_demo(enable_oauth=False)
        assert getattr(demo_local, "_queue", None) is not None
        assert not any(isinstance(c, gr.LoginButton) for c in demo_local.blocks.values())

        monkeypatch.setenv("SYSTEM", "spaces")
        monkeypatch.setenv("SPACE_ID", "harasubhojit/gaia-agent")
        monkeypatch.setenv("OAUTH_CLIENT_ID", "cid")
        monkeypatch.setenv("OAUTH_CLIENT_SECRET", "csec")
        monkeypatch.setenv("OAUTH_SCOPES", "openid profile")
        monkeypatch.setenv("OPENID_PROVIDER_URL", "https://huggingface.co")
        import gradio.oauth as go

        monkeypatch.setattr(go, "OAUTH_CLIENT_ID", "cid")
        monkeypatch.setattr(go, "OAUTH_CLIENT_SECRET", "csec")
        monkeypatch.setattr(go, "OAUTH_SCOPES", "openid profile")
        monkeypatch.setattr(go, "OPENID_PROVIDER_URL", "https://huggingface.co")
        demo_space = app.build_demo(enable_oauth=True)
        assert getattr(demo_space, "_queue", None) is not None
        assert any(isinstance(c, gr.LoginButton) for c in demo_space.blocks.values())

    def test_has_hf_oauth_context_returns_false_if_whoami_fails(self, monkeypatch):
        monkeypatch.delenv("SYSTEM", raising=False)
        monkeypatch.delenv("SPACE_ID", raising=False)
        with (
            patch("huggingface_hub.get_token", return_value="hf_bad"),
            patch("huggingface_hub.whoami", side_effect=RuntimeError("offline")),
        ):
            assert app._has_hf_oauth_context() is False


class TestEvalLocal:
    def test_saves_incrementally_and_resumes_completed_tasks(self, tmp_path: Path):
        out_file = tmp_path / "results.json"
        questions = [
            {"task_id": "t1", "file_name": "", "question": "Q1"},
            {"task_id": "t2", "file_name": "", "question": "Q2"},
        ]
        q_resp = Mock(status_code=200)
        q_resp.json.return_value = questions

        # First run: t1 succeeds, t2 fails with exception -> t1 should already be saved incrementally.
        agent1 = Mock(side_effect=["42", RuntimeError("timeout")])
        with (
            patch("eval_local.requests.get", return_value=q_resp),
            patch("eval_local.GaiaAgent", return_value=agent1),
        ):
            rc1 = eval_local.main(["--output", str(out_file)])

        assert rc1 == 0
        saved1 = eval_local._load_completed_results(out_file)
        assert list(saved1.keys()) == ["t1"]
        assert saved1["t1"]["answer"] == "42"

        # Second run with --resume: skips t1, only evaluates t2.
        agent2 = Mock(return_value="99")
        with (
            patch("eval_local.requests.get", return_value=q_resp),
            patch("eval_local.GaiaAgent", return_value=agent2),
        ):
            rc2 = eval_local.main(["--output", str(out_file), "--resume"])

        assert rc2 == 0
        agent2.assert_called_once_with("Q2", task_id="t2", file_name="")
        import json

        final_rows = json.loads(out_file.read_text(encoding="utf-8"))
        assert [(r["task_id"], r["answer"], r["error"]) for r in final_rows] == [
            ("t1", "42", ""),
            ("t2", "99", ""),
        ]

    def test_resume_preserves_unreached_and_filtered_completed_tasks(self, tmp_path: Path):
        import json

        out_file = tmp_path / "results.json"
        existing = [
            {"task_id": "t1", "file_name": "", "question": "Q1", "answer": "a1", "error": "", "elapsed_s": 1.0},
            {"task_id": "t2", "file_name": "", "question": "Q2", "answer": "a2", "error": "", "elapsed_s": 1.0},
        ]
        out_file.write_text(json.dumps(existing), encoding="utf-8")

        questions = [
            {"task_id": "t1", "file_name": "", "question": "Q1"},
            {"task_id": "t2", "file_name": "", "question": "Q2"},
            {"task_id": "t3", "file_name": "", "question": "Q3"},
        ]
        q_resp = Mock(status_code=200)
        q_resp.json.return_value = questions

        # Run only t3 with --resume -> t1 and t2 must still remain preserved in out_file!
        agent3 = Mock(return_value="a3")
        with (
            patch("eval_local.requests.get", return_value=q_resp),
            patch("eval_local.GaiaAgent", return_value=agent3),
        ):
            rc = eval_local.main(["--output", str(out_file), "--task-ids", "t3", "--resume"])

        assert rc == 0
        final_rows = json.loads(out_file.read_text(encoding="utf-8"))
        assert [(r["task_id"], r["answer"]) for r in final_rows] == [
            ("t1", "a1"),
            ("t2", "a2"),
            ("t3", "a3"),
        ]
