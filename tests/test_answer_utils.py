"""Unit tests for answer_utils."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from answer_utils import fetch_task_file, normalize_answer


class TestNormalizeAnswer:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, ""),
            ("", ""),
            ("   ", ""),
            (42, "42"),
            (0, "0"),
            (3.14, "3.14"),
            (float("nan"), ""),
            (float("inf"), ""),
            (True, "True"),
            (["broccoli", "celery", "lettuce"], "broccoli, celery, lettuce"),
            (("a", "b"), "a, b"),
            ({"final_answer": "FINAL ANSWER: 1200"}, "1200"),
            ("FINAL ANSWER: right", "right"),
            ("final answer: [New York]", "New York"),
            ("**FINAL ANSWER: 42**", "42"),
            ("Here is it: **FINAL ANSWER: 42**", "42"),
            ('FINAL ANSWER: "New York".', "New York"),
            ("```42```", "42"),
            ("FINAL ANSWER:\n42\nBecause I checked Wikipedia.", "42"),
            ("Thought: blabla\nFINAL ANSWER: **Saint Petersburg**", "Saint Petersburg"),
            ("`42`", "42"),
            ('"hello world"', "hello world"),
            ("'single'", "single"),
            ("right.", "right"),
            ("Dr.", "Dr."),
            ("89706.00", "89706.00"),
            ("line1\n  line2", "line1 line2"),
            ("1,200", "1200"),
            ("1,234,567", "1234567"),
            ("-1,200.50", "-1200.50"),
            ("89,706.00", "89706.00"),
            ("FINAL ANSWER: 1,200.", "1200"),
            ("1,200.50.", "1200.50"),
            ("1, 200", "1, 200"),
            ("1,200, 3,400", "1200, 3400"),
            ("b,e", "b, e"),
            ("Yoshida,Uehara", "Yoshida, Uehara"),
            ("a , b ,c", "a, b, c"),
            ("1,2,3", "1, 2, 3"),
            ("0,1", "0, 1"),
            ("Saint Petersburg, Russia", "Saint Petersburg, Russia"),
            ("Dr., Mr.", "Dr., Mr."),
            ("**FINAL ANSWER:** 42", "42"),
            ("**FINAL ANSWER:**\n42", "42"),
            ("**FINAL ANSWER**: 42", "42"),
            ("__FINAL ANSWER:__ New York", "New York"),
            ("*FINAL ANSWER:* 42", "42"),
            ("Dr. Smith.", "Dr. Smith"),
            ("St. Petersburg.", "St. Petersburg"),
            ("J. R. R. Tolkien.", "J. R. R. Tolkien"),
            ("U.S.", "U.S."),
            ("Ph.D.", "Ph.D."),
            ("John F.", "John F."),
            ("A.", "A"),
            ("apple,1,200", "apple, 1200"),
            ("1,200,apple", "1200, apple"),
            ("1,200,3,400", "1200, 3400"),
            ("$1,200", "$1,200"),
            ("a, b,", "a, b"),
        ],
    )
    def test_normalize_cases(self, raw, expected):
        assert normalize_answer(raw) == expected

    def test_always_returns_str(self):
        for sample in [None, 1, 2.5, ["x", 2], {"answer": 7}, "FINAL ANSWER: ok"]:
            result = normalize_answer(sample)
            assert isinstance(result, str)


class TestFetchTaskFile:
    def test_returns_none_when_empty_filename(self, tmp_path: Path):
        assert fetch_task_file("t1", "", cache_dir=tmp_path) is None
        assert fetch_task_file("t1", "   ", cache_dir=tmp_path) is None

    def test_uses_cached_file_without_network(self, tmp_path: Path):
        cached = tmp_path / "data.xlsx"
        cached.write_bytes(b"abc123")
        with patch("answer_utils.requests.get") as mock_get:
            path = fetch_task_file("t1", "sub/dir/data.xlsx", cache_dir=tmp_path)
            assert path == str(cached.resolve())
            mock_get.assert_not_called()

    def test_downloads_from_scoring_api_first(self, tmp_path: Path):
        resp = Mock(status_code=200, content=b"print(42)")
        with patch("answer_utils.requests.get", return_value=resp) as mock_get:
            path = fetch_task_file("task-123", "code.py", api_url="https://example.test", cache_dir=tmp_path)
        assert path == str((tmp_path / "code.py").resolve())
        assert Path(path).read_bytes() == b"print(42)"
        mock_get.assert_called_once_with("https://example.test/files/task-123", timeout=20)

    def test_falls_back_to_hf_hub_on_404(self, tmp_path: Path):
        resp_404 = Mock(status_code=404, content=b"")
        hub_file = tmp_path / "hub_download.png"
        hub_file.write_bytes(b"PNGDATA")

        with (
            patch("answer_utils.requests.get", return_value=resp_404),
            patch("huggingface_hub.hf_hub_download", return_value=str(hub_file)) as mock_hub,
        ):
            path = fetch_task_file("task-456", "board.png", cache_dir=tmp_path / "cache", hf_token="hf_test")

        assert path is not None
        assert Path(path).read_bytes() == b"PNGDATA"
        mock_hub.assert_called_once()

    def test_returns_none_when_api_404_and_no_hf_token(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
        resp_404 = Mock(status_code=404, content=b"")
        with (
            patch("answer_utils.requests.get", return_value=resp_404),
            patch("huggingface_hub.get_token", return_value=None),
        ):
            assert fetch_task_file("task-456", "board.png", cache_dir=tmp_path) is None

    def test_uses_cli_cached_hf_token_when_env_unset(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
        resp_404 = Mock(status_code=404, content=b"")
        hub_file = tmp_path / "hub.xlsx"
        hub_file.write_bytes(b"XLSX")
        with (
            patch("answer_utils.requests.get", return_value=resp_404),
            patch("huggingface_hub.get_token", return_value="hf_cli_tok"),
            patch("huggingface_hub.hf_hub_download", return_value=str(hub_file)) as mock_hub,
        ):
            out = fetch_task_file("task-789", "hub.xlsx", cache_dir=tmp_path / "cache")
        assert out is not None
        mock_hub.assert_called_once()
        assert mock_hub.call_args.kwargs["token"] == "hf_cli_tok"
