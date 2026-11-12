"""Unit tests for tools.py covering edge cases and error handling."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from PIL import Image

from tools import (
    _extract_youtube_id,
    _prepare_image_b64,
    _sample_video_frames,
    _search_ddg,
    _search_ddg_lite,
    analyze_image,
    read_file_content,
    transcribe_audio,
    visit_webpage,
    web_search,
    wikipedia_lookup,
    youtube_inspect,
)


class TestWebSearch:
    def test_empty_query_returns_error(self):
        assert "Error" in web_search("   ")

    def test_brave_used_when_key_set(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-secret")
        resp = Mock(status_code=200)
        resp.raise_for_status = Mock()
        resp.json.return_value = {
            "web": {"results": [{"title": "Alpha", "url": "https://a.test", "description": "desc"}]}
        }
        with patch("tools.requests.get", return_value=resp):
            out = web_search("test query", max_results=3)
        assert "[Alpha](https://a.test)" in out

    def test_falls_back_to_ddg_if_brave_fails(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-secret")
        with (
            patch("tools.requests.get", side_effect=RuntimeError("boom")),
            patch("tools._search_ddg", return_value=[{"title": "DDG", "url": "https://ddg.test", "snippet": "ok"}]),
        ):
            out = web_search("fallback query")
        assert "[DDG](https://ddg.test)" in out

    def test_ddg_uses_timeout_and_formats_results(self, monkeypatch):
        captured: dict[str, object] = {}

        class FakeDDGS:
            def __init__(self, timeout: int | None = None):
                captured["timeout"] = timeout

            def text(self, query: str, max_results: int = 5):
                return [{"title": "Dino", "href": "https://dino.test", "body": "found"}]

        monkeypatch.setitem(__import__("sys").modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
        results = _search_ddg("dinosaur FA", 3)
        assert captured["timeout"] == 6
        assert results == [{"title": "Dino", "url": "https://dino.test", "snippet": "found"}]

    def test_ddg_falls_back_when_client_lacks_timeout_kwarg(self, monkeypatch):
        class LegacyDDGS:
            def __init__(self):
                pass

            def text(self, query: str, max_results: int = 5):
                return [{"title": "Legacy", "url": "https://legacy.test", "snippet": "ok"}]

        monkeypatch.setitem(__import__("sys").modules, "ddgs", SimpleNamespace(DDGS=LegacyDDGS))
        results = _search_ddg("query", 2)
        assert results == [{"title": "Legacy", "url": "https://legacy.test", "snippet": "ok"}]

    def test_ddg_tries_yahoo_when_bing_empty_or_fails(self, monkeypatch):
        calls: list[str] = []

        class BackendDDGS:
            def __init__(self, timeout: int | None = None):
                pass

            def text(self, query: str, max_results: int = 5, backend: str = "auto"):
                calls.append(backend)
                if backend == "bing":
                    raise RuntimeError("bing blocked")
                if backend == "yahoo":
                    return [{"title": "YahooHit", "href": "https://y.test", "body": "ok"}]
                return []

        monkeypatch.setitem(__import__("sys").modules, "ddgs", SimpleNamespace(DDGS=BackendDDGS))
        results = _search_ddg("equine veterinarian", 3)
        assert calls == ["bing", "yahoo"]
        assert results == [{"title": "YahooHit", "url": "https://y.test", "snippet": "ok"}]

    def test_ddg_lite_parses_results_unwraps_uddg_and_respects_limit(self):
        html = """
        <table>
          <tr><td><a class="result-link" href="//duckduckgo.com/y.js?ad=1">Ad</a></td></tr>
          <tr><td class="result-snippet">Skip ad</td></tr>
          <tr><td>
            <a class="result-link" href="/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FDino">Dino FA</a>
          </td></tr>
          <tr><td class="result-snippet">Promoted in November 2016.</td></tr>
          <tr><td><a class="result-link" href="https://arxiv.org/abs/2307.02214">Paper</a></td></tr>
          <tr><td class="result-snippet">Abstract text.</td></tr>
          <tr><td><a class="result-link" href="https://extra.test">Extra</a></td></tr>
        </table>
        """
        resp = Mock(status_code=200, text=html)
        with patch("tools.requests.post", return_value=resp):
            out = _search_ddg_lite("dino", max_results=2)
        assert out == [
            {
                "title": "Dino FA",
                "url": "https://en.wikipedia.org/wiki/Dino",
                "snippet": "Promoted in November 2016.",
            },
            {
                "title": "Paper",
                "url": "https://arxiv.org/abs/2307.02214",
                "snippet": "Abstract text.",
            },
        ]

    def test_web_search_falls_back_to_ddg_lite_when_ddg_fails(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        lite_results = [{"title": "LiteHit", "url": "https://lite.test", "snippet": "worked"}]
        with (
            patch("tools._search_ddg", side_effect=RuntimeError("No results found.")),
            patch("tools._search_ddg_lite", return_value=lite_results),
        ):
            out = web_search("dinosaur promoted November 2016", max_results=3)
        assert "[LiteHit](https://lite.test)" in out


class TestVisitWebpage:
    def test_rejects_invalid_scheme(self):
        assert "Error" in visit_webpage("file:///etc/passwd")
        assert "Error" in visit_webpage("not a url")

    def test_html_resolves_relative_links_and_strips_scripts(self):
        html = """
        <html><head><script>alert(1)</script><style>body{}</style></head>
        <body><h1>Paper</h1><p>Link: <a href="/pdf/1234">Download PDF</a></p></body></html>
        """
        resp = Mock(status_code=200, url="https://example.org/article", text=html, content=html.encode("utf-8"))
        resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        resp.raise_for_status = Mock()

        with patch("tools.requests.get", return_value=resp):
            out = visit_webpage("https://example.org/article")
        assert "alert(1)" not in out
        assert "https://example.org/pdf/1234" in out

    def test_wikipedia_403_retries_with_wikimedia_ua(self):
        r403 = Mock(status_code=403)
        r200 = Mock(
            status_code=200,
            url="https://en.wikipedia.org/wiki/Test",
            text="<p>Hello</p>",
            content=b"<p>Hello</p>",
        )
        r200.headers = {"Content-Type": "text/html"}
        r200.raise_for_status = Mock()

        with patch("tools.requests.get", side_effect=[r403, r200]) as mock_get:
            out = visit_webpage("https://en.wikipedia.org/wiki/Test")
        assert "Hello" in out
        assert mock_get.call_count == 2

    def test_pdf_magic_bytes_detected_without_pdf_extension(self):
        pdf_bytes = b"%PDF-1.4 dummy"
        resp = Mock(status_code=200, url="https://arxiv.org/pdf/2306.0001", text="", content=pdf_bytes)
        resp.headers = {"Content-Type": "application/octet-stream"}
        resp.raise_for_status = Mock()

        with (
            patch("tools.requests.get", return_value=resp),
            patch("tools._extract_pdf_text", return_value="--- Page 1 ---\nAward NASA-80NSSC20K"),
        ):
            out = visit_webpage("https://arxiv.org/pdf/2306.0001", search_query="NASA")
        assert "NASA-80NSSC20K" in out

    def test_tail_pagination_with_negative_start_char(self):
        long_text = "A" * 500 + "END_MARKER"
        resp = Mock(status_code=200, url="https://example.org/txt", text=long_text, content=long_text.encode("utf-8"))
        resp.headers = {"Content-Type": "text/plain"}
        resp.raise_for_status = Mock()

        with patch("tools.requests.get", return_value=resp):
            out = visit_webpage("https://example.org/txt", start_char=-1, max_chars=500)
        assert "END_MARKER" in out

    def test_fixes_iso8859_default_encoding_and_rejects_binary_and_adds_tip(self):
        resp = Mock(
            status_code=200,
            url="https://example.org/jp",
            encoding="ISO-8859-1",
            apparent_encoding="utf-8",
            text="<html><body><p>horse doctor Louvrier</p></body></html>",
            content=b"<html><body><p>horse doctor Louvrier</p></body></html>",
        )
        resp.headers = {"Content-Type": "text/html"}
        resp.raise_for_status = Mock()

        with patch("tools.requests.get", return_value=resp):
            out = visit_webpage("https://example.org/jp", search_query="equine veterinarian")
        assert resp.encoding == "utf-8"
        assert "No matches found" in out
        assert "synonyms" in out

        bin_resp = Mock(status_code=200, url="https://example.org/a.png", content=b"\x89PNG\r\n")
        bin_resp.headers = {"Content-Type": "image/png"}
        bin_resp.raise_for_status = Mock()
        with patch("tools.requests.get", return_value=bin_resp):
            bin_out = visit_webpage("https://example.org/a.png")
        assert "Non-text binary content" in bin_out


class TestWikipediaLookup:
    def test_missing_title_falls_back_to_search(self):
        missing_rev = Mock(status_code=200)
        missing_rev.raise_for_status = Mock()
        missing_rev.json.return_value = {"query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}}

        search_resp = Mock(status_code=200)
        search_resp.raise_for_status = Mock()
        search_resp.json.return_value = {"query": {"search": [{"title": "Real Page", "snippet": "match"}]}}

        found_rev = Mock(status_code=200)
        found_rev.raise_for_status = Mock()
        rev_page = {"title": "Real Page", "revisions": [{"revid": 99, "timestamp": "2022-12-01T00:00:00Z"}]}
        found_rev.json.return_value = {"query": {"pages": {"1": rev_page}}}

        parse_resp = Mock(status_code=200)
        parse_resp.raise_for_status = Mock()
        parse_payload = {"title": "Real Page", "sections": [], "text": {"*": "<p>Content 2022</p>"}}
        parse_resp.json.return_value = {"parse": parse_payload}

        with patch("tools.requests.get", side_effect=[missing_rev, search_resp, found_rev, parse_resp]):
            out = wikipedia_lookup(query="", title="Missing", as_of_year=2022)
        assert "Real Page" in out
        assert "Content 2022" in out

    def test_missing_title_as_of_zero_falls_back_to_search_and_supports_search_query(self):
        missing_parse = Mock(status_code=200)
        missing_parse.raise_for_status = Mock()
        missing_parse.json.return_value = {"error": {"code": "missingtitle", "info": "doesn't exist"}}

        search_resp = Mock(status_code=200)
        search_resp.raise_for_status = Mock()
        search_resp.json.return_value = {"query": {"search": [{"title": "Real Article", "snippet": "ok"}]}}

        found_parse = Mock(status_code=200)
        found_parse.raise_for_status = Mock()
        found_parse.json.return_value = {
            "parse": {
                "title": "Real Article",
                "sections": [{"index": "1", "line": "<i>Cast</i>"}],
                "text": {"*": "<p>Alpha beta gamma Louvrier delta</p>"},
            }
        }
        sec_resp = Mock(status_code=200)
        sec_resp.raise_for_status = Mock()
        sec_resp.json.return_value = {"parse": {"text": {"*": "<p>Wojciech played a role.</p>"}}}

        with patch("tools.requests.get", side_effect=[missing_parse, search_resp, found_parse, sec_resp]):
            out = wikipedia_lookup(query="", title="WrongTitle", section="Cast", search_query="Wojciech")
        assert "Real Article" in out
        assert "Wojciech played a role." in out

    def test_prefers_exact_section_match_over_substring(self):
        parse_resp = Mock(status_code=200)
        parse_resp.raise_for_status = Mock()
        parse_resp.json.return_value = {
            "parse": {
                "title": "Topic",
                "sections": [
                    {"index": "1", "line": "Early history"},
                    {"index": "2", "line": "History"},
                ],
                "text": {"*": "<p>Full</p>"},
            }
        }
        sec_resp = Mock(status_code=200)
        sec_resp.raise_for_status = Mock()
        sec_resp.json.return_value = {"parse": {"text": {"*": "<p>Exact section body</p>"}}}

        with patch("tools.requests.get", side_effect=[parse_resp, sec_resp]) as mock_get:
            out = wikipedia_lookup(query="", title="Topic", section="History")
        assert "[Section: History]" in out
        assert "Exact section body" in out
        assert mock_get.call_args_list[1].kwargs["params"]["section"] == "2"


class TestFileTools:
    def test_read_xlsx_and_csv_and_missing(self, tmp_path: Path):
        assert "Error" in read_file_content(str(tmp_path / "nope.txt"))

        csv_path = tmp_path / "items.csv"
        pd.DataFrame({"item": ["Burger", "Soda"], "sales": [10.5, 2.5]}).to_csv(csv_path, index=False)
        assert "Burger" in read_file_content(str(csv_path))

        xlsx_path = tmp_path / "sales.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame({"item": ["Fries"], "sales": [4.25]}).to_excel(writer, sheet_name="Food", index=False)
        xlsx_out = read_file_content(str(xlsx_path))
        assert "Sheet: Food" in xlsx_out
        assert "4.25" in xlsx_out
        assert "4.25" in read_file_content(str(xlsx_path), sheet_name="food")

        latin_csv = tmp_path / "latin.csv"
        latin_csv.write_bytes("name,city\nJosé,São Paulo\n".encode("latin-1"))
        latin_out = read_file_content(str(latin_csv))
        assert "José" in latin_out
        assert "São Paulo" in latin_out

    def test_read_file_search_query_and_basename_fallback(self, tmp_path: Path, monkeypatch):
        cache_dir = tmp_path / ".cache" / "gaia_files"
        cache_dir.mkdir(parents=True)
        (cache_dir / "notes.txt").write_text("alpha\nbeta Louvrier\ngamma", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        out = read_file_content("notes.txt", search_query="Louvrier")
        assert "Louvrier" in out

    def test_prepare_image_upscales_tiny_rgba(self, tmp_path: Path):
        img_path = tmp_path / "tiny.png"
        Image.new("RGBA", (32, 48), (255, 0, 0, 128)).save(img_path)
        b64 = _prepare_image_b64(img_path, min_side=192)
        import base64

        with Image.open(io.BytesIO(base64.b64decode(b64))) as out_img:
            assert min(out_img.size) >= 192
            assert out_img.mode == "RGB"

    def test_prepare_image_preserves_crisp_glyph_pixels_and_aspect_limits(self, tmp_path: Path):
        import base64

        # Small 64x24 dark-on-light glyph pixel: 2x NEAREST + padding keeps exact black/white without blur.
        tiny_path = tmp_path / "glyph.png"
        im = Image.new("RGB", (64, 24), (255, 255, 255))
        im.putpixel((20, 12), (0, 0, 0))
        im.save(tiny_path)
        b64 = _prepare_image_b64(tiny_path, min_side=192)
        with Image.open(io.BytesIO(base64.b64decode(b64))) as out_img:
            assert min(out_img.size) >= 192
            colors = {c for _, c in (out_img.getcolors() or [])}
            assert colors == {(255, 255, 255), (0, 0, 0)}

        # Extreme wide strip (> max_side and < min_side) respects both bounds simultaneously.
        wide_path = tmp_path / "wide.png"
        Image.new("P", (3000, 50), 0).save(wide_path, transparency=0)
        b64_wide = _prepare_image_b64(wide_path, min_side=192, max_side=2048)
        with Image.open(io.BytesIO(base64.b64decode(b64_wide))) as out_wide:
            assert min(out_wide.size) >= 192
            assert max(out_wide.size) <= 2048
            assert out_wide.mode == "RGB"

    def test_analyze_image_missing_file(self, tmp_path: Path):
        assert "Error" in analyze_image(str(tmp_path / "missing.png"))

    def test_transcribe_audio_missing_file(self, tmp_path: Path):
        assert "Error" in transcribe_audio(str(tmp_path / "missing.mp3"))

    def test_transcribe_audio_openai_path(self, tmp_path: Path, monkeypatch):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"ID3")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        resp = Mock(status_code=200)
        resp.raise_for_status = Mock()
        resp.json.return_value = {"text": "strawberries, sugar"}
        with patch("tools.requests.post", return_value=resp):
            assert transcribe_audio(str(audio)) == "strawberries, sugar"

    def test_transcribe_audio_reports_no_speech_after_failed_openai_fallback(self, tmp_path: Path, monkeypatch):
        audio = tmp_path / "silent.wav"
        audio.write_bytes(b"RIFF")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
        fake_whisper = Mock()
        fake_whisper.transcribe.return_value = ([], None)
        with (
            patch("tools.requests.post", side_effect=RuntimeError("401 Unauthorized")),
            patch("tools._get_whisper_model", return_value=fake_whisper),
        ):
            assert transcribe_audio(str(audio)) == "No speech detected in audio."

    def test_prepare_image_applies_exif_transpose(self, tmp_path: Path):
        import base64

        img_path = tmp_path / "rotated.jpg"
        im = Image.new("RGB", (300, 100), (255, 255, 255))
        exif = im.getexif()
        exif[0x0112] = 6  # Rotate 90 CW / 270 CCW to restore portrait 100x300
        im.save(img_path, exif=exif)
        b64 = _prepare_image_b64(img_path, min_side=192)
        with Image.open(io.BytesIO(base64.b64decode(b64))) as out_img:
            w, h = out_img.size
            assert h > w


class TestYoutubeInspect:
    def test_extract_youtube_id_variants(self):
        assert _extract_youtube_id("https://www.youtube.com/watch?v=1htKBjuUWec") == "1htKBjuUWec"
        assert _extract_youtube_id("https://youtu.be/L1vXCYZAYYM?t=5") == "L1vXCYZAYYM"
        assert _extract_youtube_id("https://www.youtube.com/shorts/AbCdEf12345") == "AbCdEf12345"
        assert _extract_youtube_id("https://example.com/") == ""

    def test_invalid_url_returns_error(self):
        assert "Error" in youtube_inspect("https://example.com/no-video")

    def test_sample_video_frames_uses_dash_compatible_format(self, monkeypatch):
        import numpy as np

        captured: dict[str, object] = {}

        class FakeYDL:
            def __init__(self, opts):
                captured["opts"] = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def download(self, urls):
                out_path = Path(str(captured["opts"]["outtmpl"]).replace("%(ext)s", "mp4"))
                out_path.write_bytes(b"fake-video")

        class FakeCap:
            def __init__(self, _path):
                self.released = False

            def get(self, _prop):
                return 4

            def set(self, _prop, _idx):
                return True

            def read(self):
                return True, np.zeros((16, 16, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        fake_cv2 = SimpleNamespace(
            CAP_PROP_FRAME_COUNT=7,
            CAP_PROP_POS_FRAMES=1,
            CAP_PROP_FPS=5,
            COLOR_BGR2RGB=4,
            VideoCapture=FakeCap,
            cvtColor=lambda frame, _code: frame,
        )
        sys_mod = __import__("sys").modules
        monkeypatch.setitem(sys_mod, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
        monkeypatch.setitem(sys_mod, "cv2", fake_cv2)

        frames = _sample_video_frames(
            "https://www.youtube.com/watch?v=L1vXCYZAYYM",
            num_frames=2,
            start_sec=0.0,
            end_sec=0.5,
        )
        assert "bestvideo" in str(captured["opts"]["format"])
        assert len(frames) == 2

    def test_call_vlm_uses_jpeg_mime_and_8192_tokens(self, monkeypatch):
        from tools import _call_vlm

        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        resp = Mock(status_code=200)
        resp.raise_for_status = Mock()
        resp.json.return_value = {"choices": [{"message": {"content": "3"}}]}
        with patch("tools.requests.post", return_value=resp) as mock_post:
            ans = _call_vlm(["/9j/4AAQSkZJRg=="], "Count")
        assert ans == "3"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["max_tokens"] == 8192
        assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_youtube_inspect_includes_description_in_vlm_context_and_no_search_hint(self):
        meta = {"title": "Penguin Chicks", "author": "BBC", "duration": 121, "description": "Emperor and Adelie"}
        long_transcript = "[0.0s] " + ("hello " * 4000)
        with (
            patch("tools._fetch_youtube_metadata", return_value=meta),
            patch("tools._fetch_youtube_transcript", return_value=long_transcript),
            patch("tools._sample_video_frames", return_value=["/9j/abc"]),
            patch("tools._call_vlm", return_value="3") as mock_vlm,
        ):
            out = youtube_inspect(
                "https://www.youtube.com/watch?v=L1vXCYZAYYM",
                extract_frames=True,
                num_frames=12,
                question="How many species?",
                start_sec=10.0,
                end_sec=60.0,
            )
        assert "Description:\nEmperor and Adelie" in out
        assert "search_query=" not in out
        assert "Emperor and Adelie" in mock_vlm.call_args.args[1]
