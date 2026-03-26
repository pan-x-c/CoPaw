# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from copaw.local_models.model_downloader import ModelDownloader
from copaw.local_models.schema import DownloadSource


class _FakeProcess:
    def __init__(self) -> None:
        self._alive = True
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def join(self, timeout=None) -> None:
        del timeout


def test_get_recommended_models_returns_expected_ranges(monkeypatch) -> None:
    downloader = ModelDownloader()

    monkeypatch.setattr(downloader, "_detect_available_memory_gb", lambda: 3.5)
    assert downloader.get_recommended_models() is None

    monkeypatch.setattr(downloader, "_detect_available_memory_gb", lambda: 4.0)
    model_4 = downloader.get_recommended_models()
    assert model_4 is not None
    assert len(model_4) == 2
    assert model_4[0].name == "CoPaw-2B"
    assert model_4[1].name == "CoPaw-2B"

    monkeypatch.setattr(downloader, "_detect_available_memory_gb", lambda: 6.0)
    model_6 = downloader.get_recommended_models()
    assert model_6 is not None
    assert len(model_6) == 2
    assert model_6[0].name == "CoPaw-2B"
    assert model_6[1].name == "CoPaw-2B"

    monkeypatch.setattr(downloader, "_detect_available_memory_gb", lambda: 8.0)
    model_8 = downloader.get_recommended_models()
    assert model_8 is not None
    assert len(model_8) == 2
    assert model_8[0].name == "CoPaw-2B"
    assert model_8[1].name == "CoPaw-2B"

    monkeypatch.setattr(
        downloader,
        "_detect_available_memory_gb",
        lambda: 12.0,
    )

    model_12 = downloader.get_recommended_models()
    assert model_12 is not None
    assert len(model_12) == 2
    assert model_12[0].name == "CoPaw-4B"
    assert model_12[1].name == "CoPaw-4B"

    monkeypatch.setattr(
        downloader,
        "_detect_available_memory_gb",
        lambda: 16.0,
    )
    model_16 = downloader.get_recommended_models()
    assert model_16 is not None
    assert len(model_16) == 2
    assert model_16[0].name == "CoPaw-4B"
    assert model_16[1].name == "CoPaw-4B"

    monkeypatch.setattr(
        downloader,
        "_detect_available_memory_gb",
        lambda: 24.0,
    )
    model_24 = downloader.get_recommended_models()
    assert model_24 is not None
    assert len(model_24) == 2
    assert model_24[0].name == "CoPaw-9B"
    assert model_24[1].name == "CoPaw-9B"


def test_download_model_uses_reachable_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    downloader = ModelDownloader()
    captured = {}
    target_dir = tmp_path / "resolved-model-dir"

    monkeypatch.setattr(
        downloader,
        "_resolve_download_source",
        lambda: captured.setdefault("source", DownloadSource.MODELSCOPE),
    )
    monkeypatch.setattr(
        downloader,
        "_estimate_download_size",
        lambda **kwargs: 100,
    )

    class _FakeQueue:
        pass

    class _FakeContext:
        def Queue(self):
            return _FakeQueue()

        def Process(self, **kwargs):
            captured["process_kwargs"] = kwargs

            class _Process:
                def start(self):
                    captured["started"] = True

                def is_alive(self):
                    return True

            return _Process()

    downloader.__dict__["_context"] = _FakeContext()

    class _FakeThread:
        def __init__(self, **kwargs):
            captured["thread_kwargs"] = kwargs

        def start(self):
            captured["thread_started"] = True

    monkeypatch.setattr(
        "copaw.local_models.model_downloader.threading.Thread",
        _FakeThread,
    )

    downloader.download_model(
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        target_dir=target_dir,
    )

    assert captured["source"] == DownloadSource.MODELSCOPE
    assert captured["started"] is True
    assert downloader.get_download_progress()["source"] == "modelscope"
    assert downloader.__dict__["_final_dir"] == target_dir.resolve()


def test_get_download_progress_returns_idle_by_default() -> None:
    downloader = ModelDownloader()

    assert downloader.get_download_progress() == {
        "status": "idle",
        "downloaded_bytes": 0,
        "total_bytes": None,
        "speed_bytes_per_sec": 0.0,
        "source": None,
        "error": None,
        "local_path": None,
    }


def test_cancel_download_stops_active_process(tmp_path: Path) -> None:
    downloader = ModelDownloader()
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / "partial.gguf").write_bytes(b"123")
    fake_process = _FakeProcess()

    downloader.__dict__["_process"] = fake_process
    downloader.__dict__["_staging_dir"] = staging_dir
    downloader.__dict__["_progress"] = {
        "status": "downloading",
        "downloaded_bytes": 3,
        "total_bytes": 10,
        "speed_bytes_per_sec": 1.0,
        "source": "huggingface",
        "error": None,
        "local_path": None,
    }

    downloader.cancel_download()

    progress = downloader.get_download_progress()
    assert fake_process.terminated is True
    assert not staging_dir.exists()
    assert progress["status"] == "cancelled"
    assert progress["speed_bytes_per_sec"] == 0.0
