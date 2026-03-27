# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest

from copaw.local_models.manager import LocalModelManager


class _FakeLlamaCppBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    def check_llamacpp_installation(self) -> bool:
        self.calls.append(("check", None))
        return True

    def download(self) -> None:
        self.calls.append(("download", None))

    def get_download_progress(self) -> dict[str, object]:
        self.calls.append(("progress", None))
        return {"status": "downloading"}

    def cancel_download(self) -> None:
        self.calls.append(("cancel", None))

    async def setup_server(self, model_path: Path, model_name: str) -> int:
        self.calls.append(("setup", (model_path, model_name)))
        return 8080

    async def shutdown_server(self) -> None:
        self.calls.append(("shutdown", None))


class _FakeModelManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []

    def get_recommended_models(self) -> list[str]:
        self.calls.append(("recommended", None))
        return ["demo-model"]

    def is_downloaded(self, model_name: str) -> bool:
        self.calls.append(("is_downloaded", model_name))
        return model_name == "downloaded-model"

    def download_model(self, model_name: str) -> None:
        self.calls.append(("download_model", model_name))

    def get_download_progress(self) -> dict[str, object]:
        self.calls.append(("progress", None))
        return {"status": "pending"}

    def cancel_download(self) -> None:
        self.calls.append(("cancel", None))


def test_local_model_manager_forwards_sync_calls() -> None:
    fake_model_manager = _FakeModelManager()
    fake_llamacpp_backend = _FakeLlamaCppBackend()
    manager = LocalModelManager(
        model_manager=fake_model_manager,
        llamacpp_backend=fake_llamacpp_backend,
    )

    assert manager.check_llamacpp_installation() is True
    manager.start_llamacpp_download()
    assert manager.get_llamacpp_download_progress() == {
        "status": "downloading",
    }
    manager.cancel_llamacpp_download()

    assert manager.get_recommended_models() == ["demo-model"]
    assert manager.is_model_downloaded("downloaded-model") is True
    manager.start_model_download("demo-model")
    assert manager.get_model_download_progress() == {"status": "pending"}
    manager.cancel_model_download()

    assert fake_llamacpp_backend.calls == [
        ("check", None),
        ("download", None),
        ("progress", None),
        ("cancel", None),
    ]
    assert fake_model_manager.calls == [
        ("recommended", None),
        ("is_downloaded", "downloaded-model"),
        ("download_model", "demo-model"),
        ("progress", None),
        ("cancel", None),
    ]


@pytest.mark.asyncio
async def test_local_model_manager_forwards_async_server_calls(
    tmp_path: Path,
) -> None:
    fake_llamacpp_backend = _FakeLlamaCppBackend()
    manager = LocalModelManager(
        model_manager=_FakeModelManager(),
        llamacpp_backend=fake_llamacpp_backend,
    )

    port = await manager.setup_server(tmp_path / "model.gguf", "demo")
    await manager.shutdown_server()

    assert port == 8080
    assert fake_llamacpp_backend.calls == [
        ("setup", (tmp_path / "model.gguf", "demo")),
        ("shutdown", None),
    ]
