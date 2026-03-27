# -*- coding: utf-8 -*-
"""Local model management and inference."""

from .download_manager import (
    apply_download_result,
    begin_download_task,
    DownloadProgress,
    DownloadTaskResult,
    DownloadTaskStatus,
)
from .schema import (
    BackendType,
    DownloadSource,
    LocalModelInfo,
)
from .manager import LocalModelManager
from .model_manager import ModelManager
from .llamacpp import LlamaCppBackend

__all__ = [
    "BackendType",
    "DownloadSource",
    "apply_download_result",
    "begin_download_task",
    "LocalModelInfo",
    "DownloadProgress",
    "DownloadTaskResult",
    "DownloadTaskStatus",
    "LocalModelManager",
    "ModelManager",
    "LlamaCppBackend",
]
