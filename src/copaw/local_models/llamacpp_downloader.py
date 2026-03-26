# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..utils import system_info


@dataclass(frozen=True)
class DownloadProgress:
    downloaded_bytes: int
    total_bytes: Optional[int]
    percent: Optional[float]
    file_name: str
    url: str


class DownloadCancelled(Exception):
    pass


ProgressCallback = Callable[[DownloadProgress], None]


class LlamaCppDownloader:
    """
    Automatically constructs llama.cpp release download URLs based on
    the local environment and supports downloading.

    Supported strategies:
        - Windows: cpu / cuda
        - macOS: cpu
        - Linux: cpu

    CUDA version mapping:
        - 12.x -> 12.4
        - 13.x -> 13.1

    cancel_token:
        - Supports passing in threading.Event
        - Or any object implementing is_set() -> bool
    """

    def __init__(self, base_url: str, release_tag: str):
        self.base_url = base_url.rstrip("/")
        self.release_tag = release_tag

        self.os_name = self._resolve_os_name()
        self.arch = self._resolve_arch()
        self.cuda_version = self._resolve_cuda_version()
        self.backend = self._resolve_backend()

    # -----------------------------
    # Public APIs
    # -----------------------------
    def get_download_url(self) -> str:
        filename = self._build_filename()
        return f"{self.base_url}/{self.release_tag}/{filename}"

    async def download(
        self,
        dest: str | Path,
        on_progress: Optional[ProgressCallback] = None,
        cancel_token: Optional[Any] = None,
        chunk_size: int = 1024 * 1024,
        timeout: int = 30,
    ) -> Path:
        """
        Download the corresponding release package and extract it.

        Args:
          - dest:
              Destination directory for the extracted package.
              The directory will be created automatically if it does not
              exist.
          - on_progress:
              Progress callback, signature:
              on_progress(progress: DownloadProgress) -> None
          - cancel_token:
              Optional cancel token, supports any object implementing
              is_set() -> bool, e.g. threading.Event
          - chunk_size:
              Size of each read chunk
          - timeout:
              Network timeout in seconds

        Returns:
          - Path to the extraction directory

        Raises:
          - DownloadCancelled: Download cancelled by user
          - URLError / HTTPError: Network error
          - ValueError: dest is an existing file path instead of a
            directory
          - OSError: File write error
          - TypeError: cancel_token does not meet requirements
        """
        return await asyncio.to_thread(
            self._download_sync,
            dest,
            on_progress,
            cancel_token,
            chunk_size,
            timeout,
        )

    def _download_sync(
        self,
        dest: str | Path,
        on_progress: Optional[ProgressCallback] = None,
        cancel_token: Optional[Any] = None,
        chunk_size: int = 1024 * 1024,
        timeout: int = 30,
    ) -> Path:
        """Perform the blocking download and extraction workflow."""
        self._validate_cancel_token(cancel_token)

        dest_dir = self._resolve_dest_dir(dest)
        url = self.get_download_url()
        file_name = url.rsplit("/", 1)[-1]
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_path = dest_dir / file_name

        temp_path = final_path.with_name(final_path.name + ".part")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "llama-release-downloader/1.0"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_bytes = response.headers.get("Content-Length")
                total_bytes_int = (
                    int(total_bytes)
                    if total_bytes and total_bytes.isdigit()
                    else None
                )

                downloaded = 0
                last_percent_int = -1

                with open(temp_path, "wb") as f:
                    while True:
                        if self._is_cancelled(cancel_token):
                            raise DownloadCancelled(
                                "Download cancelled by user.",
                            )

                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        if on_progress:
                            percent = None
                            if total_bytes_int and total_bytes_int > 0:
                                percent = downloaded * 100.0 / total_bytes_int
                                current_percent_int = int(percent)
                                if current_percent_int == last_percent_int:
                                    continue
                                last_percent_int = current_percent_int

                            on_progress(
                                DownloadProgress(
                                    downloaded_bytes=downloaded,
                                    total_bytes=total_bytes_int,
                                    percent=percent,
                                    file_name=file_name,
                                    url=url,
                                ),
                            )

                shutil.move(str(temp_path), str(final_path))
                if self._is_cancelled(cancel_token):
                    raise DownloadCancelled("Download cancelled by user.")

                self._extract_archive(final_path, dest_dir)
                final_path.unlink(missing_ok=True)

                if on_progress:
                    on_progress(
                        DownloadProgress(
                            downloaded_bytes=downloaded,
                            total_bytes=total_bytes_int,
                            percent=100.0 if total_bytes_int else None,
                            file_name=file_name,
                            url=url,
                        ),
                    )

                return dest_dir

        except DownloadCancelled:
            self._cleanup_download_files(temp_path, final_path)
            raise
        except Exception:
            self._cleanup_download_files(temp_path, final_path)
            raise

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _resolve_dest_dir(self, dest: str | Path) -> Path:
        path = Path(dest)

        if path.exists() and not path.is_dir():
            raise ValueError("dest must be a directory path")

        return path

    def _extract_archive(self, archive_path: Path, dest_dir: Path) -> None:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{archive_path.stem}-",
                dir=str(dest_dir.parent),
            ),
        )
        try:
            shutil.unpack_archive(str(archive_path), str(staging_dir))
            self._merge_extracted_content(
                staging_dir,
                dest_dir,
                archive_path,
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _merge_extracted_content(
        self,
        staging_dir: Path,
        dest_dir: Path,
        archive_path: Path,
    ) -> None:
        extracted_entries = list(staging_dir.iterdir())
        source_root = staging_dir
        if (
            len(extracted_entries) == 1
            and extracted_entries[0].is_dir()
            and self._should_flatten_archive_root(
                extracted_entries[0],
                archive_path,
            )
        ):
            source_root = extracted_entries[0]

        for item in source_root.iterdir():
            self._merge_path(item, dest_dir / item.name)

    @staticmethod
    def _should_flatten_archive_root(
        root_dir: Path,
        archive_path: Path,
    ) -> bool:
        dir_name = root_dir.name
        archive_names = {
            archive_path.name,
            archive_path.stem,
        }
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip"):
            if archive_path.name.endswith(suffix):
                archive_names.add(archive_path.name[: -len(suffix)])

        return any(
            candidate == dir_name or candidate.startswith(dir_name)
            for candidate in archive_names
        )

    def _merge_path(self, source: Path, destination: Path) -> None:
        if source.is_symlink():
            destination.unlink(missing_ok=True)
            os.symlink(os.readlink(source), destination)
            return

        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
                symlinks=True,
            )
            return

        shutil.copy2(source, destination)

    def _cleanup_download_files(
        self,
        temp_path: Path,
        archive_path: Path,
    ) -> None:
        with suppress(FileNotFoundError):
            temp_path.unlink(missing_ok=True)
        with suppress(FileNotFoundError):
            archive_path.unlink(missing_ok=True)

    def _validate_cancel_token(self, cancel_token: Optional[Any]) -> None:
        if cancel_token is None:
            return

        is_set = getattr(cancel_token, "is_set", None)
        if not callable(is_set):
            raise TypeError(
                "cancel_token must implement is_set() -> bool, "
                "e.g. threading.Event",
            )

    def _is_cancelled(self, cancel_token: Optional[Any]) -> bool:
        if cancel_token is None:
            return False
        return bool(cancel_token.is_set())

    def _resolve_os_name(self) -> str:
        os_name = system_info.get_os_name()
        if os_name in ("windows", "macos", "linux"):
            return os_name
        raise RuntimeError(f"Unsupported OS: {os_name}")

    def _resolve_arch(self) -> str:
        arch = system_info.get_architecture()
        if arch in ("x64", "arm64"):
            return arch
        raise RuntimeError(f"Unsupported architecture: {arch}")

    def _resolve_backend(self) -> str:
        # On macOS and Linux, only CPU backend is supported
        if self.os_name in ("macos", "linux"):
            return "cpu"

        # On Windows, check for CUDA support
        if self.cuda_version is not None:
            return "cuda"
        return "cpu"

    def _resolve_cuda_version(self) -> Optional[str]:
        if self.os_name != "windows":
            return None

        cuda_version = system_info.get_cuda_version()
        if cuda_version is None:
            return None

        major = cuda_version.split(".", 1)[0]
        mapping = {
            "12": "12.4",
            "13": "13.1",
        }
        return mapping.get(major)

    def _build_filename(self) -> str:
        tag = self.release_tag

        if self.os_name == "macos":
            return f"llama-{tag}-bin-macos-{self.arch}.tar.gz"

        if self.os_name == "linux":
            return f"llama-{tag}-bin-ubuntu-{self.arch}.tar.gz"

        if self.os_name == "windows":
            if self.backend == "cuda":
                if self.arch != "x64":
                    raise RuntimeError(
                        "Windows CUDA package is only supported for x64.",
                    )
                return (
                    f"llama-{tag}-bin-win-cuda-"
                    f"{self.cuda_version}-{self.arch}.zip"
                )
            return f"llama-{tag}-bin-win-cpu-{self.arch}.zip"

        raise RuntimeError(f"Unsupported OS: {self.os_name}")
