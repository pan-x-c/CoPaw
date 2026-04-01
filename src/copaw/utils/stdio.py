# -*- coding: utf-8 -*-
"""Helpers for sanitizing process standard streams."""

from __future__ import annotations

import os
import sys
from typing import TextIO, cast

_FALLBACK_STREAMS: list[TextIO] = []


def ensure_standard_streams() -> None:
    """Replace unusable stdout/stderr streams with safe fallbacks."""
    sys.stdout = _ensure_text_stream(sys.stdout)
    sys.stderr = _ensure_text_stream(sys.stderr)


def _ensure_text_stream(stream: TextIO | None) -> TextIO:
    if _is_stream_usable(stream):
        return cast(TextIO, stream)

    fallback = _open_fallback_stream(stream)
    _FALLBACK_STREAMS.append(fallback)
    return fallback


def _is_stream_usable(stream: TextIO | None) -> bool:
    if stream is None:
        return False

    try:
        stream.flush()
    except (AttributeError, OSError, ValueError):
        return False

    try:
        stream.write("")
    except (AttributeError, OSError, ValueError):
        return False

    return True


def _open_fallback_stream(stream: TextIO | None) -> TextIO:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return open(
        os.devnull,
        "a",
        encoding=encoding,
        buffering=1,
        errors="replace",
    )
