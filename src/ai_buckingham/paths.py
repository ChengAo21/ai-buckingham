"""Per-run output paths without changing process-global configuration."""
from __future__ import annotations

from contextvars import ContextVar
import os
from pathlib import Path


_RUN_OUTPUT = ContextVar("ai_buckingham_run_output", default=None)


class ContextOutputPath(os.PathLike):
    """Resolve the original OUTPUT_DIR expression within the active run."""

    def __init__(self, default):
        self.default = Path(default)

    @property
    def current(self):
        return _RUN_OUTPUT.get() or self.default

    def __fspath__(self):
        return os.fspath(self.current)

    def __str__(self):
        return str(self.current)

    def __truediv__(self, other):
        return self.current / other

    def __getattr__(self, name):
        return getattr(self.current, name)
