"""
Logging: prefer rdagent's logger when installed; otherwise use loguru.
"""

from __future__ import annotations

from pathlib import Path

try:
    from rdagent.log import rdagent_logger as _rdagent_logger
    from rdagent.log.utils import LogColors

    class _AlphaAgentLoggerWrapper:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        @property
        def log_trace_path(self) -> Path:
            return self._inner.storage.path

        def set_trace_path(self, path) -> None:
            from rdagent.log.storage import FileStorage

            self._inner.storage = FileStorage(Path(path))

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __setattr__(self, name, value):
            if name in ("_inner",):
                object.__setattr__(self, name, value)
            else:
                setattr(self._inner, name, value)

    logger = _AlphaAgentLoggerWrapper(_rdagent_logger)

except ImportError:  # lightweight installs without rdagent
    import logging

    class LogColors:
        """ANSI codes (rdagent fallback); keep in sync with usages in e.g. llm/client."""

        END = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        MAGENTA = "\033[35m"

    _pylog = logging.getLogger("quantaalpha")

    class _NullTagCtx:
        """`with logger.tag(...):` — no-op; must be a real CM (not SimpleNamespace)."""

        __slots__ = ()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

    class _LoguruTrace:
        _path = Path.cwd() / "log"

        @property
        def path(self) -> Path:
            return self._path

        def truncate(self, time=None, **kwargs) -> None:
            """No-op without rdagent FileStorage (session resume does not trim trace files)."""
            return None

    class _StdCompat:
        """Expose ``set_trace_path`` / ``log_trace_path`` used by the pipeline."""

        storage = _LoguruTrace()

        def set_trace_path(self, path) -> None:
            self.storage._path = Path(path)
            self.storage._path.mkdir(parents=True, exist_ok=True)

        @property
        def log_trace_path(self) -> Path:
            return self.storage.path

        def info(self, msg, *a, **k):
            k.pop("tag", None)
            return _pylog.info(str(msg), *a, **k)

        def warning(self, msg, *a, **k):
            k.pop("tag", None)
            return _pylog.warning(str(msg), *a, **k)

        def error(self, msg, *a, **k):
            k.pop("tag", None)
            return _pylog.error(str(msg), *a, **k)

        def debug(self, msg, *a, **k):
            k.pop("tag", None)
            return _pylog.debug(str(msg), *a, **k)

        def tag(self, *a, **k):
            return _NullTagCtx()

        def log_object(self, *a, **k):
            return None

    logger = _StdCompat()


__all__ = ["logger", "LogColors"]
