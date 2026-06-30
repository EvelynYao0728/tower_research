"""Hard guard: the package must never write inside the public read-only tree."""
from __future__ import annotations

from pathlib import Path

PUBLIC_ROOT = Path("/home/yzyao.25/research/public").resolve()


class PublicWriteError(RuntimeError):
    """Raised when an output path resolves into the read-only public dir."""


def assert_safe_output(path: Path) -> Path:
    """Resolve `path` and refuse if it falls inside PUBLIC_ROOT."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PUBLIC_ROOT)
    except ValueError:
        return resolved
    raise PublicWriteError(
        f"Refusing to write inside read-only public dir: {resolved}"
    )
