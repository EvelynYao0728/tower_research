"""
Unified access to minute-level feature material under two physical layouts:

1) **legacy_panel** (e.g. ``simple_factors`` → quote_base_feature): one parquet per
   calendar day, many numeric feature columns + keys
   ``date, sym_root, sym_suffix, minute``.

2) **per_feature_dirs** (e.g. ``0511simple_factors``): one subdirectory per feature,
   each holding daily parquets with the same keys + a single value column named
   after the feature.

Agents should only depend on :func:`load_feature_long` and
:class:`PrivateMarketCatalog`, not on which directory a field came from.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import pandas as pd
import pyarrow.parquet as pq

LONG_KEYS = ("date", "sym_root", "sym_suffix", "minute")

# Backtest labels (under research/data/label). Must NOT appear in factor expressions — lookahead.
LABEL_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "ex_log_ret_10m",
        "log_ret_10m",
        "ex_ret_10m",
        "ret_10m",
    }
)

# Panel index columns — present in every parquet row, not loadable via $field.
PANEL_KEY_FIELD_NAMES: frozenset[str] = frozenset(LONG_KEYS) | frozenset(
    {"instrument", "datetime"}
)

# Substrings that indicate pandas / Python DSL, not factor_lib expressions.
_FORBIDDEN_EXPR_FRAGMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.fillna\s*\(", re.I), "do not use pandas .fillna()"),
    (re.compile(r"\.astype\s*\(", re.I), "do not use pandas .astype()"),
    (re.compile(r"\s->\s"), "do not use '->' assignments"),
    (re.compile(r";\s*\S"), "use a single expression (no semicolon statements)"),
    (re.compile(r"['\"].*?\+.*?['\"]"), "do not build instrument strings in the expression"),
)

_DOLLAR_FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def extract_dollar_fields(expr: str) -> set[str]:
    """Field tokens referenced as ``$name`` in a factor expression."""
    return {m.group(1) for m in _DOLLAR_FIELD_RE.finditer(expr or "")}


def validate_factor_expression_fields(
    expr: str, cfg: PrivateDataConfig | None = None
) -> tuple[bool, str]:
    """
    Ensure expression only uses loadable minute features (not labels / unknown columns).

    Returns (ok, error_message). ``error_message`` is empty when ok.
    """
    fields = extract_dollar_fields(expr)
    if not fields:
        return False, "No $feature tokens in expression."

    label_hits = sorted(fields & LABEL_FIELD_NAMES)
    if label_hits:
        return (
            False,
            "Forbidden label fields in factor expression (lookahead; labels are only for "
            f"backtest IC, not for $fields): {label_hits}. Use quote/microstructure features only.",
        )

    panel_hits = sorted(fields & PANEL_KEY_FIELD_NAMES)
    if panel_hits:
        return (
            False,
            "Panel index columns must not be used as $fields "
            f"(they are keys on every row, not minute features): {panel_hits}. "
            "Use spread/imbalance/mid_price features only; instrument is derived automatically.",
        )

    for pat, hint in _FORBIDDEN_EXPR_FRAGMENTS:
        if pat.search(expr):
            return (
                False,
                f"Invalid factor DSL: {hint}. Expression must be one function_lib formula, "
                f"not pandas/Python code. Offending fragment near: {expr[:160]!r}…",
            )

    try:
        from quantaalpha.factors.coder.expr_parser import parse_expression

        parse_expression(expr)
    except Exception as e:
        return (
            False,
            f"Expression does not parse as factor DSL ({e}). "
            "Use $feature names and function_lib calls (RANK, TS_MEAN, TS_ZSCORE, …) only.",
        )

    catalog = PrivateMarketCatalog(cfg)
    unknown: list[str] = []
    for f in sorted(fields):
        try:
            catalog.resolve_source(f)
        except KeyError:
            unknown.append(f)
    if unknown:
        avail = sorted(catalog.all_fields())
        hint = ", ".join(avail[:12]) + ("…" if len(avail) > 12 else "")
        return (
            False,
            f"Unknown feature field(s) {unknown}. Not in private feature catalog. "
            f"Examples of valid fields: {hint}",
        )
    return True, ""


def _norm_suffix(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "None"
    s = str(v).strip()
    return s if s else "None"


@dataclass
class PrivateDataConfig:
    """Default roots match the private research tree; override via env if needed."""

    legacy_panel_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "QUANTALPHA_LEGACY_PANEL_ROOT",
                "/home/yzyao.25/research/data/simple_factors",
            )
        ).expanduser().resolve()
    )
    per_feature_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "QUANTALPHA_PER_FEATURE_ROOT",
                "/home/yzyao.25/research/data/0511simple_factors",
            )
        ).expanduser().resolve()
    )

    def validate_roots_exist(self) -> None:
        for p in (self.legacy_panel_root, self.per_feature_root):
            if not p.exists():
                raise FileNotFoundError(f"Data root does not exist: {p}")


class PrivateMarketCatalog:
    """Discover fields and resolve whether a column lives in panel or per-feature storage."""

    def __init__(self, cfg: PrivateDataConfig | None = None) -> None:
        self.cfg = cfg or PrivateDataConfig()
        self._legacy_fields: set[str] | None = None
        self._per_feature_fields: set[str] | None = None

    def _sample_legacy_columns(self) -> set[str]:
        root = self.cfg.legacy_panel_root
        if not root.exists():
            return set()
        files = sorted(root.glob("*.parquet"))
        if not files:
            return set()
        names = set(pq.ParquetFile(files[0]).schema_arrow.names)
        return names

    @property
    def legacy_fields(self) -> set[str]:
        if self._legacy_fields is None:
            cols = self._sample_legacy_columns()
            self._legacy_fields = {c for c in cols if c not in LONG_KEYS}
        return self._legacy_fields

    @property
    def per_feature_fields(self) -> set[str]:
        if self._per_feature_fields is None:
            root = self.cfg.per_feature_root
            if not root.is_dir():
                self._per_feature_fields = set()
            else:
                self._per_feature_fields = {
                    p.name
                    for p in root.iterdir()
                    if p.is_dir() and any(p.glob("*.parquet"))
                }
        return self._per_feature_fields

    def all_fields(self) -> set[str]:
        return self.legacy_fields | self.per_feature_fields

    def resolve_source(
        self, field: str
    ) -> tuple[Literal["legacy_panel", "per_feature"], Path]:
        if field in self.per_feature_fields:
            return "per_feature", self.cfg.per_feature_root / field
        if field in self.legacy_fields:
            return "legacy_panel", self.cfg.legacy_panel_root
        raise KeyError(
            f"Unknown field {field!r}. "
            f"Not in per-feature dirs nor legacy panel schema sample."
        )

    def describe_for_prompt(self, max_fields: int = 80) -> str:
        """Short text for LLM system prompts."""
        fields = sorted(self.all_fields())
        if len(fields) > max_fields:
            head = ", ".join(fields[:max_fields])
            return (
                f"Minute feature material: {len(fields)} fields (showing {max_fields}). "
                f"Examples: {head}, …\n"
                f"Keys: {LONG_KEYS} (sym_suffix uses string 'None' when null in storage).\n"
                f"Daily-frequency fields: imbalance_entropy (日频 — constant within each trading day).\n"
                f"Load with: from quantaalpha.data import load_feature_long\n"
            )
        return (
            f"Minute feature fields ({len(fields)}): {', '.join(fields)}\n"
            f"Keys: {LONG_KEYS}\n"
            f"Daily-frequency fields: imbalance_entropy (日频 — one value per symbol per day, "
            f"broadcast to minute rows; do not apply intraday TS_* expecting bar changes).\n"
        )


def _iter_day_files(root: Path) -> Iterator[Path]:
    files = sorted(root.glob("*.parquet"))
    for f in files:
        if f.stem.isdigit() and len(f.stem) == 8:
            yield f


def _filter_dates(paths: Iterable[Path], dates: Sequence[str] | None) -> list[Path]:
    if not dates:
        return list(paths)
    allow = {d.replace("-", "") for d in dates}
    return [p for p in paths if p.stem.replace("-", "") in allow]


def load_feature_long(
    fields: Sequence[str],
    *,
    dates: Sequence[str] | None = None,
    cfg: PrivateDataConfig | None = None,
) -> pd.DataFrame:
    """
    Load and outer-merge requested fields on ``LONG_KEYS``.

    Parameters
    ----------
    fields:
        Feature column names (must exist in catalog).
    dates:
        Optional ``YYYYMMDD`` strings (dashes allowed); default = all days found
        in overlapping files.
    """
    if not fields:
        raise ValueError("fields must be non-empty")

    catalog = PrivateMarketCatalog(cfg)
    legacy_cols: list[str] = []
    per_feature: dict[str, Path] = {}
    for f in fields:
        kind, path = catalog.resolve_source(f)
        if kind == "legacy_panel":
            legacy_cols.append(f)
        else:
            per_feature[f] = path

    # Collect trading days (stem YYYYMMDD) from every involved source
    day_stems: set[str] = set()
    if legacy_cols:
        for fp in _filter_dates(_iter_day_files(catalog.cfg.legacy_panel_root), dates):
            day_stems.add(fp.stem.replace("-", ""))
    for subdir in per_feature.values():
        for fp in _filter_dates(_iter_day_files(subdir), dates):
            day_stems.add(fp.stem.replace("-", ""))

    if not day_stems:
        return pd.DataFrame(columns=list(LONG_KEYS) + list(fields))

    day_stems_sorted = sorted(day_stems)
    daily_frames: list[pd.DataFrame] = []

    for stem in day_stems_sorted:
        base: pd.DataFrame | None = None

        if legacy_cols:
            leg_fp = catalog.cfg.legacy_panel_root / f"{stem}.parquet"
            if leg_fp.exists():
                pf = pq.ParquetFile(leg_fp)
                avail = set(pf.schema_arrow.names)
                want = [c for c in list(LONG_KEYS) + legacy_cols if c in avail]
                base = pq.read_table(leg_fp, columns=want).to_pandas()
                base["sym_suffix"] = base["sym_suffix"].map(_norm_suffix)

        for feat_name, subdir in per_feature.items():
            fp = subdir / f"{stem}.parquet"
            if not fp.exists():
                continue
            avail = pq.ParquetFile(fp).schema_arrow.names
            need = [c for c in list(LONG_KEYS) + [feat_name] if c in avail]
            part = pq.read_table(fp, columns=need).to_pandas()
            part["sym_suffix"] = part["sym_suffix"].map(_norm_suffix)
            if base is None:
                base = part
            else:
                base = base.merge(part, on=list(LONG_KEYS), how="outer")

        if base is not None:
            daily_frames.append(base)

    if not daily_frames:
        return pd.DataFrame(columns=list(LONG_KEYS) + list(fields))

    out = pd.concat(daily_frames, ignore_index=True)
    for f in fields:
        if f not in out.columns:
            out[f] = pd.NA
    return out[list(LONG_KEYS) + list(fields)]


def list_available_fields(cfg: PrivateDataConfig | None = None) -> list[str]:
    return sorted(PrivateMarketCatalog(cfg).all_fields())
