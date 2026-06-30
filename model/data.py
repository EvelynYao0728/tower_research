"""Load and preprocess factor panels joined with forward-return labels."""
from __future__ import annotations

import gc
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm.auto import tqdm

_RESEARCH = Path(__file__).resolve().parent.parent
if str(_RESEARCH / "backtest") not in sys.path:
    sys.path.insert(0, str(_RESEARCH / "backtest"))

from single_factor_bt import io_utils  # noqa: E402

KEYS = ("date", "sym_root", "sym_suffix", "minute")
ARROW_KEYS = list(KEYS)
MERGE_ON = ["date", "minute", "ticker"]


def normalize_date_str(d: str) -> str:
    return str(d).replace("-", "")


def discover_factor_names(factor_root: Path, registry: Path | None = None) -> list[str]:
    """Return all factor directory names that contain per-day parquet files."""
    from_dirs: list[str] = []
    for p in sorted(factor_root.iterdir()):
        if not p.is_dir() or p.name == "output":
            continue
        if any(p.glob("*.parquet")):
            from_dirs.append(p.name)

    from_registry: list[str] = []
    if registry is not None and registry.is_file():
        with registry.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for item in data.get("factors", []):
            name = item["name"]
            if (factor_root / name).is_dir():
                from_registry.append(name)

    ordered: list[str] = []
    seen: set[str] = set()
    for name in from_registry + from_dirs:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def load_trade_dates(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    dates = df[col].astype(str).str.replace("-", "", regex=False).tolist()
    return dates


def available_dates(
    factor_root: Path,
    label_root: Path,
    factor_cols: Sequence[str],
    candidate_dates: Sequence[str] | None = None,
    show_progress: bool = True,
    panel_roots: Path | Sequence[Path] | None = None,
    dir_factor_cols: Sequence[str] | None = None,
) -> list[str]:
    """Return dates where label, optional panels, and every dir-factor parquet exist."""
    dir_names = list(dir_factor_cols) if dir_factor_cols is not None else list(factor_cols)

    if candidate_dates is None:
        ok_set = {p.stem.replace("-", "") for p in label_root.glob("*.parquet")}
    else:
        ok_set = {normalize_date_str(d) for d in candidate_dates}

    for root in _normalize_panel_roots(panel_roots):
        ok_set &= {p.stem.replace("-", "") for p in root.glob("*.parquet")}

    for name in dir_names:
        ok_set &= {p.stem.replace("-", "") for p in (factor_root / name).glob("*.parquet")}

    dates = sorted(ok_set)
    if show_progress:
        tqdm.write(f"Available dates: {len(dates)}")
    return dates


def split_dates(
    dates: Sequence[str],
    train_ratio: float,
    valid_ratio: float,
) -> tuple[list[str], list[str], list[str]]:
    n = len(dates)
    n_train = max(1, int(n * train_ratio))
    n_valid = max(1, int(n * valid_ratio))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    train = list(dates[:n_train])
    valid = list(dates[n_train:n_train + n_valid])
    test = list(dates[n_train + n_valid:])
    if not test:
        test = [valid.pop()] if valid else [train.pop()]
    return train, valid, test


def split_dates_two_way(
    dates: Sequence[str],
    train_ratio: float,
) -> tuple[list[str], list[str]]:
    """Time-ordered train / valid split without a held-out test set."""
    n = len(dates)
    n_train = max(1, int(n * train_ratio))
    if n_train >= n:
        n_train = max(1, n - 1)
    return list(dates[:n_train]), list(dates[n_train:])


def _normalize_panel_roots(
    panel_roots: Path | Sequence[Path] | None,
) -> list[Path]:
    if panel_roots is None:
        return []
    if isinstance(panel_roots, Path):
        return [panel_roots]
    return list(panel_roots)


def discover_panel_specs(
    panel_roots: Path | Sequence[Path] | None,
) -> list[tuple[Path, list[str]]]:
    """Return (panel_root, factor_names) for each non-empty panel directory."""
    specs: list[tuple[Path, list[str]]] = []
    for root in _normalize_panel_roots(panel_roots):
        names = discover_panel_factor_names(root)
        if names:
            specs.append((root, names))
    return specs


def discover_panel_factor_names(panel_root: Path) -> list[str]:
    """Return factor column names from a multi-column daily panel directory."""
    files = sorted(panel_root.glob("*.parquet"))
    if not files:
        return []
    schema = pq.read_schema(files[0])
    return [name for name in schema.names if name not in KEYS]


def discover_combined_factor_names(
    factor_root: Path,
    panel_roots: Path | Sequence[Path] | None = None,
    registry: Path | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (all_factors, panel_factors, dir_factors) without duplicate names."""
    panel_factors: list[str] = []
    seen_panel: set[str] = set()
    for _, names in discover_panel_specs(panel_roots):
        for name in names:
            if name not in seen_panel:
                seen_panel.add(name)
                panel_factors.append(name)
    dir_factors = discover_factor_names(factor_root, registry)

    all_factors: list[str] = []
    seen: set[str] = set()
    for name in panel_factors + dir_factors:
        if name not in seen:
            seen.add(name)
            all_factors.append(name)
    return all_factors, panel_factors, dir_factors


def panel_cache_fingerprint(
    factor_cols: Sequence[str],
    label_col: str,
    winsorize_q: float,
    session_start: int,
    session_end: int,
    panel_roots: Path | Sequence[Path] | None = None,
) -> str:
    roots = [str(p) for p in _normalize_panel_roots(panel_roots)]
    payload = "|".join(
        [
            label_col,
            f"{winsorize_q:.6g}",
            str(session_start),
            str(session_end),
            ",".join(roots),
            ",".join(factor_cols),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _day_source_paths(
    day: str,
    factor_root: Path,
    label_root: Path,
    factor_cols: Sequence[str],
    panel_roots: Path | Sequence[Path] | None = None,
    dir_factor_cols: Sequence[str] | None = None,
) -> list[Path]:
    paths = [label_root / f"{day}.parquet"]
    for root in _normalize_panel_roots(panel_roots):
        paths.append(root / f"{day}.parquet")
    dir_names = list(dir_factor_cols) if dir_factor_cols is not None else list(factor_cols)
    paths.extend(factor_root / name / f"{day}.parquet" for name in dir_names)
    return paths


def _cache_path(cache_dir: Path, fingerprint: str, day: str) -> Path:
    return cache_dir / fingerprint / f"{day}.parquet"


def _cache_is_fresh(cache_path: Path, source_paths: Sequence[Path]) -> bool:
    if not cache_path.is_file():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for path in source_paths:
        if path.is_file() and path.stat().st_mtime > cache_mtime:
            return False
    return True


def _minute_filter(minutes: np.ndarray, start: int, end: int) -> np.ndarray:
    return (minutes >= start) & (minutes <= end)


def _apply_cross_section_preprocess(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    """Per-minute winsorize + z-score via groupby.transform (no Python minute loop)."""
    factor_list = list(factor_cols)
    minutes = df["minute"]

    for col in factor_list:
        s = df[col].astype(np.float64)
        grp = s.groupby(minutes, sort=False)
        cnt = grp.transform("count")
        lo = grp.transform(lambda x: x.quantile(lower_q) if x.count() >= 5 else np.nan)
        hi = grp.transform(lambda x: x.quantile(upper_q) if x.count() >= 5 else np.nan)

        winsor_mask = cnt >= 5
        clipped = s.copy()
        if winsor_mask.any():
            clipped.loc[winsor_mask] = s.loc[winsor_mask].clip(
                lo.loc[winsor_mask], hi.loc[winsor_mask],
            )

        g2 = clipped.groupby(minutes, sort=False)
        mu = g2.transform("mean")
        sd = g2.transform("std", ddof=1)

        finite = np.isfinite(s.to_numpy())
        cnt_arr = cnt.to_numpy(dtype=np.float64)
        sd_arr = sd.to_numpy(dtype=np.float64)
        mu_arr = mu.to_numpy(dtype=np.float64)
        clipped_arr = clipped.to_numpy(dtype=np.float64)
        normal = (cnt_arr >= 2) & np.isfinite(sd_arr) & (sd_arr >= 1e-12)

        z = np.zeros(len(df), dtype=np.float64)
        z[normal] = (clipped_arr[normal] - mu_arr[normal]) / sd_arr[normal]
        z[normal & ~finite] = 0.0
        z[~normal & finite] = 0.0
        z[~normal & ~finite] = np.nan
        df[col] = z.astype(np.float32)

    return df


def preprocess_day_frame(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str,
    winsorize_q: float,
    session_start: int,
    session_end: int,
) -> pd.DataFrame:
    """Winsorize + cross-sectional z-score factors; keep raw label."""
    keep = _minute_filter(df["minute"].to_numpy(), session_start, session_end)
    df = df.loc[keep].copy()
    if df.empty:
        return df

    lower_q = winsorize_q
    upper_q = 1.0 - winsorize_q
    out = _apply_cross_section_preprocess(df, factor_cols, lower_q, upper_q)
    return out.dropna(subset=[label_col])


def _dedupe_ticker_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, minute, ticker); duplicate sym_suffix rows share values."""
    io_utils.add_ticker_inplace(df)
    if df.duplicated(MERGE_ON).any():
        df = df.drop_duplicates(subset=MERGE_ON, keep="last")
    return df


def _merge_day_tables(
    day: str,
    factor_root: Path,
    label_root: Path,
    factor_cols: Sequence[str],
    label_col: str,
    panel_roots: Path | Sequence[Path] | None = None,
    panel_factor_cols: Sequence[str] | None = None,
    dir_factor_cols: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """Inner-join label + optional multi-column panels + per-factor dirs."""
    label_path = label_root / f"{day}.parquet"
    if not label_path.is_file():
        return None

    label_df = pq.read_table(
        label_path, columns=ARROW_KEYS + [label_col],
    ).to_pandas(self_destruct=True)
    if label_df.empty:
        return None
    label_df = _dedupe_ticker_rows(label_df)
    merged = label_df.set_index(MERGE_ON)

    panel_specs = discover_panel_specs(panel_roots)
    if panel_specs and panel_factor_cols:
        panel_factor_set = set(panel_factor_cols)
        for panel_root, root_factor_cols in panel_specs:
            cols = [c for c in root_factor_cols if c in panel_factor_set]
            if not cols:
                continue
            panel_path = panel_root / f"{day}.parquet"
            if not panel_path.is_file():
                return None
            panel_df = pq.read_table(
                panel_path, columns=ARROW_KEYS + cols,
            ).to_pandas(self_destruct=True)
            if panel_df.empty:
                return None
            panel_df = _dedupe_ticker_rows(panel_df)
            merged = merged.join(
                panel_df.set_index(MERGE_ON)[cols],
                how="inner",
            )
            if merged.empty:
                return None

    dir_names = list(dir_factor_cols) if dir_factor_cols is not None else list(factor_cols)
    for name in dir_names:
        factor_path = factor_root / name / f"{day}.parquet"
        if not factor_path.is_file():
            return None
        factor_df = pq.read_table(
            factor_path, columns=ARROW_KEYS + [name],
        ).to_pandas(self_destruct=True)
        if factor_df.empty:
            return None
        factor_df = _dedupe_ticker_rows(factor_df)
        merged = merged.join(
            factor_df.set_index(MERGE_ON)[[name]],
            how="inner",
        )
        if merged.empty:
            return None

    return merged.reset_index()


def _cache_columns(factor_cols: Sequence[str], label_col: str) -> list[str]:
    return list(KEYS) + ["ticker", *factor_cols, label_col]


def _save_day_cache(
    cache_path: Path,
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cols = _cache_columns(factor_cols, label_col)
    df[cols].to_parquet(cache_path, index=False, compression="zstd")


def _load_one_day(
    day: str,
    factor_root: Path,
    label_root: Path,
    factor_cols: Sequence[str],
    label_col: str,
    winsorize_q: float,
    session_start: int,
    session_end: int,
    cache_dir: Path | None = None,
    cache_fingerprint: str | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    panel_roots: Path | Sequence[Path] | None = None,
    panel_factor_cols: Sequence[str] | None = None,
    dir_factor_cols: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    source_paths = _day_source_paths(
        day,
        factor_root,
        label_root,
        factor_cols,
        panel_roots=panel_roots,
        dir_factor_cols=dir_factor_cols,
    )
    cache_path: Path | None = None
    if use_cache and cache_dir is not None and cache_fingerprint is not None:
        cache_path = _cache_path(cache_dir, cache_fingerprint, day)
        if not refresh_cache and _cache_is_fresh(cache_path, source_paths):
            return pd.read_parquet(cache_path)

    merged = _merge_day_tables(
        day,
        factor_root,
        label_root,
        factor_cols,
        label_col,
        panel_roots=panel_roots,
        panel_factor_cols=panel_factor_cols,
        dir_factor_cols=dir_factor_cols,
    )
    if merged is None or merged.empty:
        return None

    frame = preprocess_day_frame(
        merged,
        factor_cols=factor_cols,
        label_col=label_col,
        winsorize_q=winsorize_q,
        session_start=session_start,
        session_end=session_end,
    )
    if frame is None or frame.empty:
        return None

    if cache_path is not None:
        _save_day_cache(cache_path, frame, factor_cols, label_col)
    return frame


def _load_day_job(args: tuple) -> tuple[str, pd.DataFrame | None]:
    day, kwargs = args
    return day, _load_one_day(day, **kwargs)


def _append_panel_frame(
    panel: pd.DataFrame | None,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if panel is None:
        return frame
    return pd.concat([panel, frame], ignore_index=True)


class _PanelAccumulator:
    """Batch daily frames to avoid O(n^2) copies from repeated full-panel concat."""

    def __init__(self, chunk_days: int = 10) -> None:
        self.chunk_days = chunk_days
        self._buffer: list[pd.DataFrame] = []
        self._chunks: list[pd.DataFrame] = []

    def add(self, frame: pd.DataFrame) -> None:
        self._buffer.append(frame)
        if len(self._buffer) >= self.chunk_days:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        self._chunks.append(pd.concat(self._buffer, ignore_index=True))
        self._buffer.clear()

    def finalize(self) -> pd.DataFrame:
        self._flush_buffer()
        if not self._chunks:
            raise ValueError("No data loaded for the requested date range.")
        if len(self._chunks) == 1:
            return self._chunks[0]
        return pd.concat(self._chunks, ignore_index=True)


def _resolve_parallel_result(fut) -> pd.DataFrame | None:
    try:
        _, frame = fut.result()
        return frame
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "Parallel day loader crashed (likely OOM or segfault in a worker). "
            "Retry with fewer workers, e.g. `--workers 1` or `--workers 2`.",
        ) from exc


def load_panel(
    dates: Iterable[str],
    factor_root: Path,
    label_root: Path,
    factor_cols: Sequence[str],
    label_col: str,
    winsorize_q: float = 0.01,
    session_start: int = 931,
    session_end: int = 1559,
    workers: int = 1,
    desc: str = "Loading days",
    show_progress: bool = True,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    panel_roots: Path | Sequence[Path] | None = None,
    panel_factor_cols: Sequence[str] | None = None,
    dir_factor_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load multiple trading days into a single long-format panel."""
    fingerprint = panel_cache_fingerprint(
        factor_cols, label_col, winsorize_q, session_start, session_end, panel_roots,
    )
    kwargs = dict(
        factor_root=factor_root,
        label_root=label_root,
        factor_cols=factor_cols,
        label_col=label_col,
        winsorize_q=winsorize_q,
        session_start=session_start,
        session_end=session_end,
        cache_dir=cache_dir,
        cache_fingerprint=fingerprint,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        panel_roots=panel_roots,
        panel_factor_cols=panel_factor_cols,
        dir_factor_cols=dir_factor_cols,
    )
    date_list = list(dates)
    accumulator = _PanelAccumulator(chunk_days=10)
    pbar = tqdm(
        total=len(date_list),
        desc=desc,
        unit="day",
        disable=not show_progress or len(date_list) == 0,
    )

    if workers <= 1:
        for day in date_list:
            frame = _load_one_day(day, **kwargs)
            if frame is not None and not frame.empty:
                accumulator.add(frame)
            pbar.update(1)
    else:
        batch_size = max(workers * 2, workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(date_list), batch_size):
                batch = date_list[start:start + batch_size]
                futures = [pool.submit(_load_day_job, (day, kwargs)) for day in batch]
                for fut in as_completed(futures):
                    frame = _resolve_parallel_result(fut)
                    if frame is not None and not frame.empty:
                        accumulator.add(frame)
                    pbar.update(1)
    pbar.close()

    panel = accumulator.finalize()
    panel["date"] = panel["date"].astype(str).str.replace("-", "", regex=False)
    return panel


def split_panel_by_dates(
    panel: pd.DataFrame,
    train_dates: Sequence[str],
    valid_dates: Sequence[str],
    test_dates: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Split a loaded panel into train / valid / test by date."""
    date_key = panel["date"].astype(str).str.replace("-", "", regex=False)
    train_set = {normalize_date_str(d) for d in train_dates}
    valid_set = {normalize_date_str(d) for d in valid_dates}
    test_set = {normalize_date_str(d) for d in test_dates}
    return {
        "train": panel.loc[date_key.isin(train_set)].reset_index(drop=True),
        "valid": panel.loc[date_key.isin(valid_set)].reset_index(drop=True),
        "test": panel.loc[date_key.isin(test_set)].reset_index(drop=True),
    }


def load_and_split_panels(
    train_dates: Sequence[str],
    valid_dates: Sequence[str],
    test_dates: Sequence[str],
    factor_root: Path,
    label_root: Path,
    factor_cols: Sequence[str],
    label_col: str,
    winsorize_q: float = 0.01,
    session_start: int = 931,
    session_end: int = 1559,
    train_session_end: int | None = None,
    workers: int = 1,
    show_progress: bool = True,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    panel_roots: Path | Sequence[Path] | None = None,
    panel_factor_cols: Sequence[str] | None = None,
    dir_factor_cols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load train / valid / test panels separately to avoid a full-sample peak."""
    base = dict(
        factor_root=factor_root,
        label_root=label_root,
        factor_cols=factor_cols,
        label_col=label_col,
        winsorize_q=winsorize_q,
        session_start=session_start,
        workers=workers,
        show_progress=show_progress,
        cache_dir=cache_dir,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        panel_roots=panel_roots,
        panel_factor_cols=panel_factor_cols,
        dir_factor_cols=dir_factor_cols,
    )
    panels: dict[str, pd.DataFrame] = {}
    if train_dates:
        panels["train"] = load_panel(
            train_dates,
            desc="Load train days",
            session_end=train_session_end if train_session_end is not None else session_end,
            **base,
        )
        gc.collect()
    if valid_dates:
        panels["valid"] = load_panel(
            valid_dates,
            desc="Load valid days",
            session_end=session_end,
            **base,
        )
        gc.collect()
    if test_dates:
        panels["test"] = load_panel(
            test_dates,
            desc="Load test days",
            session_end=session_end,
            **base,
        )
    if not panels:
        raise ValueError("No data loaded for the requested date range.")
    return panels


def _global_zscore_stats(
    train_df: pd.DataFrame,
    factor_cols: Sequence[str],
    chunk_rows: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-factor mean/std on train in row chunks (lower peak RAM)."""
    factor_list = list(factor_cols)
    n_features = len(factor_list)
    total_count = np.zeros(n_features, dtype=np.float64)
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_x2 = np.zeros(n_features, dtype=np.float64)

    n_rows = len(train_df)
    for start in range(0, n_rows, chunk_rows):
        chunk = train_df.iloc[start:start + chunk_rows]
        x = chunk[factor_list].to_numpy(dtype=np.float32)
        finite = np.isfinite(x)
        valid_counts = finite.sum(axis=0)
        masked = np.where(finite, x, 0.0).astype(np.float64)
        total_count += valid_counts
        sum_x += masked.sum(axis=0)
        sum_x2 += (masked ** 2).sum(axis=0)

    mu = np.divide(sum_x, total_count, out=np.zeros_like(sum_x), where=total_count > 0)
    var = np.divide(sum_x2, total_count, out=np.zeros_like(sum_x2), where=total_count > 0) - mu ** 2
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(np.isfinite(sd) & (sd >= 1e-12), sd, 1.0)
    return mu.astype(np.float64), sd.astype(np.float64)


def _apply_global_zscore_inplace(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
    chunk_rows: int = 500_000,
) -> None:
    factor_list = list(factor_cols)
    n_rows = len(df)
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        x = df.iloc[start:end][factor_list].to_numpy(dtype=np.float32)
        z = (x - mu) / sd
        z[~np.isfinite(z)] = 0.0
        df.iloc[start:end, df.columns.get_indexer(factor_list)] = z.astype(np.float32)


def apply_global_zscore(
    panels: dict[str, pd.DataFrame],
    factor_cols: Sequence[str],
    fit_split: str = "train",
) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Z-score factor columns using mean/std estimated on fit_split."""
    mu, sd = _global_zscore_stats(panels[fit_split], factor_cols)
    for df in panels.values():
        _apply_global_zscore_inplace(df, factor_cols, mu, sd)
    return panels, mu, sd


TRAINING_LABEL_COL = "_training_label"
RELEVANCE_LABEL_COL = "_relevance_label"


def _winsorize_label_by_minute(
    panel: pd.DataFrame,
    label_col: str,
    label_winsorize_q: float,
) -> pd.Series:
    """Winsorize label at each minute across all dates (existing winsorize logic)."""
    minutes = panel["minute"]
    s = panel[label_col].astype(np.float64)
    grp = s.groupby(minutes, sort=False)
    lower_q = label_winsorize_q
    upper_q = 1.0 - label_winsorize_q
    cnt = grp.transform("count")
    lo = grp.transform(lambda x: x.quantile(lower_q) if x.count() >= 5 else np.nan)
    hi = grp.transform(lambda x: x.quantile(upper_q) if x.count() >= 5 else np.nan)
    winsor_mask = cnt >= 5
    clipped = s.copy()
    if winsor_mask.any():
        clipped.loc[winsor_mask] = s.loc[winsor_mask].clip(
            lo.loc[winsor_mask], hi.loc[winsor_mask],
        )
    return clipped


def _zscore_by_date_minute(
    panel: pd.DataFrame,
    values: pd.Series,
    *,
    min_count: int = 5,
    min_std: float = 1e-8,
) -> pd.Series:
    """Cross-section z-score within each (date, minute); skip thin or flat slices."""
    frame = panel[["date", "minute"]].copy()
    frame["_v"] = values.to_numpy(dtype=np.float64)
    grp = frame.groupby(["date", "minute"], sort=False)["_v"]
    cnt = grp.transform("count")
    mu = grp.transform("mean")
    sd = grp.transform("std", ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (frame["_v"] - mu) / sd
    skip = (cnt < min_count) | (sd < min_std) | ~np.isfinite(sd)
    z.loc[skip] = frame.loc[skip, "_v"]
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.astype(np.float32)


def _rank_zscore_by_date_minute(
    panel: pd.DataFrame,
    values: pd.Series,
    *,
    min_count: int = 5,
    min_std: float = 1e-8,
) -> pd.Series:
    """Rank within (date, minute), then z-score ranks; skip thin slices."""
    frame = panel[["date", "minute"]].copy()
    frame["_v"] = values.to_numpy(dtype=np.float64)
    grp = frame.groupby(["date", "minute"], sort=False)["_v"]
    cnt = grp.transform("count")
    frame["_r"] = grp.rank(method="average").to_numpy(dtype=np.float64)
    grp_r = frame.groupby(["date", "minute"], sort=False)["_r"]
    mu = grp_r.transform("mean")
    sd = grp_r.transform("std", ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (frame["_r"] - mu) / sd
    skip = (cnt < min_count) | (sd < min_std) | ~np.isfinite(sd)
    z.loc[skip] = frame.loc[skip, "_v"]
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.astype(np.float32)


def apply_training_label(
    panel: pd.DataFrame,
    label_col: str,
    *,
    label_mode: str = "raw",
    label_winsorize_q: float = 0.01,
) -> str:
    """Add a training label column; evaluation should still use raw ``label_col``."""
    if label_mode == "raw":
        panel[TRAINING_LABEL_COL] = panel[label_col].astype(np.float32)
        return TRAINING_LABEL_COL

    minutes = panel["minute"]
    s = panel[label_col].astype(np.float64)
    grp = s.groupby(minutes, sort=False)

    if label_mode == "winsorize":
        clipped = _winsorize_label_by_minute(panel, label_col, label_winsorize_q)
        panel[TRAINING_LABEL_COL] = clipped.astype(np.float32)
        return TRAINING_LABEL_COL

    if label_mode == "winsorize_zscore":
        clipped = _winsorize_label_by_minute(panel, label_col, label_winsorize_q)
        panel[TRAINING_LABEL_COL] = _zscore_by_date_minute(panel, clipped)
        return TRAINING_LABEL_COL

    if label_mode == "winsorize_rank_zscore":
        clipped = _winsorize_label_by_minute(panel, label_col, label_winsorize_q)
        panel[TRAINING_LABEL_COL] = _rank_zscore_by_date_minute(panel, clipped)
        return TRAINING_LABEL_COL

    if label_mode == "rank":
        ranked = grp.rank(method="average")
        mu = ranked.groupby(minutes, sort=False).transform("mean")
        sd = ranked.groupby(minutes, sort=False).transform("std", ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (ranked - mu) / sd
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        panel[TRAINING_LABEL_COL] = z.astype(np.float32)
        return TRAINING_LABEL_COL

    raise ValueError(f"Unknown label_mode: {label_mode}")


def apply_relevance_label(
    panel: pd.DataFrame,
    label_col: str,
    *,
    n_relevance_bins: int = 10,
) -> str:
    """Map raw label to integer relevance 0..K-1 within each (date, minute) group."""
    grouped = panel.groupby(["date", "minute"], sort=False)[label_col]
    relevance = grouped.rank(method="first")
    n_valid = grouped.transform("count").astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.floor((relevance - 1.0) * n_relevance_bins / np.maximum(n_valid, 1.0))
    scaled = scaled.clip(lower=0, upper=max(n_relevance_bins - 1, 0))
    panel[RELEVANCE_LABEL_COL] = scaled.astype(np.int32)
    return RELEVANCE_LABEL_COL


def to_xy(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    meta = panel[list(KEYS) + ["ticker"]].copy()
    x = panel[list(factor_cols)].to_numpy(dtype=np.float32)
    y = panel[label_col].to_numpy(dtype=np.float32)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    meta_valid = meta.loc[valid].reset_index(drop=True)
    return x[valid], y[valid], meta_valid


def sort_panel_for_ranking(panel: pd.DataFrame) -> pd.DataFrame:
    """Stable sort by (date, minute) for LightGBM lambdarank group construction."""
    return panel.sort_values(["date", "minute"], kind="mergesort").reset_index(drop=True)


def ranking_group_sizes(meta: pd.DataFrame) -> np.ndarray:
    """Return group sizes aligned with rows sorted by (date, minute)."""
    return meta.groupby(["date", "minute"], sort=False).size().to_numpy(dtype=np.int32)
