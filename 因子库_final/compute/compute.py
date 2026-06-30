"""Orchestrate per-day factor computation and write parquet shards."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from compute.config import ALL_FACTORS, BCDE_FACTORS, FACTOR_GROUPS, KEYS, MB_FACTORS, O_FACTORS
from compute.factors import compute_bcde_factors, compute_mb_factors, compute_o_factors


def write_wide_shards(
    wide: pl.DataFrame,
    factor_names: tuple[str, ...],
    out_root: Path,
    date_stem: str,
    *,
    write_threads: int = 4,
) -> list[Path]:
    keys = list(KEYS)
    tasks: list[tuple[str, Path]] = []
    for name in factor_names:
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((name, out_dir / f"{date_stem}.parquet"))

    def _write_one(item: tuple[str, Path]) -> Path:
        name, out_path = item
        wide.select(keys + [name]).write_parquet(out_path, compression="zstd", statistics=False)
        return out_path

    if write_threads <= 1:
        return [_write_one(t) for t in tasks]

    with ThreadPoolExecutor(max_workers=min(write_threads, len(tasks))) as pool:
        return list(pool.map(_write_one, tasks))


def write_long_shards(
    frames: dict[str, pl.DataFrame],
    factor_names: tuple[str, ...],
    out_root: Path,
    date_stem: str,
) -> list[Path]:
    paths: list[Path] = []
    for name in factor_names:
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_stem}.parquet"
        frames[name].select(*KEYS, name).write_parquet(out_path, compression="zstd")
        paths.append(out_path)
    return paths


def compute_day(
    date_str: str,
    *,
    sf_dir: Path,
    quote_dir: Path,
    trade_dir: Path,
    groups: tuple[str, ...] = ("bcde", "mb", "o"),
    ref_root: Path | None = None,
    tick_workers: int = 4,
    tick_pool: str = "thread",
) -> dict[str, object]:
    """Compute selected factor groups for one trading day. Does not write to disk."""
    stem = date_str.replace("-", "")
    result: dict[str, object] = {"date": stem, "status": "ok", "groups": {}}

    if "mb" in groups:
        sf_path = sf_dir / f"{stem}.parquet"
        if sf_path.is_file():
            result["groups"]["mb"] = compute_mb_factors(pl.read_parquet(sf_path))
        else:
            result["groups"]["mb"] = None
            result.setdefault("warnings", []).append(f"missing simple_factors: {sf_path}")

    trade_path = trade_dir / f"{stem}.parquet"
    quote_path = quote_dir / f"{stem}.parquet"
    if not trade_path.is_file() or not quote_path.is_file():
        if "bcde" in groups or "o" in groups:
            result["status"] = "partial"
            result.setdefault("warnings", []).append("missing quote/trade TAQ")
        return result

    if "bcde" in groups:
        result["groups"]["bcde"] = compute_bcde_factors(
            trade_path, quote_path, ref_root=ref_root
        )
    if "o" in groups:
        result["groups"]["o"] = compute_o_factors(
            stem, quote_path, trade_path, tick_workers=tick_workers, tick_pool=tick_pool
        )
    return result


def generate_day(
    date_str: str,
    out_root: Path,
    *,
    sf_dir: Path,
    quote_dir: Path,
    trade_dir: Path,
    groups: tuple[str, ...] = ("bcde", "mb", "o"),
    ref_root: Path | None = None,
    tick_workers: int = 4,
    tick_pool: str = "thread",
    write_threads: int = 4,
) -> list[Path]:
    """Compute and write all requested factor shards for one day."""
    stem = date_str.replace("-", "")
    res = compute_day(
        date_str,
        sf_dir=sf_dir,
        quote_dir=quote_dir,
        trade_dir=trade_dir,
        groups=groups,
        ref_root=ref_root,
        tick_workers=tick_workers,
        tick_pool=tick_pool,
    )
    paths: list[Path] = []
    group_data = res.get("groups", {})

    if "mb" in groups and group_data.get("mb") is not None:
        paths.extend(
            write_wide_shards(group_data["mb"], MB_FACTORS, out_root, stem, write_threads=1)
        )
    if "bcde" in groups and group_data.get("bcde") is not None:
        paths.extend(
            write_wide_shards(
                group_data["bcde"], BCDE_FACTORS, out_root, stem, write_threads=write_threads
            )
        )
    if "o" in groups and group_data.get("o") is not None:
        paths.extend(write_long_shards(group_data["o"], O_FACTORS, out_root, stem))
    return paths


def pending_dates(
    dates: list[str],
    out_root: Path,
    factor_names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> list[str]:
    if overwrite:
        return dates
    pending: list[str] = []
    for d in dates:
        stem = d.replace("-", "")
        if all((out_root / name / f"{stem}.parquet").is_file() for name in factor_names):
            continue
        pending.append(stem)
    return pending


def factors_for_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for g in groups:
        names.extend(FACTOR_GROUPS[g])
    return tuple(dict.fromkeys(names))
