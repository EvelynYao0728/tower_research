"""Compare generated factor parquets against 因子库_final reference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compare_factor_day(
    ref_path: Path,
    gen_path: Path,
    factor_name: str,
) -> list[str]:
    errors: list[str] = []
    if not ref_path.is_file():
        errors.append(f"missing reference: {ref_path}")
        return errors
    if not gen_path.is_file():
        errors.append(f"missing generated: {gen_path}")
        return errors

    ref = pd.read_parquet(ref_path)
    gen = pd.read_parquet(gen_path)
    if list(ref.columns) != list(gen.columns):
        errors.append(f"columns differ: ref={list(ref.columns)} gen={list(gen.columns)}")
        return errors
    if len(ref) != len(gen):
        errors.append(f"row count: ref={len(ref)} gen={len(gen)}")

    keys = ["date", "sym_root", "sym_suffix", "minute"]
    ref_s = ref.sort_values(keys, kind="mergesort").reset_index(drop=True)
    gen_s = gen.sort_values(keys, kind="mergesort").reset_index(drop=True)

    for col in keys:
        if not ref_s[col].equals(gen_s[col]):
            errors.append(f"key column {col} differs")
            break

    a = ref_s[factor_name].to_numpy(dtype=np.float32)
    b = gen_s[factor_name].to_numpy(dtype=np.float32)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if not same.all():
        n = (~same).sum()
        max_diff = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        errors.append(f"{factor_name}: {n} rows differ, max_abs_diff={max_diff}")
    return errors
