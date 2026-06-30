"""Default paths and hyper-parameters for model training."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent

DEFAULT_FACTOR_ROOT = RESEARCH / "因子库_final"
DEFAULT_PANEL_ROOT = RESEARCH / "new_feature_1" / "basic_feature_1"
DEFAULT_PANEL_ROOT_2 = RESEARCH / "new_feature_1" / "basic_feature_2"
DEFAULT_PANEL_ROOTS = (DEFAULT_PANEL_ROOT, DEFAULT_PANEL_ROOT_2)
DEFAULT_LABEL_ROOT = RESEARCH / "data" / "label"
DEFAULT_TRADE_DATE = RESEARCH / "data" / "trade_date.csv"
DEFAULT_REGISTRY = DEFAULT_FACTOR_ROOT / "factor_registry.yaml"
DEFAULT_OUTPUT = ROOT / "output" / "linear"
DEFAULT_LGB_OUTPUT = ROOT / "output" / "lgb"
DEFAULT_LSTM_OUTPUT = ROOT / "output" / "lstm"
DEFAULT_PANEL_CACHE = ROOT / "cache" / "panels"


def default_workers() -> int:
    return min(16, os.cpu_count() or 1)


def default_lgb_workers() -> int:
    """Conservative default for heavy multi-panel LGB loads (avoids OOM)."""
    return min(4, os.cpu_count() or 1)


@dataclass
class LinearModelConfig:
    """Configuration for the cross-sectional linear (Ridge) model."""

    factor_root: Path = field(default_factory=lambda: DEFAULT_FACTOR_ROOT)
    label_root: Path = field(default_factory=lambda: DEFAULT_LABEL_ROOT)
    trade_date_csv: Path = field(default_factory=lambda: DEFAULT_TRADE_DATE)
    registry: Path = field(default_factory=lambda: DEFAULT_REGISTRY)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT)

    label_col: str = "ex_log_ret_10m"
    train_ratio: float = 0.60
    valid_ratio: float = 0.20
    # test_ratio = 1 - train - valid

    ridge_alphas: tuple[float, ...] = (
        1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0,
    )
    winsorize_quantile: float = 0.01
    session_start: int = 931
    session_end: int = 1559
    n_deciles: int = 10
    annualization_days: int = 252
    max_days: int | None = None
    workers: int = field(default_factory=default_workers)
    skip_eval: bool = False
    skip_plots: bool = False
    show_progress: bool = True
    cache_dir: Path | None = field(default_factory=lambda: DEFAULT_PANEL_CACHE)
    use_cache: bool = True
    refresh_cache: bool = False


@dataclass
class LGBModelConfig:
    """Configuration for the LightGBM cross-sectional return model."""

    factor_root: Path = field(default_factory=lambda: DEFAULT_FACTOR_ROOT)
    label_root: Path = field(default_factory=lambda: DEFAULT_LABEL_ROOT)
    trade_date_csv: Path = field(default_factory=lambda: DEFAULT_TRADE_DATE)
    registry: Path = field(default_factory=lambda: DEFAULT_REGISTRY)
    output_dir: Path = field(default_factory=lambda: DEFAULT_LGB_OUTPUT)

    label_col: str = "ex_log_ret_10m"
    train_ratio: float = 0.60
    valid_ratio: float = 0.20
    # test_ratio = 1 - train - valid

    winsorize_quantile: float = 0.01
    session_start: int = 931
    session_end: int = 1559
    train_session_end: int | None = None  # train-only minute cap (e.g. 1439 = 14:39)
    n_deciles: int = 10
    annualization_days: int = 252
    max_days: int | None = None
    workers: int = field(default_factory=default_lgb_workers)
    skip_eval: bool = False
    skip_plots: bool = False
    show_progress: bool = True
    cache_dir: Path | None = field(default_factory=lambda: DEFAULT_PANEL_CACHE)
    use_cache: bool = True
    refresh_cache: bool = False

    # LightGBM hyper-parameters (v1 defaults)
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 10.0
    n_estimators: int = 300
    early_stopping_rounds: int = 50

    # Rolling refit: early-stop on valid, then refit on train+valid (like Ridge)
    rolling_refit: bool = True

    # Label handling for training (evaluation always uses raw label_col)
    label_mode: str = "raw"  # raw | winsorize | rank | winsorize_zscore | winsorize_rank_zscore
    label_winsorize_quantile: float = 0.01

    # LightGBM objective
    objective: str = "regression"  # regression | huber | regression_l1 | lambdarank
    lambdarank_bins: int = 10


@dataclass
class LSTMModelConfig:
    """Configuration for the CPU LSTM minute-sequence return model."""

    factor_root: Path = field(default_factory=lambda: DEFAULT_FACTOR_ROOT)
    panel_roots: tuple[Path, ...] = field(default_factory=lambda: DEFAULT_PANEL_ROOTS)
    label_root: Path = field(default_factory=lambda: DEFAULT_LABEL_ROOT)
    trade_date_csv: Path = field(default_factory=lambda: DEFAULT_TRADE_DATE)
    registry: Path = field(default_factory=lambda: DEFAULT_REGISTRY)
    output_dir: Path = field(default_factory=lambda: DEFAULT_LSTM_OUTPUT)

    label_col: str = "ex_log_ret_10m"
    train_ratio: float = 0.80

    winsorize_quantile: float = 0.01
    session_start: int = 931
    session_end: int = 1559
    n_deciles: int = 10
    annualization_days: int = 252
    max_days: int | None = None
    workers: int = field(default_factory=default_lgb_workers)
    skip_eval: bool = False
    skip_plots: bool = False
    show_progress: bool = True
    cache_dir: Path | None = field(default_factory=lambda: DEFAULT_PANEL_CACHE)
    use_cache: bool = True
    refresh_cache: bool = False

    # Sequence model
    seq_len: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 4096
    learning_rate: float = 1e-3
    max_epochs: int = 15
    early_stopping_patience: int = 3
    max_train_sequences: int | None = 300_000
    max_valid_sequences: int | None = 80_000
    device: str = "cpu"
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
