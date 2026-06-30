import os
from typing import Optional

from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.core.conf import ExtendedSettingsConfigDict

_DEFAULT_PANEL = "/home/yzyao.25/research/data/simple_factors"


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = ExtendedSettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = os.environ.get("QUANTALPHA_FACTOR_DATA", _DEFAULT_PANEL)
    """Minute feature material (parquet shards) symlinked into each factor workspace."""

    data_folder_debug: str = os.environ.get("QUANTALPHA_FACTOR_DATA_DEBUG", _DEFAULT_PANEL)
    """Same as data_folder by default; override for smaller debug slices."""

    simple_background: bool = True
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 1200
    """Timeout in seconds for each factor implementation execution"""

    factor_eval_workers: int = 4
    """``factor_eval`` 并行进程数；运行时以 ``QUANTALPHA_FACTOR_EVAL_WORKERS`` 为准。"""
    inprocess_factor_eval: bool = True
    """
    模板走 ``factor_eval`` 时，在 CoSTEER 进程内直接 ``calculate_factor_to_parquet``，
    避免子进程启动 + 重复 symlink 全量 parquet；仍计算全部交易日。
    """

    skip_workspace_data_symlink: bool = True
    """``factor_eval`` 从 ``QUANTALPHA_*_ROOT`` 读数据，无需把 simple_factors 链到 workspace。"""

    debug_panel_max_days: int = 3
    """CoSTEER Debugging（``execute(Debug)``）仅 eval 最近 N 个交易日；通过后全量物化。"""

    promote_full_panel_after_debug: bool = True
    """``factor_calculate`` 结束后、回测前，将因子重算为全部交易日。"""

    write_per_feature_shards_on_promote: bool = True
    """全量物化时写入 ``QUANTALPHA_PER_FEATURE_ROOT/<因子名>/`` 日分片。"""

    with_knowledge: bool = False
    """CoSTEER RAG 向量检索；Claude 网关通常无 embedding，默认关闭。"""

    knowledge_self_gen: bool = False
    """是否自动生成知识库条目（依赖 embedding 时建议关闭）。"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""
    
    factor_zoo_path: Optional[str] = None
    """Path to the CSV file containing the factor zoo database (e.g., Alpha101 factors).
    If None, only free arguments ratio and unique variables ratio checks will be performed.
    Novelty check (duplication detection) requires a factor zoo file."""
    
    duplication_threshold: int = 8
    """Threshold for duplication detection. If duplicated subtree size exceeds this value, 
    the factor will be rejected."""

    symbol_length_threshold: int = 300
    """Maximum allowed symbol length (SL) for factor expressions. 
    Expressions longer than this threshold will be rejected to prevent overfitting."""
    
    base_features_threshold: int = 6
    """Maximum allowed number of unique base features (ER) in factor expressions.
    Base features are raw variables like $close, $open, $high, $low, $volume.
    Expressions using more than this number of distinct base features will be rejected."""


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
