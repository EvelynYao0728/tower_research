"""
Structured summary of evolution runs: propose → construct → backtest → feedback,
per planning direction (original phase).

Written to ``factor_iteration_report.json`` under the experiment log root for
offline review and downstream optimization tooling.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantaalpha.pipeline.evolution.controller import EvolutionController


_PHASE_ORDER = {"original": 0}


def build_iteration_report(controller: EvolutionController) -> dict[str, Any]:
    pool = controller.pool
    trajs = pool.get_all()
    rows: list[dict[str, Any]] = []
    for t in sorted(
        trajs,
        key=lambda x: (
            x.round_idx,
            _PHASE_ORDER.get(x.phase.value if hasattr(x.phase, "value") else str(x.phase), 9),
            x.direction_id,
        ),
    ):
        rows.append(
            {
                "trajectory_id": t.trajectory_id,
                "phase": t.phase.value,
                "round_idx": t.round_idx,
                "direction_id": t.direction_id,
                "parent_ids": list(t.parent_ids),
                "primary_metric_rank_ic": t.get_primary_metric(),
                "metrics": {k: v for k, v in (t.backtest_metrics or {}).items() if v is not None},
                "factor_names": [f.get("name") for f in (t.factors or [])],
                "factor_expressions": [f.get("expression") for f in (t.factors or [])],
                "hypothesis_preview": (t.hypothesis or "")[:500],
                "feedback_preview": (t.feedback or "")[:500],
                "successful_by_rank_ic_rule": t.is_successful(),
            }
        )
    best = controller.get_best_trajectories(top_n=8)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Each entry in iterations is one full inner loop: 假设提出 → 因子构造/代码 → 私有特征回测 → 反馈；"
            "每个 planning 方向一条轨迹；max_rounds 控制外层重复次数。"
            "best_by_rank_ic 按 RankIC 排序。"
        ),
        "pool_statistics": pool.get_statistics(),
        "evolution_config": {
            "max_rounds": controller.config.max_rounds,
            "num_directions": controller.config.num_directions,
        },
        "iterations": rows,
        "best_by_rank_ic": [
            {
                "trajectory_id": b.trajectory_id,
                "phase": b.phase.value,
                "round_idx": b.round_idx,
                "RankIC": b.get_primary_metric(),
            }
            for b in best
        ],
    }


def write_iteration_report_json(controller: EvolutionController, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "factor_iteration_report.json"
    path.write_text(
        json.dumps(build_iteration_report(controller), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
