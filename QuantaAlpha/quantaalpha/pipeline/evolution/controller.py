"""
Evolution controller: multi-direction **original** exploration only.

Mutation / crossover phases have been removed. Each planning direction runs one
full inner loop (propose → construct → calculate → backtest → feedback).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from quantaalpha.log import logger
from .trajectory import StrategyTrajectory, TrajectoryPool, RoundPhase


@dataclass
class EvolutionConfig:
    """Configuration for multi-direction original exploration."""

    num_directions: int = 2
    steps_per_loop: int = 5
    max_rounds: int = 1
    parallel_enabled: bool = False
    pool_save_path: Optional[str] = None
    fresh_start: bool = True


class EvolutionController:
    """Schedules original-phase tasks per planning direction."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        pool_path = Path(config.pool_save_path) if config.pool_save_path else None
        self.pool = TrajectoryPool(save_path=pool_path, fresh_start=config.fresh_start)
        self._current_round = 0
        self._directions_completed: set[int] = set()

    def get_current_state(self) -> dict[str, Any]:
        return {
            "round": self._current_round,
            "phase": RoundPhase.ORIGINAL.value,
            "directions_completed": list(self._directions_completed),
            "pool_stats": self.pool.get_statistics(),
        }

    def is_complete(self) -> bool:
        return self._current_round >= self.config.max_rounds

    def get_next_task(self) -> Optional[dict[str, Any]]:
        if self.is_complete():
            logger.info("Evolution complete: reached max_rounds=%s", self.config.max_rounds)
            return None
        return self._get_original_task()

    def get_all_tasks_for_current_phase(self) -> list[dict[str, Any]]:
        if self.is_complete():
            return []
        tasks = []
        for d in range(self.config.num_directions):
            if d not in self._directions_completed:
                tasks.append(
                    {
                        "phase": RoundPhase.ORIGINAL,
                        "direction_id": d,
                        "parent_trajectories": [],
                        "strategy_suffix": "",
                        "round_idx": self._current_round,
                    }
                )
        if not tasks and self._directions_completed:
            self._current_round += 1
            self._directions_completed.clear()
            logger.info(
                "All directions finished for round %s; evolution %s",
                self._current_round - 1,
                "complete" if self.is_complete() else "continuing",
            )
        return tasks

    def advance_phase_after_parallel_completion(self, completed_tasks: list[dict[str, Any]]):
        if not completed_tasks:
            return
        for task in completed_tasks:
            if task.get("phase") == RoundPhase.ORIGINAL:
                self._directions_completed.add(task["direction_id"])
        if len(self._directions_completed) >= self.config.num_directions:
            self._current_round += 1
            self._directions_completed.clear()
            logger.info(
                "Original phase batch done; round=%s/%s",
                self._current_round,
                self.config.max_rounds,
            )

    def _get_original_task(self) -> Optional[dict[str, Any]]:
        for d in range(self.config.num_directions):
            if d not in self._directions_completed:
                return {
                    "phase": RoundPhase.ORIGINAL,
                    "direction_id": d,
                    "parent_trajectories": [],
                    "strategy_suffix": "",
                    "round_idx": self._current_round,
                }
        self._current_round += 1
        self._directions_completed.clear()
        if self.is_complete():
            logger.info("All original exploration rounds complete")
            return None
        return self._get_original_task()

    def report_task_complete(self, task: dict[str, Any], trajectory: StrategyTrajectory):
        self.pool.add(trajectory)
        if task["phase"] == RoundPhase.ORIGINAL:
            self._directions_completed.add(task["direction_id"])
            logger.info("Original round complete for direction %s", task["direction_id"])

    def create_trajectory_from_loop_result(
        self,
        task: dict[str, Any],
        hypothesis: Any,
        experiment: Any,
        feedback: Any,
    ) -> StrategyTrajectory:
        phase = task["phase"]
        direction_id = task["direction_id"]
        round_idx = task["round_idx"]
        traj_id = StrategyTrajectory.generate_id(direction_id, round_idx, phase)

        hypothesis_text = str(hypothesis) if hypothesis else ""
        hypothesis_details = {}
        if hypothesis:
            for attr in (
                "hypothesis",
                "reason",
                "concise_reason",
                "concise_observation",
                "concise_justification",
                "concise_knowledge",
            ):
                if hasattr(hypothesis, attr):
                    hypothesis_details[attr] = getattr(hypothesis, attr, "")

        factors = []
        if experiment and hasattr(experiment, "sub_tasks"):
            for idx, task_obj in enumerate(experiment.sub_tasks):
                factor_info = {
                    "name": getattr(task_obj, "factor_name", f"factor_{idx}"),
                    "expression": getattr(task_obj, "factor_expression", ""),
                    "description": getattr(task_obj, "factor_description", ""),
                }
                if (
                    hasattr(experiment, "sub_workspace_list")
                    and idx < len(experiment.sub_workspace_list)
                ):
                    ws = experiment.sub_workspace_list[idx]
                    if ws and hasattr(ws, "code_dict") and ws.code_dict:
                        factor_info["code"] = ws.code_dict.get("factor.py", "")
                factors.append(factor_info)

        factor_names = [f.get("name", "") for f in factors if f.get("name")]
        backtest_result = getattr(experiment, "result", None) if experiment else None
        backtest_metrics, metrics_by_factor = self._resolve_backtest_metrics(
            factor_names=factor_names,
            experiment=experiment,
        )
        extra_info: dict[str, Any] = {}
        if metrics_by_factor:
            extra_info["backtest_metrics_by_factor"] = metrics_by_factor

        feedback_text = str(feedback) if feedback else ""
        feedback_details = {}
        if feedback:
            for attr in (
                "observations",
                "hypothesis_evaluation",
                "new_hypothesis",
                "reason",
                "decision",
            ):
                if hasattr(feedback, attr):
                    feedback_details[attr] = getattr(feedback, attr, "")

        parent_ids = [p.trajectory_id for p in task.get("parent_trajectories", [])]

        return StrategyTrajectory(
            trajectory_id=traj_id,
            direction_id=direction_id,
            round_idx=round_idx,
            phase=phase,
            hypothesis=hypothesis_text,
            hypothesis_details=hypothesis_details,
            factors=factors,
            backtest_result=backtest_result,
            backtest_metrics=backtest_metrics,
            feedback=feedback_text,
            feedback_details=feedback_details,
            parent_ids=parent_ids,
            extra_info=extra_info,
        )

    def _resolve_backtest_metrics(
        self,
        *,
        factor_names: list[str],
        experiment: Any,
    ) -> tuple[dict[str, Optional[float]], dict[str, dict[str, Optional[float]]]]:
        from quantaalpha.backtest.summary_metrics import resolve_trajectory_backtest_metrics

        try:
            return resolve_trajectory_backtest_metrics(
                factor_names=factor_names,
                experiment=experiment,
            )
        except Exception as e:
            logger.warning("Failed to resolve backtest metrics: %s", e)
            return {}, {}

    def get_best_trajectories(self, top_n: int = 5) -> list[StrategyTrajectory]:
        all_trajs = self.pool.get_all()
        valid = [t for t in all_trajs if t.is_successful()]
        valid.sort(key=lambda t: t.get_primary_metric() or 0, reverse=True)
        return valid[:top_n]

    def save_state(self, path: Path):
        import json

        state = {
            "current_round": self._current_round,
            "current_phase": RoundPhase.ORIGINAL.value,
            "directions_completed": list(self._directions_completed),
            "config": {
                "num_directions": self.config.num_directions,
                "max_rounds": self.config.max_rounds,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved evolution state to %s", path)

    def load_state(self, path: Path):
        import json

        if not path.exists():
            logger.warning("State file not found: %s", path)
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        self._current_round = state.get("current_round", 0)
        self._directions_completed = set(state.get("directions_completed", []))
        logger.info("Loaded evolution state from %s", path)
