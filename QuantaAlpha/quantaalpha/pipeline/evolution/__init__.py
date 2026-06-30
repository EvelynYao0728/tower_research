"""
Multi-direction factor exploration (original phase only).

Trajectory pool tracks hypothesis → factor → backtest → feedback per direction.
"""

from .trajectory import StrategyTrajectory, TrajectoryPool, RoundPhase
from .controller import EvolutionController, EvolutionConfig

__all__ = [
    "StrategyTrajectory",
    "TrajectoryPool",
    "RoundPhase",
    "EvolutionController",
    "EvolutionConfig",
]
