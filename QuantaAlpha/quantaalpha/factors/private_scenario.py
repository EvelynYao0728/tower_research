"""
Scenario descriptions for LLM agents: private parquet microstructure material +
``research/backtest`` evaluation outputs.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from quantaalpha.core.scenario import Scenario
from quantaalpha.core.experiment import Task
from quantaalpha.data.private_catalog import PrivateDataConfig, PrivateMarketCatalog


def _tpl_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "private_scenario.yaml"


def _substitute_template(template: str, **kwargs: str) -> str:
    """Replace only explicit ``{key}`` placeholders; leave other ``{...}`` intact.

    ``str.format`` treats every brace as a field (e.g. LaTeX ``s_j`` → KeyError).
    """
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", value)
    return out


class PrivateAlphaAgentScenario(Scenario):
    """Replaces Qlib/rdagent scenario strings for the alpha mining loop."""

    def __init__(self, use_local: bool = True, *args, **kwargs) -> None:
        del use_local, args, kwargs  # API compatibility with former Qlib scenario
        super().__init__()
        self._cfg = PrivateDataConfig()
        self._catalog = PrivateMarketCatalog(self._cfg)
        raw = yaml.safe_load(_tpl_path().read_text(encoding="utf-8"))
        self._tpl = raw

    @property
    def background(self) -> str:
        return self._tpl["background"]

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return _substitute_template(
            self._tpl["source_data_template"],
            catalog_text=self._catalog.describe_for_prompt(),
            legacy_root=str(self._cfg.legacy_panel_root),
            per_feature_root=str(self._cfg.per_feature_root),
        )

    @property
    def interface(self) -> str:
        return self._tpl["interface"]

    @property
    def output_format(self) -> str:
        return self._tpl["output_format"]

    @property
    def simulator(self) -> str:
        return self._tpl["simulator"]

    @property
    def rich_style_description(self) -> str:
        return self._tpl.get("rich_style_description", "")

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        del task, filtered_tag, simple_background
        parts = [
            "## Background\n",
            self.background,
            "\n## Source data\n",
            self.get_source_data_desc(),
            "\n## Coding interface\n",
            self.interface,
            "\n## Factor value output format\n",
            self.output_format,
            "\n## Backtest / metrics\n",
            self.simulator,
        ]
        return "\n".join(parts)

    @property
    def experiment_setting(self) -> str | None:
        return self._tpl.get("experiment_setting")


class PrivateFactorScenario(PrivateAlphaAgentScenario):
    """Alias for clarity in configs."""

    pass
