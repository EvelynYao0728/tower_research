"""
Pipeline component wiring (class paths loaded dynamically at runtime).
"""

from quantaalpha.core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict


class BasePropSetting(ExtendedBaseSettings):
    """Shared loop component paths."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    hypothesis2experiment: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


class BaseFacSetting(ExtendedBaseSettings):
    """Alpha agent loop component paths."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    construction: str = ""
    calculation: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


class AlphaAgentFactorBasePropSetting(BasePropSetting):
    """Main path: LLM factor mining + iteration."""

    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.private_scenario.PrivateAlphaAgentScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.AlphaAgentHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.AlphaAgentHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.coder.FactorParser"
    runner: str = "quantaalpha.factors.private_runner.PrivateFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 5


class FactorBackTestBasePropSetting(BasePropSetting):
    """Re-run backtest on existing factor expressions."""

    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.private_scenario.PrivateAlphaAgentScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.EmptyHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.BacktestHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.coder.FactorCoder"
    runner: str = "quantaalpha.factors.private_runner.PrivateFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.QlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 1


ALPHA_AGENT_FACTOR_PROP_SETTING = AlphaAgentFactorBasePropSetting()
FACTOR_BACK_TEST_PROP_SETTING = FactorBackTestBasePropSetting()
