from quantaalpha.coder.costeer import CoSTEER
from quantaalpha.coder.costeer.evaluators import CoSTEERMultiEvaluator
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.coder.evaluators import FactorEvaluatorForCoder
from quantaalpha.factors.coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy, FactorParsingStrategy, FactorRunningStrategy
)
from quantaalpha.factors.factor_materialize import promote_experiment_after_debug
from quantaalpha.core.scenario import Scenario
from quantaalpha.log import logger


class FactorCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(
            *args,
            settings=setting,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            with_knowledge=FACTOR_COSTEER_SETTINGS.with_knowledge,
            knowledge_self_gen=FACTOR_COSTEER_SETTINGS.knowledge_self_gen,
            **kwargs,
        )

    def develop(self, exp):
        exp = super().develop(exp)
        if FACTOR_COSTEER_SETTINGS.promote_full_panel_after_debug:
            logger.info(
                "CoSTEER Debugging 完成，开始全量交易日物化（FACTOR_CoSTEER_DEBUG_PANEL_MAX_DAYS=%s 仅用于调试）",
                FACTOR_COSTEER_SETTINGS.debug_panel_max_days,
            )
            promote_experiment_after_debug(exp)
        return exp
        


class FactorParser(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FactorParsingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(
            *args,
            settings=setting,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            with_knowledge=FACTOR_COSTEER_SETTINGS.with_knowledge,
            knowledge_self_gen=FACTOR_COSTEER_SETTINGS.knowledge_self_gen,
            **kwargs,
        )
        
        
class FactorCoder(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = FactorRunningStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(*args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)
