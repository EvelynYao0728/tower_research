from quantaalpha.coder.costeer.evaluators import CoSTEERSingleFeedback
from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.core.evolving_agent import RAGEvoAgent
from quantaalpha.core.evolving_framework import EvolvableSubjects


class FilterFailedRAGEvoAgent(RAGEvoAgent):
    def filter_evolvable_subjects_by_feedback(
        self, evo: EvolvableSubjects, feedback: CoSTEERSingleFeedback
    ) -> EvolvableSubjects:
        assert isinstance(evo, EvolvingItem)
        assert isinstance(feedback, list)
        assert len(evo.sub_workspace_list) == len(feedback)

        for index in range(len(evo.sub_workspace_list)):
            ws = evo.sub_workspace_list[index]
            if ws is not None and feedback[index] and not feedback[index].final_decision:
                # 避免循环 import FactorFBWorkspace：用 duck typing。
                # 因子工作区若 shutil.rmtree 整目录会删掉 factor.py，PrivateFactorRunner 无法再 execute。
                soft_clear = getattr(ws, "clear_execution_artifacts_only", None)
                if callable(soft_clear):
                    soft_clear()
                else:
                    ws.clear()
        return evo
