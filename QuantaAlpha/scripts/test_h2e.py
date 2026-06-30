import os
from pathlib import Path
from jinja2 import Environment, StrictUndefined
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.core.prompts import Prompts
from quantaalpha.factors.proposal import _normalize_factor_experiment_dict

qa = Prompts(file_path=Path("quantaalpha/factors/prompts/proposal.yaml"))
base = Prompts(file_path=Path("quantaalpha/factors/prompts/prompts.yaml"))
hypothesis_text = "OFI and spread predict short-term returns"
system_prompt = Environment(undefined=StrictUndefined).from_string(
    qa["hypothesis2experiment"]["system_prompt"]
).render(
    targets="factors",
    scenario="minute features $spread_mean $imbalance_mean",
    experiment_output_format=base["factor_experiment_output_format"],
)
user_prompt = Environment(undefined=StrictUndefined).from_string(
    qa["hypothesis2experiment"]["user_prompt"]
).render(
    targets="factors",
    target_hypothesis=hypothesis_text,
    hypothesis_and_feedback="none",
    function_lib_description=base["function_lib_description"],
    target_list=[],
    RAG=None,
    expression_duplication=None,
)
resp = APIBackend().build_messages_and_create_chat_completion(
    user_prompt, system_prompt, json_mode=True
)
print("len", len(resp))
print("head", repr(resp[:400]))
d = _normalize_factor_experiment_dict(robust_json_parse(resp))
print("factors", list(d.keys()))
for k, v in d.items():
    print(k, "expr", repr((v.get("expression") or "")[:120]))
