"""
Offline / CI stubs for :class:`quantaalpha.llm.client.APIBackend` chat completions.

Enable with ``QUANTALPHA_LLM_STUB=1`` (or ``true`` / ``yes``). Intended for
``./run.sh`` smoke runs without calling external LLM APIs.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from quantaalpha.core.template import CodeTemplate
from quantaalpha.log import logger

_STUB_EXPR = "RANK($spread_mean)"
_stub_factor_i = 0


def llm_stub_enabled() -> bool:
    v = os.environ.get("QUANTALPHA_LLM_STUB", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _flatten_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    sys_parts: list[str] = []
    usr_parts: list[str] = []
    for m in messages:
        role = m.get("role") or ""
        content = m.get("content") or ""
        if role == "system":
            sys_parts.append(content)
        elif role == "user":
            usr_parts.append(content)
    return "\n".join(sys_parts), "\n".join(usr_parts)


def _parse_factor_name(user_c: str) -> str:
    m = re.search(r"factor_name:\s*(\S+)", user_c)
    return (m.group(1).strip() if m else "StubMicro").replace('"', "").replace("'", "")[:64]


def _render_factor_py(expression: str, factor_name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    tpl_path = root / "factors" / "coder" / "template.jinjia2"
    return CodeTemplate(template_path=tpl_path).render(
        expression=expression, factor_name=factor_name
    )


def _stub_factor_experiment_payload() -> str:
    global _stub_factor_i
    _stub_factor_i += 1
    name = f"StubMicro{_stub_factor_i}"
    payload = {
        name: {
            "description": "Cross-sectional rank of spread_mean (stub pipeline).",
            "formulation": "RANK(spread_mean)",
            "expression": _STUB_EXPR,
            "variables": {},
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def maybe_stub_chat_response(
    messages: list[dict[str, Any]],
    *,
    json_mode: bool,
    reasoning_flag: bool,
) -> tuple[str | None, str | None]:
    """
    Returns ``(content, finish_reason)`` when stub handles this request;
    ``(None, None)`` to fall through to the real API.
    """
    del reasoning_flag  # stub paths do not distinguish reasoning vs chat
    if not llm_stub_enabled():
        return None, None

    sys_c, usr_c = _flatten_messages(messages)
    c = f"{sys_c}\n{usr_c}"
    if os.environ.get("QUANTALPHA_STUB_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            Path("/tmp/quantaalpha_stub_debug.txt").write_text(
                "json_mode=%r\nSYS_HEAD\n%s\n\nUSR_HEAD\n%s\n"
                % (json_mode, sys_c[:1200], usr_c[:1200]),
                encoding="utf-8",
            )
        except OSError:
            pass

    # --- Deterministic branches (order: most specific first) ---

    if (
        json_mode
        and "final_decision" in c
        and "final_feedback" in c
        and (
            "code_feedback" in c
            or "--------------Code feedback" in c
            or "Code feedback" in c
        )
    ):
        return (
            json.dumps(
                {"final_decision": True, "final_feedback": "stub: accept for smoke run"},
                ensure_ascii=False,
            ),
            "stop",
        )

    if json_mode and "output dataframe info" in usr_c.lower():
        return (
            json.dumps(
                {
                    "output_format_feedback": "stub: long panel with datetime/instrument keys",
                    "output_format_decision": True,
                },
                ensure_ascii=False,
            ),
            "stop",
        )

    if json_mode and "The target hypothesis you are targeting" in usr_c:
        logger.info("LLM stub: hypothesis2experiment JSON")
        return _stub_factor_experiment_payload(), "stop"

    if json_mode and "The user is working on generating new hypotheses" in sys_c:
        if "The target hypothesis you are targeting" not in usr_c and (
            "first round" in usr_c.lower()
            or "former hypothesis and the corresponding feedbacks" in usr_c
        ):
            logger.info("LLM stub: AlphaAgent hypothesis JSON")
            hyp = {
                "hypothesis": "Order-flow spread level contains short-horizon cross-sectional alpha (stub).",
                "concise_observation": "Microstructure: spread_mean varies across names and minutes.",
                "concise_knowledge": "Use minute-level $spread_mean from private parquet.",
                "concise_justification": "Tighter spread names may behave differently in ranking.",
                "concise_specification": "Single-feature rank: RANK($spread_mean).",
            }
            return json.dumps(hyp, ensure_ascii=False), "stop"

    if json_mode and "Target hypothesis:" in usr_c and "Combined Results:" in usr_c:
        logger.info("LLM stub: experiment feedback JSON")
        fb = {
            "Observations": "stub: metrics look finite; proceed exploration.",
            "Feedback for Hypothesis": "stub: acceptable for smoke.",
            "New Hypothesis": "",
            "Reasoning": "stub pipeline",
            "Replace Best Result": "no",
        }
        return json.dumps(fb, ensure_ascii=False), "stop"

    if json_mode and '"expr"' in sys_c and "CORRECTED_FACTOR_EXPRESSION" in sys_c:
        logger.info("LLM stub: corrected expr JSON")
        return json.dumps({"expr": _STUB_EXPR}, ensure_ascii=False), "stop"

    if json_mode and "The Python code as a string" in sys_c:
        fname = _parse_factor_name(usr_c)
        code = _render_factor_py(_STUB_EXPR, fname)
        logger.info("LLM stub: full factor.py JSON")
        return json.dumps({"code": code}, ensure_ascii=False), "stop"

    if not json_mode and "Generate EXACTLY" in usr_c and "Initial direction:" in usr_c:
        m = re.search(r"EXACTLY\s+(\d+)", usr_c)
        n = int(m.group(1)) if m else 2
        n = max(1, min(n, 8))
        dirs = [f"[stub planning {i+1}] microstructure depth slice" for i in range(n)]
        logger.info("LLM stub: planning directions text")
        body = "```json\n" + json.dumps({"directions": dirs}, ensure_ascii=False) + "\n```"
        return body, "stop"

    if not json_mode and "--------------Execution feedback:---------------" in usr_c:
        logger.info("LLM stub: code critic text")
        return (
            "stub: No critical issues; template execution path is acceptable for smoke.",
            "stop",
        )

    if not json_mode and "--------------Factor information to similar error" in usr_c:
        logger.info("LLM stub: error-summary critics text")
        return "No critics found", "stop"

    logger.warning(
        "QUANTALPHA_LLM_STUB: unmatched prompt; returning minimal JSON object.",
    )
    if json_mode:
        return json.dumps({"stub_fallback": True, "note": "extend quantaalpha/llm/stub_chat.py"}), "stop"
    return "stub: unmatched non-json completion", "stop"
