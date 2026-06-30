from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import random
import re
import sqlite3
import ssl
import time
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tiktoken

from quantaalpha.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass
from quantaalpha.log import LogColors, logger
from quantaalpha.log import logger
from quantaalpha.llm.config import LLM_SETTINGS

DEFAULT_QLIB_DOT_PATH = Path("./")


def _message_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _split_system_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Extract OpenAI-style system messages for Anthropic Messages API."""
    system_chunks: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            text = _message_content_to_str(m.get("content", "")).strip()
            if text:
                system_chunks.append(text)
        elif role in ("user", "assistant"):
            out.append(dict(m))
        else:
            out.append(dict(m))
    system = "\n\n".join(system_chunks).strip() or None
    return system, out


def _merge_system_into_user_messages(messages: list[dict], system: str) -> list[dict]:
    if not system.strip():
        return [dict(m) for m in messages]
    out = [dict(m) for m in messages]
    for i, m in enumerate(out):
        if m.get("role") == "user":
            prev = _message_content_to_str(m.get("content", ""))
            out[i] = {**m, "content": f"{system}\n\n---\n\n{prev}"}
            return out
    out.insert(0, {"role": "user", "content": system})
    return out


def _normalize_openai_base_url(base_url: str) -> str:
    """
    OpenAI SDK 的 base_url 须以 ``/v1`` 结尾；仅填网关根域名时会打到网站首页并返回 HTML 字符串。
    """
    u = (base_url or "").strip().rstrip("/")
    if not u:
        return "https://api.anthropic.com/v1"
    if u.endswith("/v1"):
        return u
    if "://" in u and not re.search(r"/v\d+(?:/|$)", u):
        logger.warning(
            "ANTHROPIC_BASE_URL 缺少 /v1 后缀，已自动补全: %s -> %s/v1",
            u,
            u,
        )
        return f"{u}/v1"
    return u


def _parse_chat_completion_response(response: Any, *, base_url: str) -> tuple[str, str | None]:
    """从 OpenAI ChatCompletion 或网关误返回的 HTML/字符串中提取文本。"""
    if isinstance(response, str):
        snippet = response.strip()[:200].replace("\n", " ")
        raise RuntimeError(
            "LLM 网关返回了非 JSON 响应（多为 base_url 未带 /v1，请求到了网站首页）。"
            f" 请设置 ANTHROPIC_BASE_URL={_normalize_openai_base_url(base_url)} 。"
            f" 响应片段: {snippet!r}"
        )
    if not getattr(response, "choices", None):
        raise RuntimeError(f"Unexpected LLM response type: {type(response)!r}")
    choice = response.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None) or ""
    finish_reason = getattr(choice, "finish_reason", None)
    return content, finish_reason


def _should_use_system_top_level(base_url: str, *, force: bool, merge_into_user: bool) -> bool:
    if force or merge_into_user:
        return True
    setting = LLM_SETTINGS.chat_system_top_level
    if setting is True:
        return True
    if setting is False:
        return False
    host = (base_url or "").lower()
    return "packyapi" in host or "anthropic" in host or "codesuc" in host


def _prepare_messages_for_chat_api(
    messages: list[dict],
    *,
    base_url: str,
    force_system_top_level: bool = False,
    merge_system_into_user: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """
    PackyAPI / Anthropic Messages API 不接受 messages 中的 role=system；
    将 system 挪到顶层 extra_body，或合并进首条 user。
    """
    if not _should_use_system_top_level(
        base_url, force=force_system_top_level, merge_into_user=merge_system_into_user
    ):
        return messages, {}

    system, api_messages = _split_system_messages(messages)
    if not system:
        return messages, {}

    if merge_system_into_user:
        return _merge_system_into_user_messages(api_messages, system), {}

    return api_messages, {"system": system}


def md5_hash(input_string: str) -> str:
    hash_md5 = hashlib.md5(usedforsecurity=False)
    input_bytes = input_string.encode("utf-8")
    hash_md5.update(input_bytes)
    return hash_md5.hexdigest()


def _extract_outer_brace_object(text: str) -> str | None:
    """Extract first top-level {...} respecting both ' and \" strings."""
    start = text.find("{")
    if start < 0:
        return None
    in_str: str | None = None
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalize_jsonish(text: str) -> str:
    """Loosen common LLM JSON quirks for literal_eval / json.loads."""
    s = text.strip()
    s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnull\b", "None", s, flags=re.IGNORECASE)
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _try_parse_dict_literal(text: str) -> dict | None:
    """Parse Python dict literals (single-quoted keys/strings) from model output."""
    s = _normalize_jsonish((text or "").strip())
    if not s.startswith("{"):
        return None
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return {str(k): v for k, v in obj.items()}
    except (ValueError, SyntaxError, TypeError):
        return None
    return None


def _regex_extract_json_field(text: str, key: str) -> str | None:
    """Best-effort field extraction from malformed LLM JSON-ish text."""
    patterns = [
        rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
        rf'"{re.escape(key)}"\s*:\s*\'((?:[^\'\\]|\\.)*)\'',
        rf"'{re.escape(key)}'\s*:\s*'((?:[^'\\]|\\.)*)'",
        rf"'{re.escape(key)}'\s*:\s*\"((?:[^\"\\]|\\.)*)\"",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n").replace('\\"', '"')
    return None


def _strip_markdown_inline(text: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("---")]
    return " ".join(lines)


def _extract_markdown_section(text: str, title: str) -> str | None:
    pat = rf"#{1,3}\s*{re.escape(title)}\s*\n+(.*?)(?=\n#{1,3}\s|\n---|\Z)"
    m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: next markdown H2/H3 or horizontal rule
    pat2 = rf"##\s*{re.escape(title)}\s*\n+(.*?)(?=\n##|\n---|\Z)"
    m2 = re.search(pat2, text, re.DOTALL | re.IGNORECASE)
    return m2.group(1).strip() if m2 else None


def _markdown_structured_fallback(text: str) -> dict | None:
    """PackyAPI / Claude sometimes returns ## Hypothesis markdown instead of JSON."""
    hyp = (
        _extract_markdown_section(text, "Hypothesis")
        or _extract_markdown_section(text, "Formal Hypothesis")
        or _extract_markdown_section(text, "Research Hypothesis")
    )
    if not hyp:
        m = re.search(
            r"##\s+([^\n]*[Hh]ypothesis[^\n]*)\s*\n+(.*?)(?=\n##|\n---|\Z)",
            text,
            re.DOTALL,
        )
        if m:
            hyp = m.group(2).strip()
    if not hyp:
        return None
    reasoning = _extract_markdown_section(text, "Reasoning Keys") or ""
    return {
        "hypothesis": _strip_markdown_inline(hyp)[:4000],
        "concise_observation": _strip_markdown_inline(
            _extract_markdown_section(text, "Observation") or hyp
        )[:500],
        "concise_knowledge": _strip_markdown_inline(
            _extract_markdown_section(text, "Knowledge") or reasoning or hyp
        )[:500],
        "concise_justification": _strip_markdown_inline(
            _extract_markdown_section(text, "Justification") or reasoning or hyp
        )[:500],
        "concise_specification": _strip_markdown_inline(
            _extract_markdown_section(text, "Specification") or reasoning or hyp
        )[:500],
    }


def _fallback_extract_json_fields(text: str, keys: list[str] | None = None) -> dict | None:
    keys = keys or [
        "hypothesis",
        "concise_observation",
        "concise_knowledge",
        "concise_justification",
        "concise_specification",
        "reason",
        "concise_reason",
    ]
    out: dict[str, str] = {}
    for key in keys:
        val = _regex_extract_json_field(text, key)
        if val is not None and val.strip():
            out[key] = val.strip()
    if out.get("hypothesis") or len(out) >= 2:
        return out
    return None


def _try_parse_json_candidates(text: str) -> dict | None:
    for candidate in (
        text,
        _extract_outer_brace_object(text),
        text[text.find("{") : text.rfind("}") + 1] if "{" in text and "}" in text else None,
    ):
        if not candidate:
            continue
        block = candidate.strip()
        for parser in (json.loads, _try_parse_dict_literal):
            try:
                if parser is json.loads:
                    obj = json.loads(block)
                else:
                    obj = parser(block)
                if isinstance(obj, dict) and obj:
                    return obj
            except (json.JSONDecodeError, TypeError):
                continue
        norm = _normalize_jsonish(block)
        try:
            obj = json.loads(norm)
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            pass
        parsed = _try_parse_dict_literal(norm)
        if parsed:
            return parsed
    return None


def robust_json_parse(text: str, max_retries: int = 3) -> dict:
    """
    Robust JSON parser: handles extra data, LaTeX escapes, markdown-wrapped JSON.
    Raises json.JSONDecodeError if all strategies fail.
    """
    original_text = text

    # Markdown sections (PackyAPI / Claude often ignore JSON instructions)
    if original_text.lstrip().startswith("#") or re.search(
        r"(?m)^#{1,3}\s+.*[Hh]ypothesis", original_text
    ):
        parsed = _markdown_structured_fallback(original_text)
        if parsed is not None:
            logger.info("robust_json_parse: recovered fields from markdown sections")
            return parsed

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    parsed = _try_parse_dict_literal(text)
    if parsed is not None:
        return parsed
    
    # Strategy 2: extract JSON code block
    json_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_block_pattern, text)
    if matches:
        for match in matches:
            block = match.strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                parsed = _try_parse_dict_literal(block)
                if parsed is not None:
                    return parsed
                continue
    
    # Strategy 3: find first complete JSON object (extra data)
    brace_count = 0
    start_idx = -1
    end_idx = -1
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
            
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                end_idx = i
                break
    
    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            parsed = _try_parse_dict_literal(json_str)
            if parsed is not None:
                return parsed
            # Strategy 4: fix LaTeX escapes
            fixed_str = json_str
            latex_commands = ['text', 'frac', 'left', 'right', 'times', 'cdot', 'sqrt', 
                              'sum', 'prod', 'int', 'alpha', 'beta', 'gamma', 'delta']
            for cmd in latex_commands:
                fixed_str = re.sub(r'(?<!\\)\\(' + cmd + r')', r'\\\\\1', fixed_str)
            fixed_str = re.sub(r'(?<!\\)\\([_\{\}\[\]])', r'\\\\\1', fixed_str)
            
            try:
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                parsed = _try_parse_dict_literal(fixed_str)
                if parsed is not None:
                    return parsed
    
    # Strategy 5: looser JSON extraction
    potential_jsons = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    for pj in potential_jsons:
        try:
            result = json.loads(pj)
            if isinstance(result, dict) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            parsed = _try_parse_dict_literal(pj)
            if parsed is not None and len(parsed) > 0:
                return parsed
            continue

    parsed = _try_parse_json_candidates(original_text)
    if parsed is not None:
        return parsed

    parsed = _fallback_extract_json_fields(original_text)
    if parsed is not None:
        logger.info("robust_json_parse: recovered fields via regex fallback")
        return parsed

    parsed = _markdown_structured_fallback(original_text)
    if parsed is not None:
        logger.info("robust_json_parse: recovered fields from markdown sections")
        return parsed

    preview = original_text[:300].replace("\n", "\\n")
    logger.warning(
        "robust_json_parse failed (len=%d), preview: %s",
        len(original_text),
        preview,
    )
    raise json.JSONDecodeError(
        f"Could not parse JSON; original text length: {len(original_text)}",
        original_text,
        0,
    )


try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:
    logger.warning("azure.identity is not installed.")

try:
    import openai
except ImportError:
    logger.warning("openai is not installed.")

try:
    from llama import Llama
except ImportError:
    logger.warning("llama is not installed.")


class ConvManager:
    """
    This is a conversation manager of LLM
    It is for convenience of exporting conversation for debugging.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_QLIB_DOT_PATH / "llm_conv",
        recent_n: int = 10,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.recent_n = recent_n

    def _rotate_files(self) -> None:
        pairs = []
        for f in self.path.glob("*.json"):
            m = re.match(r"(\d+).json", f.name)
            if m is not None:
                n = int(m.group(1))
                pairs.append((n, f))
        pairs.sort(key=lambda x: x[0])
        for n, f in pairs[: self.recent_n][::-1]:
            if (self.path / f"{n+1}.json").exists():
                (self.path / f"{n+1}.json").unlink()
            f.rename(self.path / f"{n+1}.json")

    def append(self, conv: tuple[list, str]) -> None:
        self._rotate_files()
        with (self.path / "0.json").open("w") as file:
            json.dump(conv, file)
        # TODO: reseve line breaks to make it more convient to edit file directly.


class SQliteLazyCache(SingletonBaseClass):
    def __init__(self, cache_location: str) -> None:
        super().__init__()
        self.cache_location = cache_location
        db_file_exist = Path(cache_location).exists()
        # TODO: sqlite3 does not support multiprocessing.
        self.conn = sqlite3.connect(cache_location, timeout=20)
        self.c = self.conn.cursor()
        if not db_file_exist:
            self.c.execute(
                """
                CREATE TABLE chat_cache (
                    md5_key TEXT PRIMARY KEY,
                    chat TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE embedding_cache (
                    md5_key TEXT PRIMARY KEY,
                    embedding TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE message_cache (
                    conversation_id TEXT PRIMARY KEY,
                    message TEXT
                )
                """,
            )
            self.conn.commit()

    def chat_get(self, key: str) -> str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT chat FROM chat_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        if result is None:
            return None
        return result[0]

    def embedding_get(self, key: str) -> list | dict | str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT embedding FROM embedding_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        if result is None:
            return None
        return json.loads(result[0])

    def chat_set(self, key: str, value: str) -> None:
        md5_key = md5_hash(key)
        self.c.execute(
            "INSERT OR REPLACE INTO chat_cache (md5_key, chat) VALUES (?, ?)",
            (md5_key, value),
        )
        self.conn.commit()

    def embedding_set(self, content_to_embedding_dict: dict) -> None:
        for key, value in content_to_embedding_dict.items():
            md5_key = md5_hash(key)
            self.c.execute(
                "INSERT OR REPLACE INTO embedding_cache (md5_key, embedding) VALUES (?, ?)",
                (md5_key, json.dumps(value)),
            )
        self.conn.commit()

    def message_get(self, conversation_id: str) -> list[str]:
        self.c.execute("SELECT message FROM message_cache WHERE conversation_id=?", (conversation_id,))
        result = self.c.fetchone()
        if result is None:
            return []
        return json.loads(result[0])

    def message_set(self, conversation_id: str, message_value: list[str]) -> None:
        self.c.execute(
            "INSERT OR REPLACE INTO message_cache (conversation_id, message) VALUES (?, ?)",
            (conversation_id, json.dumps(message_value)),
        )
        self.conn.commit()


class SessionChatHistoryCache(SingletonBaseClass):
    def __init__(self) -> None:
        """load all history conversation json file from self.session_cache_location"""
        self.cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)

    def message_get(self, conversation_id: str) -> list[str]:
        return self.cache.message_get(conversation_id)

    def message_set(self, conversation_id: str, message_value: list[str]) -> None:
        self.cache.message_set(conversation_id, message_value)


class ChatSession:
    def __init__(self, api_backend: Any, conversation_id: str | None = None, system_prompt: str | None = None) -> None:
        self.conversation_id = str(uuid.uuid4()) if conversation_id is None else conversation_id
        self.system_prompt = system_prompt if system_prompt is not None else LLM_SETTINGS.default_system_prompt
        self.api_backend = api_backend

    def build_chat_completion_message(self, user_prompt: str) -> list[dict[str, Any]]:
        history_message = SessionChatHistoryCache().message_get(self.conversation_id)
        messages = history_message
        if not messages:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_chat_completion_message_and_calculate_token(self, user_prompt: str) -> Any:
        messages = self.build_chat_completion_message(user_prompt)
        return self.api_backend.calculate_token_from_messages(messages)

    def build_chat_completion(self, user_prompt: str, **kwargs: Any) -> str:
        """
        this function is to build the session messages
        user prompt should always be provided
        """
        messages = self.build_chat_completion_message(user_prompt)

        with logger.tag(f"session_{self.conversation_id}"):
            response = self.api_backend._try_create_chat_completion_or_embedding(  # noqa: SLF001
                messages=messages,
                chat_completion=True,
                **kwargs,
            )

        messages.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        SessionChatHistoryCache().message_set(self.conversation_id, messages)
        return response

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def display_history(self) -> None:
        # TODO: Realize a beautiful presentation format for history messages
        pass


class APIBackend:
    """
    This is a unified interface for different backends.

    (xiao) thinks integrate all kinds of API in a single class is not a good design.
    So we should split them into different classes in `oai/backends/` in the future.
    """

    # FIXME: (xiao) We should avoid using self.xxxx.
    # Instead, we can use LLM_SETTINGS directly. If it's difficult to support different backend settings, we can split them into multiple BaseSettings.
    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        chat_api_key: str | None = None,
        chat_model: str | None = None,
        reasoning_model: str | None = None,
        chat_api_base: str | None = None,
        chat_api_version: str | None = None,
        embedding_api_key: str | None = None,
        embedding_model: str | None = None,
        embedding_api_base: str | None = None,
        embedding_api_version: str | None = None,
        use_chat_cache: bool | None = None,
        dump_chat_cache: bool | None = None,
        use_embedding_cache: bool | None = None,
        dump_embedding_cache: bool | None = None,
    ) -> None:
        if LLM_SETTINGS.use_llama2:
            self.generator = Llama.build(
                ckpt_dir=LLM_SETTINGS.llama2_ckpt_dir,
                tokenizer_path=LLM_SETTINGS.llama2_tokenizer_path,
                max_seq_len=LLM_SETTINGS.max_tokens,
                max_batch_size=LLM_SETTINGS.llams2_max_batch_size,
            )
            self.encoder = None
        elif LLM_SETTINGS.use_gcr_endpoint:
            gcr_endpoint_type = LLM_SETTINGS.gcr_endpoint_type
            if gcr_endpoint_type == "llama2_70b":
                self.gcr_endpoint_key = LLM_SETTINGS.llama2_70b_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.llama2_70b_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.llama2_70b_endpoint
            elif gcr_endpoint_type == "llama3_70b":
                self.gcr_endpoint_key = LLM_SETTINGS.llama3_70b_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.llama3_70b_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.llama3_70b_endpoint
            elif gcr_endpoint_type == "phi2":
                self.gcr_endpoint_key = LLM_SETTINGS.phi2_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi2_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi2_endpoint
            elif gcr_endpoint_type == "phi3_4k":
                self.gcr_endpoint_key = LLM_SETTINGS.phi3_4k_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi3_4k_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi3_4k_endpoint
            elif gcr_endpoint_type == "phi3_128k":
                self.gcr_endpoint_key = LLM_SETTINGS.phi3_128k_endpoint_key
                self.gcr_endpoint_deployment = LLM_SETTINGS.phi3_128k_endpoint_deployment
                self.gcr_endpoint = LLM_SETTINGS.phi3_128k_endpoint
            else:
                error_message = f"Invalid gcr_endpoint_type: {gcr_endpoint_type}"
                raise ValueError(error_message)
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.gcr_endpoint_key),
            }
            self.gcr_endpoint_temperature = LLM_SETTINGS.gcr_endpoint_temperature
            self.gcr_endpoint_top_p = LLM_SETTINGS.gcr_endpoint_top_p
            self.gcr_endpoint_do_sample = LLM_SETTINGS.gcr_endpoint_do_sample
            self.gcr_endpoint_max_token = LLM_SETTINGS.gcr_endpoint_max_token
            if not os.environ.get("PYTHONHTTPSVERIFY", "") and hasattr(ssl, "_create_unverified_context"):
                ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
            self.chat_model_map = json.loads(LLM_SETTINGS.chat_model_map)
            self.chat_model = LLM_SETTINGS.chat_model if chat_model is None else chat_model
            self.encoder = None
        else:
            self.use_azure = LLM_SETTINGS.use_azure
            self.chat_use_azure_token_provider = LLM_SETTINGS.chat_use_azure_token_provider
            self.embedding_use_azure_token_provider = LLM_SETTINGS.embedding_use_azure_token_provider
            self.managed_identity_client_id = LLM_SETTINGS.managed_identity_client_id

            # Claude: anthropic_api_key > openai_api_key (alias) > env
            self.chat_api_key = (
                chat_api_key
                or LLM_SETTINGS.chat_openai_api_key
                or LLM_SETTINGS.anthropic_api_key
                or LLM_SETTINGS.openai_api_key
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            ).strip()
            if not self.chat_api_key:
                raise RuntimeError(
                    "未配置 Claude API Key：请在 QuantaAlpha/.env 中设置 ANTHROPIC_API_KEY，"
                    "或执行: grep ANTHROPIC_API_KEY configs/.env.example >> .env 后编辑 .env"
                )
            # OpenAI SDK 亦会读 OPENAI_API_KEY，与 Anthropic 兼容层对齐
            os.environ.setdefault("OPENAI_API_KEY", self.chat_api_key)
            self.embedding_api_key = (
                embedding_api_key
                or LLM_SETTINGS.embedding_openai_api_key
                or LLM_SETTINGS.openai_api_key
                or os.environ.get("OPENAI_API_KEY")
            )
            
            self.base_url = (
                LLM_SETTINGS.anthropic_base_url
                or LLM_SETTINGS.openai_base_url
                or os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            
            self.embedding_base_url = (
                LLM_SETTINGS.embedding_base_url
                or os.environ.get("EMBEDDING_BASE_URL")
            )

            _emb_explicit = (LLM_SETTINGS.embedding_api_key or "").strip() or (
                os.environ.get("EMBEDDING_API_KEY") or ""
            ).strip()
            if _emb_explicit:
                self.embedding_api_key = _emb_explicit


            self.chat_model = LLM_SETTINGS.chat_model if chat_model is None else chat_model
            self.reasoning_model = LLM_SETTINGS.reasoning_model if reasoning_model is None else reasoning_model
            self.chat_model_map = json.loads(LLM_SETTINGS.chat_model_map)

            self.base_url = _normalize_openai_base_url(str(self.base_url or ""))
            if not (self.reasoning_model or "").strip():
                self.reasoning_model = self.chat_model
            # self.encoder = self._get_encoder()
            
            self.chat_api_base = LLM_SETTINGS.chat_azure_api_base if chat_api_base is None else chat_api_base
            self.chat_api_version = (
                LLM_SETTINGS.chat_azure_api_version if chat_api_version is None else chat_api_version
            )
            self.chat_stream = LLM_SETTINGS.chat_stream
            self.chat_seed = LLM_SETTINGS.chat_seed

            self.embedding_model = LLM_SETTINGS.embedding_model if embedding_model is None else embedding_model
            self.embedding_api_base = (
                LLM_SETTINGS.embedding_azure_api_base if embedding_api_base is None else embedding_api_base
            )
            self.embedding_api_version = (
                LLM_SETTINGS.embedding_azure_api_version if embedding_api_version is None else embedding_api_version
            )

            # Claude 不提供 OpenAI 兼容 embedding；未单独配 EMBEDDING_* 时不复用对话 Key
            if (self.embedding_model or "").strip():
                if not (self.embedding_api_key or "").strip():
                    self.embedding_api_key = self.chat_api_key
                if not (self.embedding_base_url or "").strip():
                    self.embedding_base_url = self.base_url

            _want_emb = bool((self.embedding_model or "").strip())
            _azure_emb_ok = self.embedding_use_azure_token_provider or (self.embedding_api_key or "").strip()
            _openai_emb_ok = (self.embedding_api_key or "").strip()

            if self.use_azure:
                if self.chat_use_azure_token_provider or self.embedding_use_azure_token_provider:
                    dac_kwargs = {}
                    if self.managed_identity_client_id is not None:
                        dac_kwargs["managed_identity_client_id"] = self.managed_identity_client_id
                    credential = DefaultAzureCredential(**dac_kwargs)
                    token_provider = get_bearer_token_provider(
                        credential,
                        "https://cognitiveservices.azure.com/.default",
                    )
                if self.chat_use_azure_token_provider:
                    self.chat_client = openai.AzureOpenAI(
                        azure_ad_token_provider=token_provider,
                        api_version=self.chat_api_version,
                        azure_endpoint=self.chat_api_base,
                    )
                else:
                    self.chat_client = openai.AzureOpenAI(
                        api_key=self.chat_api_key,
                        api_version=self.chat_api_version,
                        azure_endpoint=self.chat_api_base,
                    )

                if _want_emb and _azure_emb_ok:
                    if self.embedding_use_azure_token_provider:
                        self.embedding_client = openai.AzureOpenAI(
                            azure_ad_token_provider=token_provider,
                            api_version=self.embedding_api_version,
                            azure_endpoint=self.embedding_api_base,
                        )
                    else:
                        self.embedding_client = openai.AzureOpenAI(
                            api_key=self.embedding_api_key,
                            api_version=self.embedding_api_version,
                            azure_endpoint=self.embedding_api_base,
                        )
                else:
                    self.embedding_client = None
            else:
                self.chat_client = openai.OpenAI(api_key=self.chat_api_key, base_url=self.base_url)
                if _want_emb and _openai_emb_ok:
                    _emb_base = (self.embedding_base_url or "").strip() or (self.base_url or "").strip()
                    self.embedding_client = openai.OpenAI(
                        api_key=self.embedding_api_key,
                        base_url=_emb_base,
                    )
                else:
                    self.embedding_client = None

        self.dump_chat_cache = LLM_SETTINGS.dump_chat_cache if dump_chat_cache is None else dump_chat_cache
        self.use_chat_cache = LLM_SETTINGS.use_chat_cache if use_chat_cache is None else use_chat_cache
        self.dump_embedding_cache = (
            LLM_SETTINGS.dump_embedding_cache if dump_embedding_cache is None else dump_embedding_cache
        )
        self.use_embedding_cache = (
            LLM_SETTINGS.use_embedding_cache if use_embedding_cache is None else use_embedding_cache
        )
        if self.dump_chat_cache or self.use_chat_cache or self.dump_embedding_cache or self.use_embedding_cache:
            self.cache_file_location = LLM_SETTINGS.prompt_cache_path
            self.cache = SQliteLazyCache(cache_location=self.cache_file_location)

        # transfer the config to the class if the config is not supposed to change during the runtime
        self.use_llama2 = LLM_SETTINGS.use_llama2
        self.use_gcr_endpoint = LLM_SETTINGS.use_gcr_endpoint
        self.retry_wait_seconds = LLM_SETTINGS.retry_wait_seconds

    def _get_encoder(self):
        """
        tiktoken.encoding_for_model(self.chat_model) does not cover all cases it should consider.

        This function attempts to handle several edge cases.
        """

        # 1) cases
        def _azure_patch(model: str) -> str:
            """
            When using Azure API, self.chat_model is the deployment name that can be any string.
            For example, it may be `gpt-4o_2024-08-06`. But tiktoken.encoding_for_model can't handle this.
            """
            return model.replace("_", "-")

        model = self.chat_model
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Failed to get encoder. Trying to patch the model name")
            for patch_func in [_azure_patch]:
                try:
                    return tiktoken.encoding_for_model(patch_func(model))
                except KeyError:
                    logger.error(f"Failed to get encoder even after patching with {patch_func.__name__}")
                    raise

    def build_chat_session(
        self,
        conversation_id: str | None = None,
        session_system_prompt: str | None = None,
    ) -> ChatSession:
        """
        conversation_id is a 256-bit string created by uuid.uuid4() and is also
        the file name under session_cache_folder/ for each conversation
        """
        return ChatSession(self, conversation_id, session_system_prompt)

    def build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list[dict] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> list[dict]:
        """
        build the messages to avoid implementing several redundant lines of code

        """
        if former_messages is None:
            former_messages = []
        # shrink multiple break will recursively remove multiple breaks(more than 2)
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            if system_prompt is not None:
                while "\n\n\n" in system_prompt:
                    system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        messages.extend(former_messages[-1 * LLM_SETTINGS.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_messages_and_create_chat_completion(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list | None = None,
        chat_cache_prefix: str = "",
        *,
        shrink_multiple_break: bool = False,
        **kwargs: Any,
    ) -> str:
        if former_messages is None:
            former_messages = []
        messages = self.build_messages(
            user_prompt,
            system_prompt,
            former_messages,
            shrink_multiple_break=shrink_multiple_break,
        )
        return self._try_create_chat_completion_or_embedding(
            messages=messages,
            chat_completion=True,
            chat_cache_prefix=chat_cache_prefix,
            **kwargs,
        )

    def create_embedding(self, input_content: str | list[str], **kwargs: Any) -> list[Any] | Any:
        input_content_list = [input_content] if isinstance(input_content, str) else input_content
        resp = self._try_create_chat_completion_or_embedding(
            input_content_list=input_content_list,
            embedding=True,
            **kwargs,
        )
        if isinstance(input_content, str):
            return resp[0]
        return resp

    def _create_chat_completion_auto_continue(self, messages: list, **kwargs: dict) -> str:
        """
        Call the chat completion function and automatically continue the conversation if the finish_reason is length.
        TODO: This function only continues once, maybe need to continue more than once in the future.
        """
        response, finish_reason = self._create_chat_completion_inner_function(messages=messages, **kwargs)

        if finish_reason == "length":
            new_message = deepcopy(messages)
            new_message.append({"role": "assistant", "content": response})
            new_message.append(
                {
                    "role": "user",
                    "content": "continue the former output with no overlap",
                },
            )
            new_response, finish_reason = self._create_chat_completion_inner_function(messages=new_message, **kwargs)
            return response + new_response
        return response

    def _try_create_chat_completion_or_embedding(
        self,
        max_retry: int = 10,
        *,
        chat_completion: bool = False,
        embedding: bool = False,
        **kwargs: Any,
    ) -> Any:
        assert not (chat_completion and embedding), "chat_completion and embedding cannot be True at the same time"
        max_retry = LLM_SETTINGS.max_retry if LLM_SETTINGS.max_retry is not None else max_retry
        for i in range(max_retry):
            try:
                # import pdb; pdb.set_trace()
                if embedding:
                    return self._create_embedding_inner_function(**kwargs)
                if chat_completion:
                    return self._create_chat_completion_auto_continue(**kwargs)
            except openai.BadRequestError as e:  # noqa: PERF203
                logger.warning(e)
                logger.warning(f"Retrying {i+1}th time...")
                err_msg = str(getattr(e, "message", "") or e)
                if "'messages' must contain the word 'json' in some form" in err_msg:
                    kwargs["add_json_in_prompt"] = True
                elif 'Unexpected role "system"' in err_msg or "Unexpected role 'system'" in err_msg:
                    kwargs["force_system_top_level"] = True
                    if i >= 1:
                        kwargs["merge_system_into_user"] = True
                elif embedding and "maximum context length" in err_msg:
                    kwargs["input_content_list"] = [
                        content[: len(content) // 2] for content in kwargs.get("input_content_list", [])
                    ]
                # Wait before retry to avoid rate limit
                if i < max_retry - 1:
                    time.sleep(self.retry_wait_seconds)
            except Exception as e:  # noqa: BLE001
                logger.warning(e)
                logger.warning(f"Retrying {i+1}th time...")
                if i < max_retry - 1:
                    time.sleep(self.retry_wait_seconds)
        error_message = f"Failed to create chat completion after {max_retry} retries."
        raise RuntimeError(error_message)

    def _create_embedding_inner_function(
        self, input_content_list: list[str], **kwargs: Any
    ) -> list[Any]:  # noqa: ARG002
        if os.environ.get("QUANTALPHA_LLM_STUB", "").strip().lower() in ("1", "true", "yes", "on"):
            dim = 8
            return [[0.0] * dim for _ in input_content_list]

        content_to_embedding_dict = {}
        filtered_input_content_list = []
        if self.use_embedding_cache:
            for content in input_content_list:
                cache_result = self.cache.embedding_get(content)
                if cache_result is not None:
                    content_to_embedding_dict[content] = cache_result
                else:
                    filtered_input_content_list.append(content)
        else:
            filtered_input_content_list = input_content_list

        if self.embedding_client is None:
            if not getattr(self, "_embedding_fallback_warned", False):
                logger.warning(
                    "未配置 embedding 客户端，CoSTEER 向量检索将使用零向量占位（不影响因子计算/回测）。"
                    "可选：在 .env 设置 EMBEDDING_MODEL + EMBEDDING_API_KEY，或 FACTOR_CoSTEER_WITH_KNOWLEDGE=false"
                )
                self._embedding_fallback_warned = True
            dim = 8
            for content in filtered_input_content_list:
                content_to_embedding_dict[content] = [0.0] * dim
            return [content_to_embedding_dict[content] for content in input_content_list]

        if len(filtered_input_content_list) > 0:
            # Adjust batch size by model (DashScope text-embedding-v4 is slower)
            batch_size = LLM_SETTINGS.embedding_max_str_num
            batch_wait_seconds = LLM_SETTINGS.embedding_batch_wait_seconds
            batches = [
                filtered_input_content_list[i : i + batch_size]
                for i in range(0, len(filtered_input_content_list), batch_size)
            ]
            
            for batch_idx, sliced_filtered_input_content_list in enumerate(batches):
                if self.use_azure:
                    response = self.embedding_client.embeddings.create(
                        model=self.embedding_model,
                        input=sliced_filtered_input_content_list,
                    )
                else:
                    response = self.embedding_client.embeddings.create(
                        model=self.embedding_model,
                        input=sliced_filtered_input_content_list,
                    )
                for index, data in enumerate(response.data):
                    content_to_embedding_dict[sliced_filtered_input_content_list[index]] = data.embedding

                if self.dump_embedding_cache:
                    self.cache.embedding_set(content_to_embedding_dict)
                
                # Wait between batches to avoid API overload
                if batch_idx < len(batches) - 1 and batch_wait_seconds > 0:
                    time.sleep(batch_wait_seconds)
        return [content_to_embedding_dict[content] for content in input_content_list]

    def _build_log_messages(self, messages: list[dict], max_prompt_length: int = 100) -> str:
        """Build log string from messages (content truncated to max_prompt_length)."""
        log_messages = ""
        for m in messages:
            role = m['role']
            content = m['content']
            if len(content) > max_prompt_length:
                display_content = content[:max_prompt_length] + f"... [{len(content)} chars]"
            else:
                display_content = content
            
            log_messages += (
                f"\n{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END}"
                f"{LogColors.CYAN}{role}{LogColors.END}\n"
                f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                f"{LogColors.CYAN}{display_content}{LogColors.END}\n"
            )
        return log_messages

    def _create_chat_completion_inner_function(  # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict],
        reasoning_flag = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        chat_cache_prefix: str = "",
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        *,
        json_mode: bool = False,
        add_json_in_prompt: bool = False,
        seed: Optional[int] = None,
        force_system_top_level: bool = False,
        merge_system_into_user: bool = False,
    ) -> str:
        """
        seed : Optional[int]
            When retrying with cache enabled, it will keep returning the same results.
            To make retries useful, we need to enable a seed.
            This seed is different from `self.chat_seed` for GPT. It is for the local cache mechanism enabled by QuantaAlpha locally.
        """
        if seed is None and LLM_SETTINGS.use_auto_chat_cache_seed_gen:
            seed = LLM_CACHE_SEED_GEN.get_next_seed()

        # reasoning_flag clears json_mode and breaks PackyAPI/Claude JSON calls; prefer chat path.
        if json_mode and reasoning_flag:
            reasoning_flag = False

        # TODO: we can add this function back to avoid so much `self.cfg.log_llm_chat_content`
        if LLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        from quantaalpha.llm.stub_chat import maybe_stub_chat_response

        stub_text, stub_fin = maybe_stub_chat_response(
            messages, json_mode=json_mode, reasoning_flag=reasoning_flag
        )
        if stub_text is not None:
            return stub_text, stub_fin or "stop"

        # TODO: fail to use loguru adaptor due to stream response
        input_content_json = json.dumps(messages)
        input_content_json = (
            chat_cache_prefix + input_content_json + f"<seed={seed}/>"
        )  # FIXME this is a hack to make sure the cache represents the round index
        if self.use_chat_cache:
            cache_result = self.cache.chat_get(input_content_json)
            if cache_result is not None:
                if LLM_SETTINGS.log_llm_chat_content:
                    display_cr = cache_result[:200] + f"... [{len(cache_result)} chars]" if len(cache_result) > 200 else cache_result
                    logger.info(f"{LogColors.CYAN}Response(cached):{display_cr}{LogColors.END}", tag="llm_messages")
                return cache_result, None

        if temperature is None:
            temperature = LLM_SETTINGS.chat_temperature
        if max_tokens is None:
            max_tokens = LLM_SETTINGS.chat_max_tokens
        cap = LLM_SETTINGS.chat_max_tokens_cap
        if cap and cap > 0:
            max_tokens = min(int(max_tokens), int(cap))
        if frequency_penalty is None:
            frequency_penalty = LLM_SETTINGS.chat_frequency_penalty
        if presence_penalty is None:
            presence_penalty = LLM_SETTINGS.chat_presence_penalty

        # Use index 4 to skip the current function and intermediate calls,
        # and get the locals of the caller's frame.
        caller_locals = inspect.stack()[4].frame.f_locals
        if "self" in caller_locals:
            tag = caller_locals["self"].__class__.__name__
        else:
            tag = inspect.stack()[4].function
            
        if reasoning_flag:
            model = self.reasoning_model
            json_mode = None
        else:
            model = self.chat_model_map.get(tag, self.chat_model)

        finish_reason = None
        if self.use_llama2:
            response = self.generator.chat_completion(
                messages,  # type: ignore
                max_gen_len=max_tokens,
                temperature=temperature,
            )
            resp = response[0]["generation"]["content"]
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
        elif self.use_gcr_endpoint:
            body = str.encode(
                json.dumps(
                    {
                        "input_data": {
                            "input_string": messages,
                            "parameters": {
                                "temperature": self.gcr_endpoint_temperature,
                                "top_p": self.gcr_endpoint_top_p,
                                "max_new_tokens": self.gcr_endpoint_max_token,
                            },
                        },
                    },
                ),
            )

            req = urllib.request.Request(self.gcr_endpoint, body, self.headers)  # noqa: S310
            response = urllib.request.urlopen(req)  # noqa: S310
            resp = json.loads(response.read().decode())["output"]
            if LLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.CYAN}Response:{resp}{LogColors.END}", tag="llm_messages")
        else:
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=self.chat_stream,
                seed=self.chat_seed,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            if LLM_SETTINGS.claude_omit_unsupported_params:
                kwargs.pop("seed", None)
                kwargs.pop("frequency_penalty", None)
                kwargs.pop("presence_penalty", None)

            if json_mode:
                skip_rf = LLM_SETTINGS.claude_skip_json_response_format
                for message in messages:
                    if message["role"] == "system":
                        message["content"] = (
                            message["content"]
                            + "\n\nCRITICAL: Reply with ONE valid JSON object only. "
                            "No markdown fences, no explanation before or after."
                        )
                        break
                if skip_rf or add_json_in_prompt:
                    for message in messages[::-1]:
                        if message["role"] == "user":
                            message["content"] = (
                                message["content"] + "\nPlease respond in valid JSON only."
                            )
                            break
                if not skip_rf:
                    kwargs["response_format"] = {"type": "json_object"}

            api_messages, system_extra = _prepare_messages_for_chat_api(
                messages,
                base_url=self.base_url,
                force_system_top_level=force_system_top_level,
                merge_system_into_user=merge_system_into_user,
            )
            kwargs["messages"] = api_messages
            if system_extra:
                kwargs["extra_body"] = system_extra

            response = self.chat_client.chat.completions.create(**kwargs)

            
            if self.chat_stream:
                resp = ""
                for chunk in response:
                    if not getattr(chunk, "choices", None):
                        continue
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        resp += content
                    if getattr(choice, "finish_reason", None) is not None:
                        finish_reason = choice.finish_reason

                if LLM_SETTINGS.log_llm_chat_content:
                    display_resp = resp[:200] + f"... [{len(resp)} chars]" if len(resp) > 200 else resp
                    logger.info(f"{LogColors.CYAN}Response:{display_resp}{LogColors.END}", tag="llm_messages")

            else:
                resp, finish_reason = _parse_chat_completion_response(
                    response, base_url=self.base_url
                )
                if LLM_SETTINGS.log_llm_chat_content:
                    display_resp = resp[:200] + f"... [{len(resp)} chars]" if len(resp) > 200 else resp
                    logger.info(f"{LogColors.CYAN}Response:{display_resp}{LogColors.END}", tag="llm_messages")
                    logger.info(
                        json.dumps(
                            {
                                "tag": tag,
                                "total_tokens": response.usage.total_tokens,
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "model": model,
                            }
                        ),
                        tag="llm_messages",
                    )
            if json_mode or reasoning_flag:
                if not (resp or "").strip():
                    logger.warning("LLM returned empty content (json_mode=%s)", json_mode)
                else:
                    try:
                        parsed = robust_json_parse(resp)
                        resp = json.dumps(parsed, ensure_ascii=False)
                    except json.JSONDecodeError:
                        logger.warning(
                            "json_mode response not normalized (len=%d), passing raw to downstream parser",
                            len(resp),
                        )
        if not (resp or "").strip() and not self.use_llama2 and not self.use_gcr_endpoint:
            fb_messages, _ = _prepare_messages_for_chat_api(
                deepcopy(messages),
                base_url=self.base_url,
                force_system_top_level=True,
                merge_system_into_user=merge_system_into_user,
            )
            for message in fb_messages:
                if message["role"] == "system":
                    message["content"] = (
                        message["content"]
                        + "\n\nCRITICAL: Reply with ONE valid JSON object only. "
                        "No markdown fences, no explanation before or after."
                    )
                    break
            for message in fb_messages[::-1]:
                if message["role"] == "user":
                    message["content"] = message["content"] + "\nPlease respond in valid JSON only."
                    break
            kwargs_fb = dict(
                model=model,
                messages=fb_messages,
                max_tokens=min(int(max_tokens or 4096), 8192),
                temperature=temperature,
                stream=False,
            )
            if LLM_SETTINGS.claude_omit_unsupported_params:
                kwargs_fb.pop("seed", None)
            try:
                logger.warning(
                    "Empty LLM response; retrying once with prompt-only JSON (max_tokens=%s)",
                    kwargs_fb["max_tokens"],
                )
                response_fb = self.chat_client.chat.completions.create(**kwargs_fb)
                resp, finish_reason = _parse_chat_completion_response(
                    response_fb, base_url=self.base_url
                )
            except Exception as fb_err:  # noqa: BLE001
                logger.warning("Empty-response fallback failed: %s", fb_err)
        if not (resp or "").strip():
            raise ValueError("Empty LLM response from chat completion")
        if self.dump_chat_cache:
            self.cache.chat_set(input_content_json, resp)
        return resp, finish_reason

    def calculate_token_from_messages(self, messages: list[dict]) -> int:
        return 0
        if self.use_llama2 or self.use_gcr_endpoint:
            logger.warning("num_tokens_from_messages() is not implemented for model llama2.")
            return 0  # TODO implement this function for llama2

        if "gpt4" in self.chat_model or "gpt-4" in self.chat_model:
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message = 4  # every message follows <start>{role/name}\n{content}<end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoder.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <start>assistant<message>
        return num_tokens

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: str | None,
        former_messages: list[dict] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        if former_messages is None:
            former_messages = []
        messages = self.build_messages(
            user_prompt, system_prompt, former_messages, shrink_multiple_break=shrink_multiple_break
        )
        return self.calculate_token_from_messages(messages)


def calculate_embedding_distance_between_str_list(
    source_str_list: list[str],
    target_str_list: list[str],
) -> list[list[float]]:
    if not source_str_list or not target_str_list:
        return [[]]

    embeddings = APIBackend().create_embedding(source_str_list + target_str_list)

    source_embeddings = embeddings[: len(source_str_list)]
    target_embeddings = embeddings[len(source_str_list) :]

    source_embeddings_np = np.array(source_embeddings)
    target_embeddings_np = np.array(target_embeddings)

    source_embeddings_np = source_embeddings_np / np.linalg.norm(source_embeddings_np, axis=1, keepdims=True)
    target_embeddings_np = target_embeddings_np / np.linalg.norm(target_embeddings_np, axis=1, keepdims=True)
    similarity_matrix = np.dot(source_embeddings_np, target_embeddings_np.T)

    return similarity_matrix.tolist()
