"""
QuantaAlpha LLM configuration.

Claude Sonnet 4.6 via Anthropic OpenAI-compatible API only.
"""

from __future__ import annotations

from pathlib import Path

from quantaalpha.core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict


def _default_env_file() -> str:
    root = Path(__file__).resolve().parents[2]
    return str(root / ".env")


class LLMSettings(ExtendedBaseSettings):
    model_config = ExtendedSettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=_default_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    log_llm_chat_content: bool = True
    max_retry: int = 30
    retry_wait_seconds: int = 15
    dump_chat_cache: bool = False
    use_chat_cache: bool = False
    dump_embedding_cache: bool = False
    use_embedding_cache: bool = False
    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    max_past_message_include: int = 10

    use_auto_chat_cache_seed_gen: bool = False
    init_chat_cache_seed: int = 42

    # Claude / Anthropic (OpenAI SDK compatibility layer)
    anthropic_api_key: str = ""
    """Anthropic API Key；环境变量 ANTHROPIC_API_KEY。"""

    anthropic_base_url: str = "https://api.anthropic.com/v1"
    """OpenAI 兼容端点，见 https://docs.anthropic.com/en/api/openai-sdk"""

    openai_api_key: str = ""
    """与 anthropic_api_key 二选一（历史别名，填 Anthropic Key 即可）。"""

    openai_base_url: str = ""
    """与 anthropic_base_url 二选一。"""

    chat_openai_api_key: str = ""
    chat_model: str = "claude-sonnet-4-6"
    reasoning_model: str = "claude-sonnet-4-6"
    chat_max_tokens: int = 3000
    """Completion token limit; PackyAPI 等对过大值可能返回空 content，见 chat_max_tokens_cap。"""
    chat_max_tokens_cap: int = 16384
    """实际上限 min(chat_max_tokens, cap)，避免中转网关空响应。"""
    chat_temperature: float = 0.5
    chat_stream: bool = True
    chat_seed: int | None = None
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    chat_token_limit: int = 100000
    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions."
    factor_mining_timeout: int = 999999

    # Anthropic 兼容层忽略 response_format / seed / penalties，默认走提示词 JSON
    claude_skip_json_response_format: bool = True
    claude_omit_unsupported_params: bool = True
    # None=自动（packyapi/anthropic 域名启用）；True=system 走顶层 extra_body；False=OpenAI 风格
    chat_system_top_level: bool | None = None

    # Embedding (optional, for RAG — 需单独配置非 Claude 的 embedding 服务)
    embedding_openai_api_key: str = ""
    embedding_model: str = ""
    embedding_max_str_num: int = 3
    embedding_batch_wait_seconds: float = 2.0
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    # Azure (optional)
    use_azure: bool = False
    chat_use_azure_token_provider: bool = False
    embedding_use_azure_token_provider: bool = False
    managed_identity_client_id: str | None = None
    chat_azure_api_base: str = ""
    chat_azure_api_version: str = ""
    embedding_azure_api_base: str = ""
    embedding_azure_api_version: str = ""

    # Offline/endpoint (rarely used)
    use_llama2: bool = False
    use_gcr_endpoint: bool = False

    chat_model_map: str = "{}"


LLM_SETTINGS = LLMSettings()
