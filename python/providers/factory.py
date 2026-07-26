"""
factory.py — Provider factory for the Morrowind AI LLM abstraction layer.

Imports are done lazily inside get_provider() so that missing optional SDKs
(openai, anthropic, etc.) don't crash the process when those providers are
not selected.

Usage:
    from providers.factory import get_provider

    llm = get_provider({"provider": "gemini", "model": "gemini-2.5-flash"})
    resp = await llm.complete(system="...", messages=[...])

Supported provider values:
    "gemini"    — google.generativeai (GOOGLE_API_KEY in ~/.nemoclaw_env)
    "openai"    — openai SDK         (OPENAI_API_KEY in ~/.nemoclaw_env)
    "anthropic" — anthropic SDK      (ANTHROPIC_API_KEY in ~/.nemoclaw_env)
    "ollama"    — local Ollama HTTP  (no auth required)
    "llamacpp"  — llama.cpp server   (no auth by default)
"""

from .base import LLMProvider


def get_provider(provider_cfg: dict) -> LLMProvider:
    """
    Instantiate and return the appropriate LLMProvider.

    Args:
        provider_cfg: Dict with at minimum a "provider" key. Example shapes:

            {"provider": "gemini",    "model": "gemini-2.5-flash"}
            {"provider": "openai",    "model": "gpt-4o"}
            {"provider": "anthropic", "model": "claude-sonnet-4-6"}
            {"provider": "ollama",    "model": "llama3.2",
             "base_url": "http://localhost:11434"}
            {"provider": "llamacpp",  "base_url": "http://localhost:8080"}

    Returns:
        An LLMProvider instance ready to call .complete().

    Raises:
        ValueError: if the provider name is unknown.
        ImportError: if the required SDK for the selected provider is not installed.
    """
    name: str = provider_cfg.get("provider", "").lower()

    if name == "gemini":
        from .gemini_provider import GeminiProvider  # lazy import
        return GeminiProvider(provider_cfg)

    if name == "openai":
        from .openai_provider import OpenAIProvider  # lazy import
        return OpenAIProvider(provider_cfg)

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider  # lazy import
        return AnthropicProvider(provider_cfg)

    if name == "ollama":
        from .ollama_provider import OllamaProvider  # lazy import
        return OllamaProvider(provider_cfg)

    if name == "llamacpp":
        from .llamacpp_provider import LlamaCppProvider  # lazy import
        return LlamaCppProvider(provider_cfg)

    # Локальная модель по OpenAI-совместимому протоколу: LM Studio, Ollama,
    # llama.cpp. С запасным облачным провайдером на случай, когда сервер не
    # поднят — иначе NPC просто замолчат, и это выглядит как поломка мода.
    if name in ("local", "lmstudio", "lm_studio"):
        from .local_provider import LocalProvider  # lazy import
        return LocalProvider(provider_cfg)

    raise ValueError(
        f"Unknown provider '{name}'. "
        "Valid options: gemini, openai, anthropic, ollama, llamacpp, local"
    )
