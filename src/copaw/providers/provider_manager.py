# -*- coding: utf-8 -*-
"""A Manager class to handle all providers, including built-in and custom ones.
It provides a unified interface to manage providers, such as listing available
providers, adding/removing custom providers, and fetching provider details."""

from typing import Dict, List

from copaw.providers.provider import ModelInfo, DefaultProvider, Provider
from copaw.providers.openai_provider import OpenAIProvider
from copaw.providers.anthropic_provider import AnthropicProvider
from copaw.providers.ollama_provider import OllamaProvider


# -------------------------------------------------------
# Built-in provider definitions and their default models.
# -------------------------------------------------------

MODELSCOPE_MODELS: List[ModelInfo] = [
    ModelInfo(
        id="Qwen/Qwen3-235B-A22B-Instruct-2507",
        name="Qwen3-235B-A22B-Instruct-2507",
    ),
    ModelInfo(id="deepseek-ai/DeepSeek-V3.2", name="DeepSeek-V3.2"),
]

DASHSCOPE_MODELS: List[ModelInfo] = [
    ModelInfo(id="qwen3-max", name="Qwen3 Max"),
    ModelInfo(
        id="qwen3-235b-a22b-thinking-2507",
        name="Qwen3 235B A22B Thinking",
    ),
    ModelInfo(id="deepseek-v3.2", name="DeepSeek-V3.2"),
]

ALIYUN_CODINGPLAN_MODELS: List[ModelInfo] = [
    ModelInfo(id="qwen3.5-plus", name="Qwen3.5 Plus"),
    ModelInfo(id="glm-5", name="GLM-5"),
    ModelInfo(id="glm-4.7", name="GLM-4.7"),
    ModelInfo(id="MiniMax-M2.5", name="MiniMax M2.5"),
    ModelInfo(id="kimi-k2.5", name="Kimi K2.5"),
    ModelInfo(id="qwen3-max-2026-01-23", name="Qwen3 Max 2026-01-23"),
    ModelInfo(id="qwen3-coder-next", name="Qwen3 Coder Next"),
    ModelInfo(id="qwen3-coder-plus", name="Qwen3 Coder Plus"),
]

OPENAI_MODELS: List[ModelInfo] = [
    ModelInfo(id="gpt-5.2", name="GPT-5.2"),
    ModelInfo(id="gpt-5", name="GPT-5"),
    ModelInfo(id="gpt-5-mini", name="GPT-5 Mini"),
    ModelInfo(id="gpt-5-nano", name="GPT-5 Nano"),
    ModelInfo(id="gpt-4.1", name="GPT-4.1"),
    ModelInfo(id="gpt-4.1-mini", name="GPT-4.1 Mini"),
    ModelInfo(id="gpt-4.1-nano", name="GPT-4.1 Nano"),
    ModelInfo(id="o3", name="o3"),
    ModelInfo(id="o4-mini", name="o4-mini"),
    ModelInfo(id="gpt-4o", name="GPT-4o"),
    ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini"),
]

AZURE_OPENAI_MODELS: List[ModelInfo] = [
    ModelInfo(id="gpt-5-chat", name="GPT-5 Chat"),
    ModelInfo(id="gpt-5-mini", name="GPT-5 Mini"),
    ModelInfo(id="gpt-5-nano", name="GPT-5 Nano"),
    ModelInfo(id="gpt-4.1", name="GPT-4.1"),
    ModelInfo(id="gpt-4.1-mini", name="GPT-4.1 Mini"),
    ModelInfo(id="gpt-4.1-nano", name="GPT-4.1 Nano"),
    ModelInfo(id="gpt-4o", name="GPT-4o"),
    ModelInfo(id="gpt-4o-mini", name="GPT-4o Mini"),
]

ANTHROPIC_MODELS: List[ModelInfo] = []

PROVIDER_MODELSCOPE = OpenAIProvider(
    id="modelscope",
    name="ModelScope",
    base_url="https://api-inference.modelscope.cn/v1",
    api_key_prefix="ms",
    models=MODELSCOPE_MODELS,
)

PROVIDER_DASHSCOPE = OpenAIProvider(
    id="dashscope",
    name="DashScope",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_prefix="sk",
    models=DASHSCOPE_MODELS,
)

PROVIDER_ALIYUN_CODINGPLAN = OpenAIProvider(
    id="aliyun-codingplan",
    name="Aliyun Coding Plan",
    base_url="https://coding.dashscope.aliyuncs.com/v1",
    api_key_prefix="sk-sp",
    models=ALIYUN_CODINGPLAN_MODELS,
)

PROVIDER_LLAMACPP = DefaultProvider(
    id="llamacpp",
    name="llama.cpp (Local)",
    is_local=True,
)

PROVIDER_MLX = DefaultProvider(
    id="mlx",
    name="MLX (Local, Apple Silicon)",
    is_local=True,
)

PROVIDER_OPENAI = OpenAIProvider(
    id="openai",
    name="OpenAI",
    api_key_prefix="sk-",
    models=OPENAI_MODELS,
)

PROVIDER_AZURE_OPENAI = OpenAIProvider(
    id="azure-openai",
    name="Azure OpenAI",
    api_key_prefix="",
    models=AZURE_OPENAI_MODELS,
)

PROVIDER_ANTHROPIC = AnthropicProvider(
    id="anthropic",
    name="Anthropic",
    api_key_prefix="sk-ant-",
    models=ANTHROPIC_MODELS,
    chat_model="AnthropicChatModel",
)

PROVIDER_OLLAMA = OllamaProvider(
    id="ollama",
    name="Ollama",
)

BUILTIN_PROVIDERS = {
    PROVIDER_MODELSCOPE.id: PROVIDER_MODELSCOPE,
    PROVIDER_DASHSCOPE.id: PROVIDER_DASHSCOPE,
    PROVIDER_ALIYUN_CODINGPLAN.id: PROVIDER_ALIYUN_CODINGPLAN,
    PROVIDER_OPENAI.id: PROVIDER_OPENAI,
    PROVIDER_AZURE_OPENAI.id: PROVIDER_AZURE_OPENAI,
    PROVIDER_ANTHROPIC.id: PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA.id: PROVIDER_OLLAMA,
    PROVIDER_LLAMACPP.id: PROVIDER_LLAMACPP,
    PROVIDER_MLX.id: PROVIDER_MLX,
}


class ProviderManager:
    def __init__(self) -> None:
        # Initialize provider manager, load providers from registry and store
        # any necessary state (e.g., cached models).
        self.custom_providers: Dict[str, Provider] = {}

    def list_providers(self) -> List[Provider]:
        # Return a list of available providers, including both built-in and
        # custom ones. This can be used to populate the UI dropdown.
        return list(BUILTIN_PROVIDERS.values()) + list(
            self.custom_providers.values(),
        )

    def get_provider(self, provider_id: str) -> Provider | None:
        # Return a provider instance by its ID. This will be used to create
        # chat model instances for the agent.
        if provider_id in BUILTIN_PROVIDERS:
            return BUILTIN_PROVIDERS[provider_id]
        if provider_id in self.custom_providers:
            return self.custom_providers[provider_id]
        return None

    def add_custom_provider(self, provider_data: Provider):
        # Add a new custom provider with the given data. This will update the
        # providers.json file and make the new provider available in the UI.
        if provider_data.id in BUILTIN_PROVIDERS:
            raise ValueError(
                f"'{provider_data.id}' conflicts with a built-in provider.",
            )
        provider_data.is_custom = True
        self.custom_providers[provider_data.id] = provider_data

    def remove_custom_provider(self, provider_id: str):
        # Remove a custom provider by its ID. This will update the
        # providers.json file and remove the provider from the UI.
        if provider_id in self.custom_providers:
            del self.custom_providers[provider_id]
