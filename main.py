"""LLM fallback plugin for KiraAI.

The plugin only builds an event-scoped model group. KiraAI's native
AgentExecutor remains responsible for calling models and handling failover.
"""

from __future__ import annotations

from typing import Optional

from core.chat.message_utils import KiraMessageBatchEvent
from core.logging_manager import get_logger
from core.plugin import BasePlugin, Priority, on
from core.provider import (
    LLMModelClient,
    LLMRequest,
    LLMResponse,
    ProviderAPIError,
)


logger = get_logger("kira-ai-plugin-llm-fallback", "cyan")


class _ValidatedLLMClient(LLMModelClient):
    """Convert unusable provider responses into native failover exceptions."""

    def __init__(self, client: LLMModelClient):
        super().__init__(client.model)
        self._client = client

    async def chat(self, request: LLMRequest, **kwargs) -> LLMResponse:
        response = await self._client.chat(request, **kwargs)

        if response is None:
            raise ProviderAPIError(
                f"Model {self.model.model_id} returned None instead of an LLMResponse"
            )

        if not isinstance(response, LLMResponse):
            raise ProviderAPIError(
                f"Model {self.model.model_id} returned an invalid response type: "
                f"{type(response).__name__}"
            )

        # A useful response must either communicate content or request a tool.
        # Explicit KiraAI silence such as <msg/> remains valid non-empty text.
        text = response.text_response
        has_text = isinstance(text, str) and bool(text.strip())
        if not has_text and not response.tool_calls:
            raise ProviderAPIError(
                f"Model {self.model.model_id} returned an empty LLMResponse"
            )

        return response


class LLMFallbackPlugin(BasePlugin):
    """Append configured fallback LLMs to the current event model group."""

    _CONFIG_KEYS = ("fallback_model_1", "fallback_model_2")

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._fallback_model_uuids: tuple[str, ...] = ()
        self._warned_unavailable: set[str] = set()

    async def initialize(self):
        """Normalize configuration without retaining model client instances."""
        configured: list[str] = []
        seen: set[str] = set()

        for key in self._CONFIG_KEYS:
            model_uuid = self._normalize_model_uuid(self.plugin_cfg.get(key))
            if not model_uuid or model_uuid in seen:
                continue
            seen.add(model_uuid)
            configured.append(model_uuid)

        self._fallback_model_uuids = tuple(configured)
        self._warned_unavailable.clear()

        if configured:
            logger.info(
                "[LLMFallback] Initialized with %d fallback model(s): %s",
                len(configured),
                " -> ".join(configured),
            )
        else:
            logger.warning(
                "[LLMFallback] No fallback model is configured; the plugin will remain inactive"
            )

    async def terminate(self):
        """Release the small amount of in-memory configuration state."""
        self._fallback_model_uuids = ()
        self._warned_unavailable.clear()

    @on.im_batch_message(priority=Priority.LOW)
    async def append_fallback_models(
        self,
        event: KiraMessageBatchEvent,
        *args,
        **kwargs,
    ):
        """Append valid fallback clients while preserving an existing group."""
        if not self._fallback_model_uuids:
            return

        existing_group = [
            client
            for client in (event.model_group or [])
            if isinstance(client, LLMModelClient)
        ]

        if existing_group:
            model_group = list(existing_group)
        else:
            default_client = self._get_default_client()
            if default_client is None:
                return
            model_group = [default_client]

        seen = {self._client_key(client) for client in model_group}
        appended = 0

        for model_uuid in self._fallback_model_uuids:
            client = self._get_configured_client(model_uuid)
            if client is None:
                continue

            key = self._client_key(client)
            if key in seen:
                continue

            seen.add(key)
            model_group.append(client)
            appended += 1

        # Avoid touching the event when configuration adds no effective model.
        if not appended:
            return

        # AgentExecutor already fails over on ProviderAPIError. Wrapping the
        # event-scoped clients lets it apply the same native path when a
        # provider incorrectly returns None or an unusable empty response.
        event.model_group = [
            client
            if isinstance(client, _ValidatedLLMClient)
            else _ValidatedLLMClient(client)
            for client in model_group
        ]
        logger.debug(
            "[LLMFallback] Event %s model group: %s",
            event.event_id,
            " -> ".join(self._client_label(client) for client in model_group),
        )

    @staticmethod
    def _normalize_model_uuid(value) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    @staticmethod
    def _client_key(client: LLMModelClient) -> tuple[str, str]:
        return client.model.provider_id, client.model.model_id

    @staticmethod
    def _client_label(client: LLMModelClient) -> str:
        provider = client.model.provider_name or client.model.provider_id
        return f"{provider}:{client.model.model_id}"

    def _get_default_client(self) -> Optional[LLMModelClient]:
        issue_key = "__default_llm__"
        try:
            client = self.ctx.get_default_llm_client()
        except Exception as exc:
            self._warn_once(
                issue_key,
                "[LLMFallback] Cannot resolve the default LLM; leaving the event unchanged: "
                f"{type(exc).__name__}: {exc}",
            )
            return None

        if not isinstance(client, LLMModelClient):
            self._warn_once(
                issue_key,
                "[LLMFallback] The configured default model is not an LLM client; "
                "leaving the event unchanged",
            )
            return None

        self._warned_unavailable.discard(issue_key)
        return client

    def _get_configured_client(self, model_uuid: str) -> Optional[LLMModelClient]:
        try:
            client = self.ctx.get_llm_client(model_uuid=model_uuid)
        except Exception as exc:
            self._warn_once(
                model_uuid,
                f"[LLMFallback] Cannot resolve fallback model {model_uuid!r}; skipping it: "
                f"{type(exc).__name__}: {exc}",
            )
            return None

        if not isinstance(client, LLMModelClient):
            self._warn_once(
                model_uuid,
                f"[LLMFallback] Fallback model {model_uuid!r} is unavailable or is not an LLM; "
                "skipping it",
            )
            return None

        self._warned_unavailable.discard(model_uuid)
        return client

    def _warn_once(self, issue_key: str, message: str):
        if issue_key in self._warned_unavailable:
            return
        self._warned_unavailable.add(issue_key)
        logger.warning(message)
