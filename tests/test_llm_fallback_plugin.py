from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from core.agent.agent_executor import AgentExecutionContext, AgentExecutor
from core.agent.message import OpenAIMessage
from core.provider import (
    LLMModelClient,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ModelType,
    ProviderAPIError,
)
PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODULE_NAME = "kira_ai_plugin_llm_fallback_tests"
MODULE_SPEC = importlib.util.spec_from_file_location(MODULE_NAME, PLUGIN_DIR / "main.py")
assert MODULE_SPEC and MODULE_SPEC.loader
PLUGIN_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_NAME] = PLUGIN_MODULE
MODULE_SPEC.loader.exec_module(PLUGIN_MODULE)
LLMFallbackPlugin = PLUGIN_MODULE.LLMFallbackPlugin


class ScriptedLLM(LLMModelClient):
    def __init__(
        self,
        provider_id: str,
        model_id: str,
        outcomes: list[LLMResponse | Exception] | None = None,
    ):
        super().__init__(
            ModelInfo(
                model_type=ModelType.LLM,
                model_id=model_id,
                provider_id=provider_id,
                provider_name=provider_id,
            )
        )
        self.outcomes = list(outcomes or [])
        self.calls: list[LLMRequest] = []

    async def chat(self, request: LLMRequest, **kwargs) -> LLMResponse:
        self.calls.append(request)
        if not self.outcomes:
            raise AssertionError(f"Unexpected call to {self.model.model_id}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakePluginContext:
    def __init__(
        self,
        default_client: LLMModelClient | None,
        clients: dict[str, LLMModelClient | None] | None = None,
        default_error: Exception | None = None,
    ):
        self.default_client = default_client
        self.clients = clients or {}
        self.default_error = default_error
        self.resolved: list[str] = []

    def get_default_llm_client(self):
        if self.default_error:
            raise self.default_error
        return self.default_client

    def get_llm_client(self, model_uuid: str):
        self.resolved.append(model_uuid)
        return self.clients.get(model_uuid)


class FakeToolManager:
    async def execute_tool(self, event, resp: LLMResponse, tool_set=None):
        resp.tool_results = [
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "content": "tool-result",
            }
            for tool_call in resp.tool_calls
        ]


def make_event(model_group=None, event_id: str = "event-1"):
    return SimpleNamespace(
        event_id=event_id,
        sid=f"test:dm:{event_id}",
        model_group=[] if model_group is None else model_group,
        is_stopped=False,
    )


def make_plugin(ctx: FakePluginContext, **cfg) -> LLMFallbackPlugin:
    return LLMFallbackPlugin(ctx, cfg)


async def collect_steps(model_group: list[LLMModelClient], max_steps: int = 1):
    event = make_event(model_group=model_group)
    request = LLMRequest(messages=[OpenAIMessage(role="user", content="hello")])
    context = AgentExecutionContext(
        event=event,
        request=request,
        new_messages=[],
        model_group=model_group,
    )
    executor = AgentExecutor(FakeToolManager())
    return [step async for step in executor.run(context, max_steps=max_steps)]


@pytest.mark.asyncio
async def test_builds_default_and_two_fallbacks_in_configured_order():
    primary = ScriptedLLM("primary-provider", "primary")
    fallback_1 = ScriptedLLM("backup-a", "model-a")
    fallback_2 = ScriptedLLM("backup-b", "model-b")
    ctx = FakePluginContext(
        primary,
        {
            "backup-a:model-a": fallback_1,
            "backup-b:model-b": fallback_2,
        },
    )
    plugin = make_plugin(
        ctx,
        fallback_model_1=" backup-a:model-a ",
        fallback_model_2="backup-b:model-b",
    )
    await plugin.initialize()
    event = make_event()

    await plugin.append_fallback_models(event)

    assert event.model_group == [primary, fallback_1, fallback_2]
    assert ctx.resolved == ["backup-a:model-a", "backup-b:model-b"]


@pytest.mark.asyncio
async def test_preserves_existing_group_and_deduplicates_fallbacks():
    default = ScriptedLLM("default-provider", "default")
    custom = ScriptedLLM("custom-provider", "custom")
    existing_backup = ScriptedLLM("backup-a", "model-a")
    duplicate_instance = ScriptedLLM("backup-a", "model-a")
    fallback_2 = ScriptedLLM("backup-b", "model-b")
    ctx = FakePluginContext(
        default,
        {
            "backup-a:model-a": duplicate_instance,
            "backup-b:model-b": fallback_2,
        },
    )
    plugin = make_plugin(
        ctx,
        fallback_model_1="backup-a:model-a",
        fallback_model_2="backup-b:model-b",
    )
    await plugin.initialize()
    event = make_event([custom, existing_backup])

    await plugin.append_fallback_models(event)

    assert event.model_group == [custom, existing_backup, fallback_2]
    assert default not in event.model_group


@pytest.mark.asyncio
async def test_empty_duplicate_and_unavailable_config_leave_event_safe():
    primary = ScriptedLLM("primary-provider", "primary")

    empty_plugin = make_plugin(FakePluginContext(primary))
    await empty_plugin.initialize()
    empty_event = make_event()
    await empty_plugin.append_fallback_models(empty_event)
    assert empty_event.model_group == []

    duplicate_ctx = FakePluginContext(
        primary,
        {"primary-provider:primary": ScriptedLLM("primary-provider", "primary")},
    )
    duplicate_plugin = make_plugin(
        duplicate_ctx,
        fallback_model_1="primary-provider:primary",
        fallback_model_2="primary-provider:primary",
    )
    await duplicate_plugin.initialize()
    duplicate_event = make_event()
    await duplicate_plugin.append_fallback_models(duplicate_event)
    assert duplicate_event.model_group == []
    assert duplicate_ctx.resolved == ["primary-provider:primary"]

    missing_ctx = FakePluginContext(primary, {"missing:model": None})
    missing_plugin = make_plugin(missing_ctx, fallback_model_1="missing:model")
    await missing_plugin.initialize()
    missing_event = make_event()
    await missing_plugin.append_fallback_models(missing_event)
    await missing_plugin.append_fallback_models(missing_event)
    assert missing_event.model_group == []
    assert missing_ctx.resolved == ["missing:model", "missing:model"]


@pytest.mark.asyncio
async def test_unavailable_default_does_not_promote_a_fallback_to_primary():
    fallback = ScriptedLLM("backup", "model")
    ctx = FakePluginContext(
        None,
        {"backup:model": fallback},
        default_error=ValueError("default_llm not set"),
    )
    plugin = make_plugin(ctx, fallback_model_1="backup:model")
    await plugin.initialize()
    event = make_event()

    await plugin.append_fallback_models(event)

    assert event.model_group == []
    assert ctx.resolved == []


@pytest.mark.asyncio
async def test_primary_failure_falls_back_and_preserves_cached_tokens():
    primary = ScriptedLLM(
        "primary-provider",
        "primary",
        [ProviderAPIError("primary unavailable")],
    )
    fallback = ScriptedLLM(
        "backup-provider",
        "backup",
        [LLMResponse("<msg>ok</msg>", input_tokens=30, output_tokens=5, cached_tokens=20)],
    )

    steps = await collect_steps([primary, fallback])

    assert len(steps) == 1
    assert steps[0].state == "success"
    assert steps[0].model_id == "backup"
    assert steps[0].llm_response.text_response == "<msg>ok</msg>"
    assert steps[0].llm_response.cached_tokens == 20
    assert primary.calls[0] is fallback.calls[0]


@pytest.mark.asyncio
async def test_second_failure_uses_third_model():
    primary = ScriptedLLM("p", "primary", [ProviderAPIError("p failed")])
    fallback_1 = ScriptedLLM("f1", "fallback-1", [ProviderAPIError("f1 failed")])
    fallback_2 = ScriptedLLM("f2", "fallback-2", [LLMResponse("<msg>third</msg>")])

    steps = await collect_steps([primary, fallback_1, fallback_2])

    assert steps[0].state == "success"
    assert steps[0].model_id == "fallback-2"
    assert len(primary.calls) == len(fallback_1.calls) == len(fallback_2.calls) == 1


@pytest.mark.asyncio
async def test_programming_error_is_not_converted_to_failover():
    primary = ScriptedLLM("p", "primary", [RuntimeError("programming bug")])
    fallback = ScriptedLLM("f", "fallback", [LLMResponse("must not run")])

    with pytest.raises(RuntimeError, match="programming bug"):
        await collect_steps([primary, fallback])

    assert len(primary.calls) == 1
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_each_agent_step_starts_from_primary_again():
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "test_tool", "arguments": "{}"},
    }
    primary = ScriptedLLM(
        "p",
        "primary",
        [
            ProviderAPIError("temporary outage"),
            LLMResponse("<msg>primary recovered</msg>"),
        ],
    )
    fallback = ScriptedLLM(
        "f",
        "fallback",
        [LLMResponse("", tool_calls=[tool_call])],
    )

    steps = await collect_steps([primary, fallback], max_steps=2)

    assert len(steps) == 2
    assert steps[0].has_tool_calls is True
    assert steps[1].has_tool_calls is False
    assert steps[1].llm_response.text_response == "<msg>primary recovered</msg>"
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_and_new_events_use_independent_group_lists():
    primary = ScriptedLLM("p", "primary")
    fallback = ScriptedLLM("f", "fallback")
    ctx = FakePluginContext(primary, {"f:fallback": fallback})
    plugin = make_plugin(ctx, fallback_model_1="f:fallback")
    await plugin.initialize()
    first = make_event(event_id="first")
    second = make_event(event_id="second")

    await asyncio.gather(
        plugin.append_fallback_models(first),
        plugin.append_fallback_models(second),
    )

    assert first.model_group == [primary, fallback]
    assert second.model_group == [primary, fallback]
    assert first.model_group is not second.model_group

    replacement_primary = ScriptedLLM("p2", "new-primary")
    ctx.default_client = replacement_primary
    third = make_event(event_id="third")
    await plugin.append_fallback_models(third)
    assert third.model_group == [replacement_primary, fallback]


def test_manifest_and_schema_are_valid_and_expose_llm_model_selects():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((PLUGIN_DIR / "schema.json").read_text(encoding="utf-8"))

    assert manifest["plugin_id"] == "kira-ai-plugin-llm-fallback"
    assert PLUGIN_DIR.name == manifest["plugin_id"]
    assert manifest["core_version"] == ">=2.23.0"
    assert "repo" not in manifest
    assert set(schema) == {"fallback_model_1", "fallback_model_2"}
    for field in schema.values():
        assert field["type"] == "model_select"
        assert field["model_type"] == "llm"
        assert field["default"] == ""
        assert {"zh", "en"}.issubset(field["locales"])
