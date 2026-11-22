import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import get_current_run_tree, tracing_context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from hiremate_ai.agents.root.memory import get_saver
from hiremate_ai.config import settings
from hiremate_ai.llm import tracing
from hiremate_ai.llm.client import get_llm
from hiremate_ai.llm.fake import FakeLlm, ToolCall
from hiremate_ai.llm.tracing import TracingNotConfiguredError, traced
from hiremate_ai.main import app

CONTRACT_DIR = Path(__file__).resolve().parents[3] / "contract"


def spans(exporter: InMemorySpanExporter) -> tuple[ReadableSpan, ...]:
    tracing._client().flush()
    return exporter.get_finished_spans()


def test_off_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    monkeypatch.setattr(tracing, "_client", client)

    with traced("/agents/report/compose"):
        assert get_current_run_tree() is None
        RunnableLambda(lambda text: text).invoke("리포트")

    client.assert_not_called()


def test_groups_langchain_runs_into_one_trace(langfuse: InMemorySpanExporter) -> None:
    with traced("report/compose"):
        RunnableLambda(lambda text: text).invoke("리포트")

    created = spans(langfuse)
    root = next(span for span in created if span.parent is None)
    child = next(span for span in created if span.name == "RunnableLambda")
    assert root.name == "report/compose"
    assert child.parent is not None and root.context is not None
    assert child.parent.span_id == root.context.span_id
    assert child.context is not None and child.context.trace_id == root.context.trace_id


def test_without_name_each_langchain_run_is_a_trace(langfuse: InMemorySpanExporter) -> None:
    with traced():
        RunnableLambda(lambda text: text).invoke("하나")
        RunnableLambda(lambda text: text).invoke("둘")

    created = spans(langfuse)
    assert len(created) == 2
    assert all(span.parent is None for span in created)
    assert len({span.context.trace_id for span in created if span.context}) == 2


def test_sse_request_is_one_trace_closed_after_stream(langfuse: InMemorySpanExporter) -> None:
    context = json.loads((CONTRACT_DIR / "ChatContext.json").read_text(encoding="utf-8"))
    app.dependency_overrides[get_llm] = lambda: FakeLlm(ToolCall("edit", {"action": "CLEAR", "scope": {}}))
    app.dependency_overrides[get_saver] = lambda: InMemorySaver()
    try:
        response = TestClient(app).post("/agents/root/plan", json={"context": context})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    created = spans(langfuse)
    roots = [span for span in created if span.parent is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.name == "/agents/root/plan"
    assert root.context is not None and root.end_time is not None
    assert "LangGraph" in [span.name for span in created]
    assert all(span.context is not None and span.context.trace_id == root.context.trace_id for span in created)
    assert all(span.end_time is not None and span.end_time <= root.end_time for span in created)


def test_routes_without_model_send_nothing(langfuse: InMemorySpanExporter) -> None:
    app.dependency_overrides[get_saver] = lambda: InMemorySaver()
    try:
        response = TestClient(app).post("/agents/root/history", json={"threadId": "대화방"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert spans(langfuse) == ()


@pytest.mark.parametrize("missing", ["public_key", "secret_key", "base_url"])
def test_on_without_connection_settings_fails(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_tracing_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", SecretStr("test-only"))
    monkeypatch.setattr(settings, "langfuse_base_url", "http://localhost:3000")
    monkeypatch.setattr(settings, f"langfuse_{missing}", SecretStr("") if missing == "secret_key" else "")
    tracing._client.cache_clear()

    with pytest.raises(TracingNotConfiguredError, match="LANGFUSE"), traced("report/compose"):
        pass


def test_client_passes_explicit_connection_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", SecretStr("test-only"))
    monkeypatch.setattr(settings, "langfuse_base_url", "http://localhost:3000/")
    monkeypatch.setattr(settings, "langfuse_environment", "test")
    client = MagicMock(spec=Langfuse)
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(tracing, "Langfuse", factory)
    tracing._client.cache_clear()
    try:
        assert tracing._client() is client
        parameters = factory.call_args.kwargs
        assert parameters["base_url"] == "http://localhost:3000"
        assert parameters["public_key"] == "pk-test"
        assert parameters["secret_key"] == "test-only"
        assert parameters["environment"] == "test"
        assert parameters["tracing_enabled"] is True
        assert parameters["tracer_provider"] is not None
    finally:
        tracing._client.cache_clear()


async def test_parallel_requests_keep_separate_traces(langfuse: InMemorySpanExporter) -> None:
    async def invoke(name: str) -> None:
        with traced(name):
            await asyncio.sleep(0)
            await RunnableLambda(lambda text: text).ainvoke(name)

    await asyncio.gather(invoke("request-a"), invoke("request-b"))

    created = spans(langfuse)
    roots = [span for span in created if span.parent is None]
    assert {span.name for span in roots} == {"request-a", "request-b"}
    assert len({span.context.trace_id for span in roots if span.context}) == 2
    for root in roots:
        assert root.context is not None
        children = [span for span in created if span.parent and span.parent.span_id == root.context.span_id]
        assert len(children) == 1


def test_callback_does_not_escape_the_request(langfuse: InMemorySpanExporter) -> None:
    with traced("inside"):
        RunnableLambda(lambda text: text).invoke("기록")
    before = spans(langfuse)

    RunnableLambda(lambda text: text).invoke("기록하지 않음")

    assert spans(langfuse) == before


def test_legacy_langsmith_context_is_disabled(langfuse: InMemorySpanExporter) -> None:
    with tracing_context(enabled=True), traced("langfuse-only"):
        assert get_current_run_tree() is None
        RunnableLambda(lambda text: text).invoke("가상 자료")

    assert len(spans(langfuse)) == 2


def test_error_is_recorded_and_callback_is_restored(langfuse: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="synthetic failure"), traced("failed-request"):
        raise ValueError("synthetic failure")

    [root] = spans(langfuse)
    assert root.status.status_code is StatusCode.ERROR
    RunnableLambda(lambda text: text).invoke("요청 밖")
    assert spans(langfuse) == (root,)


def test_callback_uses_the_same_normalized_key(langfuse: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", f" {settings.langfuse_public_key} ")

    with traced("normalized-key"):
        RunnableLambda(lambda text: text).invoke("가상 자료")

    assert {span.name for span in spans(langfuse)} == {"normalized-key", "RunnableLambda"}


async def test_generation_records_provider_tokens(langfuse: InMemorySpanExporter) -> None:
    response = {
        "id": "chatcmpl-tracing-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "요약"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=response))
    async with httpx.AsyncClient(transport=transport) as http:
        model = ChatOpenAI(
            model="test-model", api_key=SecretStr("test-only"), base_url="http://llm.test/v1", http_async_client=http
        )
        with traced("tokens"):
            await model.ainvoke("가상 자료")

    generations = [s for s in spans(langfuse) if (s.attributes or {}).get("langfuse.observation.type") == "generation"]
    assert len(generations) == 1
    attributes = generations[0].attributes or {}
    assert attributes["langfuse.observation.model.name"] == "test-model"
    assert json.loads(str(attributes["langfuse.observation.usage_details"])) == {"input": 12, "output": 5, "total": 17}
