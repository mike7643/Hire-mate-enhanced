import json
from collections.abc import Callable
from contextlib import nullcontext

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hiremate_ai.agents.analysis.application.normalize import DocumentPlan
from hiremate_ai.config import Settings
from hiremate_ai.llm import client, tracing
from hiremate_ai.llm.call_log import CallRecord
from hiremate_ai.llm.chat import LlmOutputError, model_call_budget


class RecordingLog:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    async def record(self, record: CallRecord) -> None:
        self.records.append(record)


def replace_http_transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    async def send(_transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        return handler(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)


def configure_local(monkeypatch: pytest.MonkeyPatch, url: str = "http://host.docker.internal:11434/v1/") -> None:
    monkeypatch.setattr(
        client,
        "settings",
        Settings(_env_file=None, llm_profile="local", local_llm_base_url=url),
    )


async def test_local_normalization_sends_the_original_schema_to_gemma(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_local(monkeypatch)
    requests: list[httpx.Request] = []
    plan = {"title": None, "blocks": [{"type": "paragraph", "segments": ["1"]}]}

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:26b-a4b-it-qat",
                "message": {"content": json.dumps(plan)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 120,
                "eval_count": 24,
            },
        )

    replace_http_transport(monkeypatch, respond)
    call_log = RecordingLog()
    result = await client.build_document_normalization_llm(call_log).structured(
        DocumentPlan, agent="application.normalize", system="원문 보존", user="번호가 붙은 PDF 조각"
    )

    assert result == DocumentPlan.model_validate(plan)
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST" and str(request.url) == "http://host.docker.internal:11434/api/chat"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "gemma4:26b-a4b-it-qat",
        "messages": [
            {"role": "system", "content": "원문 보존"},
            {"role": "user", "content": "번호가 붙은 PDF 조각"},
        ],
        "stream": False,
        "think": False,
        "format": DocumentPlan.model_json_schema(by_alias=True),
        "options": {"temperature": 0, "num_ctx": 32768, "num_predict": -1, "seed": 20261116},
        "keep_alive": "15m",
    }
    assert len(call_log.records) == 1
    record = call_log.records[0]
    assert (record.agent, record.model, record.status) == ("application.normalize", "gemma4:26b-a4b-it-qat", "OK")
    assert (record.input_tokens, record.output_tokens) == (120, 24)
    assert record.output == result.model_dump(mode="json")


@pytest.mark.parametrize(
    ("reason", "label", "expected"),
    [
        ("length", '{"title":null,"blocks":[]}', "출력 길이 상한"),
        ("stop", '{"invalid":"본문이 들어 있는 잘못된 출력"}', "스키마와 다르다"),
    ],
)
async def test_normalization_rejects_bad_model_responses_without_retry_or_fallback(
    monkeypatch: pytest.MonkeyPatch, reason: str, label: str, expected: str
) -> None:
    configure_local(monkeypatch)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:26b-a4b-it-qat",
                "message": {"content": label},
                "done": True,
                "done_reason": reason,
                "prompt_eval_count": 10,
                "eval_count": 3,
            },
        )

    replace_http_transport(monkeypatch, respond)
    calls = RecordingLog()
    llm = client.build_document_normalization_llm(calls)
    with pytest.raises(LlmOutputError, match=expected):
        await llm.structured(DocumentPlan, agent="application.normalize", system="원문 보존", user="가상 자료")
    assert len(requests) == len(calls.records) == 1
    record = calls.records[0]
    assert record.status == "ERROR" and record.output is None
    assert record.error and expected in record.error and label not in record.error
    assert (record.input_tokens, record.output_tokens) == (10, 3)


async def test_local_normalization_honors_an_exhausted_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_local(monkeypatch)

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"예산 없이 전송: {request.url}")

    replace_http_transport(monkeypatch, forbidden)
    llm = client.build_document_normalization_llm(RecordingLog())
    with model_call_budget(0), pytest.raises(LlmOutputError, match="호출 상한"):
        await llm.structured(DocumentPlan, agent="application.normalize", system="원문 보존", user="가상 자료")


@pytest.mark.parametrize(
    ("content", "error"),
    [
        pytest.param('{"title":["1"],"blocks":[]}', None, id="success"),
        pytest.param('{"invalid":"synthetic model output"}', LlmOutputError, id="invalid-output"),
    ],
)
async def test_local_normalization_records_only_generation_metadata_under_the_job_trace(
    monkeypatch: pytest.MonkeyPatch, langfuse: InMemorySpanExporter, content: str, error: type[Exception] | None
) -> None:
    configure_local(monkeypatch)
    replace_http_transport(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "model": "gemma4:26b-a4b-it-qat",
                "message": {"content": content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 13,
                "eval_count": 5,
            },
        ),
    )
    model = client.build_document_normalization_llm(RecordingLog())
    outcome = pytest.raises(error) if error is not None else nullcontext()
    with tracing.traced("worker/application/normalize"), outcome:
        await model.structured(DocumentPlan, agent="application.normalize", system="원문 보존", user="가상 자료")
    tracing._client().flush()
    spans = langfuse.get_finished_spans()
    root = next(s for s in spans if s.parent is None)
    generations = [s for s in spans if (s.attributes or {}).get("langfuse.observation.type") == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation.parent == root.context
    assert generation.name == "application.normalize"
    attributes = generation.attributes or {}
    assert attributes["langfuse.observation.model.name"] == "gemma4:26b-a4b-it-qat"
    assert json.loads(str(attributes["langfuse.observation.usage_details"])) == {"input": 13, "output": 5, "total": 18}
    payload_keys = {"langfuse.observation.input", "langfuse.observation.output"}
    assert (attributes.keys() & payload_keys) == set()
    if error is not None:
        assert attributes["langfuse.observation.level"] == "ERROR"
        assert attributes["langfuse.observation.status_message"] == error.__name__
    assert not generation.events
