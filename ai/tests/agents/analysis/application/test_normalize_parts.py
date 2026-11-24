"""긴 서류를 쪽 묶음으로 나눠 묻고, 통째 호출이 실패하면 쪽마다 다시 묻는 규칙.

로컬 모델은 요청이 길면(대략 2만 토큰 넘게) 앞 몇 쪽만 답하고 멈춘다. 그래서 요청 글자 수가 한도를 넘으면 처음부터
연속한 쪽 묶음으로 나눠 묻는다. 한도 안의 서류는 지금과 똑같은 요청 한 번으로 묻는다.
"""

import json
from typing import Any
from uuid import uuid4

import pytest

from hiremate_ai.agents.analysis.application.normalize import (
    DocumentPlan,
    NormalizationLimitError,
    normalize_with_report,
)
from hiremate_ai.config import settings
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentSection, DocumentType
from hiremate_ai.llm.chat import LlmFailureReason, LlmOutputError
from hiremate_ai.llm.fake import FakeLlm, LlmCall

_LINE = "본문 문장이 이어지는 긴 줄입니다 "


def _page(number: int, *lines: str) -> dict[str, Any]:
    spans = [{"text": line, "x": 20, "y": 20 + 20 * i, "endX": 20 + 9 * len(line)} for i, line in enumerate(lines)]
    return {"number": number, "text": "\n".join(lines), "issues": [], "spans": spans, "imageCount": 0, "formCount": 0}


def _request(*pages: dict[str, Any]) -> DocumentNormalizationRequest:
    return DocumentNormalizationRequest.model_validate(
        {
            "companyId": str(uuid4()),
            "recruitmentId": str(uuid4()),
            "applicationId": str(uuid4()),
            "documentId": str(uuid4()),
            "revision": 1,
            "documentType": DocumentType.COVER_LETTER,
            "pages": list(pages),
        }
    )


def _plan(*blocks: dict[str, Any], title: list[str] | None = None) -> DocumentPlan:
    return DocumentPlan.model_validate({"title": title, "blocks": list(blocks)})


def _payload(call: LlmCall) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(call.user.split("\n", 1)[1].rsplit("\n", 1)[0])
    return data


def _four_pages() -> list[dict[str, Any]]:
    """쪽마다 두 줄. 조각 1-2는 1쪽, 3-4는 2쪽, 5-6은 3쪽, 7-8은 4쪽이다."""
    return [_page(n, f"{n}쪽 첫 줄 " + _LINE * 3, f"{n}쪽 둘째 줄 " + _LINE * 3) for n in range(1, 5)]


async def _two_pages_limit(pages: list[dict[str, Any]]) -> int:
    """요청의 쪽 하나 글자 수를 재서, 두 쪽까지만 들어가는 한도를 만든다."""
    probe = FakeLlm(_plan({"type": "paragraph", "segments": ["1-8"]}))
    await normalize_with_report(probe, _request(*pages))
    one = len(json.dumps(_payload(probe.calls[0])["pages"][0], ensure_ascii=False))
    return 2 * one + one // 2


@pytest.fixture(autouse=True)
def _local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_profile", "local")


async def test_a_document_within_the_limit_is_asked_in_one_call() -> None:
    llm = FakeLlm(_plan({"type": "paragraph", "segments": ["1-8"]}))
    document, report = await normalize_with_report(llm, _request(*_four_pages()))
    assert [call.agent for call in llm.calls] == ["application.normalize"]
    assert _payload(llm.calls[0])["page_count"] == 4
    assert not report.missing


async def test_a_long_document_is_asked_in_consecutive_page_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = _four_pages()
    monkeypatch.setattr(settings, "normalization_part_chars", await _two_pages_limit(pages))
    llm = FakeLlm(
        _plan({"type": "paragraph", "segments": ["1-2"]}, {"type": "paragraph", "segments": ["3-4"]}),
        _plan({"type": "paragraph", "segments": ["5-6"]}, {"type": "paragraph", "segments": ["7-8"]}),
    )
    document, report = await normalize_with_report(llm, _request(*pages))
    assert [call.agent for call in llm.calls] == [
        "application.normalize.part_1_2",
        "application.normalize.part_3_4",
    ]
    first, second = (_payload(call) for call in llm.calls)
    assert [p["page"] for p in first["pages"]] == [1, 2] and first["page_count"] == 2
    assert [p["page"] for p in second["pages"]] == [3, 4] and second["segment_count"] == 4
    assert [s["id"] for p in second["pages"] for s in p["segments"]] == [5, 6, 7, 8]
    assert [block.pages for block in document.blocks] == [[1], [2], [3], [4]]
    assert not report.missing


async def test_text_at_the_start_of_a_part_continues_the_last_section(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = _four_pages()
    monkeypatch.setattr(settings, "normalization_part_chars", await _two_pages_limit(pages))
    llm = FakeLlm(
        _plan({"type": "section", "title": ["1"], "blocks": [{"type": "paragraph", "segments": ["2-4"]}]}),
        _plan(
            {"type": "paragraph", "segments": ["5"]},
            {"type": "section", "title": ["6"], "blocks": [{"type": "paragraph", "segments": ["7-8"]}]},
        ),
    )
    document, _ = await normalize_with_report(llm, _request(*pages))
    first, second = document.blocks
    assert isinstance(first, DocumentSection) and isinstance(second, DocumentSection)
    assert [block.pages for block in first.blocks] == [[1, 2], [3]]
    assert second.title is not None and second.title.startswith("3쪽 둘째 줄")


async def test_a_failed_whole_call_is_asked_again_page_by_page() -> None:
    def broken(call: LlmCall) -> DocumentPlan:
        raise LlmOutputError("구조화 응답이 잘렸다", reason=LlmFailureReason.FORMAT)

    pages = _four_pages()[:2]
    llm = FakeLlm(
        broken,
        _plan({"type": "paragraph", "segments": ["1-2"]}),
        _plan({"type": "paragraph", "segments": ["3-4"]}),
    )
    document, report = await normalize_with_report(llm, _request(*pages))
    assert [call.agent for call in llm.calls] == [
        "application.normalize",
        "application.normalize.retry.page_1",
        "application.normalize.retry.page_2",
    ]
    assert [block.pages for block in document.blocks] == [[1], [2]] and not report.missing


async def test_a_failed_call_on_a_one_page_document_is_raised() -> None:
    def broken(call: LlmCall) -> DocumentPlan:
        raise LlmOutputError("구조화 응답이 잘렸다", reason=LlmFailureReason.FORMAT)

    with pytest.raises(LlmOutputError):
        await normalize_with_report(FakeLlm(broken), _request(_page(1, "한 쪽짜리 " + _LINE)))


async def test_a_failed_part_is_asked_again_page_by_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = _four_pages()
    monkeypatch.setattr(settings, "normalization_part_chars", await _two_pages_limit(pages))

    def broken(call: LlmCall) -> DocumentPlan:
        raise TimeoutError

    llm = FakeLlm(
        _plan({"type": "paragraph", "segments": ["1-2"]}, {"type": "paragraph", "segments": ["3-4"]}),
        broken,
        _plan({"type": "paragraph", "segments": ["5-6"]}),
        _plan({"type": "paragraph", "segments": ["7-8"]}),
    )
    document, report = await normalize_with_report(llm, _request(*pages))
    assert [call.agent for call in llm.calls][2:] == [
        "application.normalize.retry.page_3",
        "application.normalize.retry.page_4",
    ]
    assert [block.pages for block in document.blocks] == [[1], [2], [3], [4]] and not report.missing


async def test_missing_pages_are_asked_again_beyond_the_general_agent_call_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_agent_max_calls", 3)
    pages = [_page(n, f"{n}쪽 " + _LINE) for n in range(1, 7)]
    retries = [_plan({"type": "paragraph", "segments": [str(n)]}) for n in range(2, 7)]
    llm = FakeLlm(_plan({"type": "paragraph", "segments": ["1"]}), *retries)
    _, report = await normalize_with_report(llm, _request(*pages))
    assert len(llm.calls) == 6 and not report.missing


async def test_a_document_over_the_page_limit_is_rejected_before_asking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "normalization_max_pages", 3)
    llm = FakeLlm()
    with pytest.raises(NormalizationLimitError, match="4쪽"):
        await normalize_with_report(llm, _request(*_four_pages()))
    assert not llm.calls
