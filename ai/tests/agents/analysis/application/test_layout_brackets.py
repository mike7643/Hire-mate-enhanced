"""괄호로 감싼 소제목 줄(〈소제목〉, [소제목], <소제목>)을 섹션 제목으로 세우는 규칙.

자기소개서는 질문 아래 본문을 괄호 소제목으로 나누는 일이 많다. 소제목이 본문과 같은 글자 모양이면 글자 모양으로는
찾지 못하고, 모델은 이 줄을 문단으로 두기도 한다. 같은 괄호로 감싼 한 줄이 두 번 넘게 본문 앞에 나오면 소제목으로
보고, 그 줄로 시작하는 문단을 섹션으로 세운다. 모델 요청은 바꾸지 않는다.
"""

from typing import Any
from uuid import uuid4

from hiremate_ai.agents.analysis.application.layout_structure import revise_structure
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, build_segments
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentType, PdfPage

_BODY = "본문 문장이 이어지는 충분히 긴 설명 줄입니다"


def _revise(lines: list[str], *blocks: dict[str, Any]) -> list[Any]:
    spans = [dict(text=t, x=60, y=100 + 20 * i, endX=60 + 9 * len(t), size=10.0) for i, t in enumerate(lines)]
    page = PdfPage.model_validate(
        dict(number=1, text="\n".join(lines), spans=spans, layout=dict(width=600, height=900))
    )
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=DocumentType.COVER_LETTER,
            pages=[page],
        )
    )
    plan = DocumentPlan.model_validate({"title": None, "blocks": list(blocks)})
    return _shape(revise_structure(plan, build_segments([page]), request).blocks)


def _shape(blocks: list[Any]) -> list[Any]:
    shaped: list[Any] = []
    for block in blocks:
        if block.type == "section":
            shaped.append((block.title, _shape(block.blocks)))
        elif block.type == "paragraph":
            shaped.append(block.segments)
        else:
            shaped.append(block.type)
    return shaped


def _p(*refs: str) -> dict[str, Any]:
    return {"type": "paragraph", "segments": list(refs)}


def _s(title: list[str] | None, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "section", "title": title, "blocks": list(blocks)}


def test_bracketed_subtitle_paragraphs_become_subsections_of_the_question() -> None:
    lines = ["1. 지원 동기", "[처음 만난 문제]", _BODY, _BODY, "[해결한 방법]", _BODY]
    shaped = _revise(lines, _s(["1"], _p("2"), _p("3-4"), _p("5"), _p("6")))
    assert shaped == [(["1"], [(["2"], [["3-4"]]), (["5"], [["6"]])])]


def test_a_subtitle_joined_to_its_body_is_split_off_as_the_title() -> None:
    lines = ["1. 지원 동기", "〈처음 만난 문제〉", _BODY, _BODY, "〈해결한 방법〉", _BODY]
    shaped = _revise(lines, _s(["1"], _p("2-4"), _p("5-6")))
    assert shaped == [(["1"], [(["2"], [["3-4"]]), (["5"], [["6"]])])]


def test_text_before_the_first_subtitle_stays_with_the_question() -> None:
    lines = ["1. 지원 동기", _BODY, "<처음 만난 문제>", _BODY, "<해결한 방법>", _BODY]
    shaped = _revise(lines, _s(["1"], _p("2"), _p("3-4"), _p("5-6")))
    assert shaped == [(["1"], [["2"], (["3"], [["4"]]), (["5"], [["6"]])])]


def test_a_bracketed_line_used_once_stays_a_paragraph() -> None:
    lines = ["1. 지원 동기", "〈처음 만난 문제〉", _BODY, _BODY]
    assert _revise(lines, _s(["1"], _p("2"), _p("3-4"))) == [(["1"], [["2"], ["3-4"]])]


def test_a_subtitle_with_nothing_after_it_in_its_section_stays_a_paragraph() -> None:
    lines = ["1. 지원 동기", "〈처음 만난 문제〉", _BODY, "〈해결한 방법〉", "2. 입사 후 포부", _BODY]
    shaped = _revise(lines, _s(["1"], _p("2"), _p("3"), _p("4")), _s(["5"], _p("6")))
    assert shaped == [(["1"], [(["2"], [["3"]]), ["4"]]), (["5"], [["6"]])]


def test_a_bracketed_last_line_is_not_a_subtitle_candidate() -> None:
    lines = ["1. 지원 동기", "〈처음 만난 문제〉", _BODY, "〈끝〉"]
    assert _revise(lines, _s(["1"], _p("2"), _p("3"), _p("4"))) == [(["1"], [["2"], ["3"], ["4"]])]


def test_subtitles_the_model_already_made_sections_are_left_alone() -> None:
    lines = ["1. 지원 동기", "[처음 만난 문제]", _BODY, "[해결한 방법]", _BODY]
    blocks = (_s(["1"], _s(["2"], _p("3")), _s(["4"], _p("5"))),)
    assert _revise(lines, *blocks) == [(["1"], [(["2"], [["3"]]), (["4"], [["5"]])])]


def test_an_empty_subtitle_section_the_model_made_stays_a_section() -> None:
    """모델이 본문 조각을 빠뜨려 빈 소제목 섹션이 남았다. 빠진 조각은 조립 때 제목 뒤에 되살아나므로 섹션으로 둔다."""
    lines = ["1. 지원 동기", "[처음 만난 문제]", _BODY, "[해결한 방법]", _BODY]
    shaped = _revise(lines, _s(["1"], _s(["2"]), _s(["4"], _p("5"))))
    assert shaped == [(["1"], [(["2"], []), (["4"], [["5"]])])]


def test_a_subtitle_the_model_repeated_at_the_start_of_its_own_text_stays_one_section() -> None:
    """실제 Gemma 응답 모양: 소제목 섹션의 제목 조각을 첫 문단에도 넣었다.

    빈 섹션과 같은 제목 섹션으로 쪼개지 않는다.
    """
    lines = ["1. 지원 동기", "[처음 만난 문제]", _BODY, "[해결한 방법]", _BODY]
    shaped = _revise(lines, _s(["1"]), _s(["2"], _p("2-3")), _s(["4"], _p("4-5")))
    assert shaped == [(["1"], []), (["2"], [["3"]]), (["4"], [["5"]])]


def _revise_pages(pages: list[list[str]], *blocks: dict[str, Any]) -> list[Any]:
    """여러 쪽 서류. 줄마다 조각 하나이고, 조각 번호는 쪽을 넘어 이어진다."""
    built = []
    for number, lines in enumerate(pages, start=1):
        spans = [dict(text=t, x=60, y=100 + 20 * i, endX=60 + 9 * len(t), size=10.0) for i, t in enumerate(lines)]
        built.append(
            PdfPage.model_validate(
                dict(number=number, text="\n".join(lines), spans=spans, layout=dict(width=600, height=900))
            )
        )
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=DocumentType.COVER_LETTER,
            pages=built,
        )
    )
    plan = DocumentPlan.model_validate({"title": None, "blocks": list(blocks)})
    return _shape(revise_structure(plan, build_segments(built), request).blocks)


def test_a_subtitle_at_the_foot_of_a_page_heads_the_text_on_the_next_page() -> None:
    pages = [["1. 지원 동기", "〈처음 만난 문제〉", _BODY, "〈해결한 방법〉"], [_BODY]]
    shaped = _revise_pages(pages, _s(["1"], _p("2"), _p("3"), _p("4"), _p("5")))
    assert shaped == [(["1"], [(["2"], [["3"]]), (["4"], [["5"]])])]


def test_text_the_model_split_off_at_a_page_break_rejoins_the_section_it_continues() -> None:
    """앞 쪽 마지막 줄이 단 끝까지 차 있으면 문단이 다음 쪽으로 이어진다. 모델이 그 자리에서 연 제목 없는 섹션의
    앞부분은 앞 섹션 글이다."""
    pages = [["1. 지원 동기", "〈처음 만난 문제〉", _BODY, _BODY], [_BODY, "〈해결한 방법〉", _BODY]]
    shaped = _revise_pages(pages, _s(["1"], _p("2"), _p("3-4")), _s(None, _p("5"), _p("6"), _p("7")))
    assert shaped == [(["1"], [(["2"], [["3-4"], ["5"]]), (["6"], [["7"]])])]


def test_a_section_the_model_opened_after_a_finished_paragraph_stays_apart() -> None:
    """앞 쪽 마지막 줄이 짧게 끝나면(문단이 끝남) 다음 쪽의 제목 없는 섹션을 옮기지 않는다."""
    pages = [["1. 지원 동기", _BODY, "짧게 끝난 줄."], [_BODY, _BODY]]
    shaped = _revise_pages(pages, _s(["1"], _p("2-3")), _s(None, _p("4"), _p("5")))
    assert shaped == [(["1"], [["2-3"]]), (None, [["4"], ["5"]])]
