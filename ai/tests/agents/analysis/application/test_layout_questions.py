"""모델이 제목 없이 나눈 자기소개서 문항을 문항 제목 줄로 세우는 규칙.

문항 제목("1. 지원동기")이 본문보다 조금 큰 글자로 번호가 이어지면 문항이다. 모델이 문항 제목을 제목 없는 섹션의 첫
문단에 넣고, 그 문항의 소제목 섹션을 문항 밖 형제로 두면, 문항 제목으로 섹션을 세우고 다음 문항 앞까지의 형제를 안에
넣는다. 모델 요청은 바꾸지 않는다.
"""

from typing import Any
from uuid import uuid4

from hiremate_ai.agents.analysis.application.layout_structure import revise_structure
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, build_segments
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentType, PdfPage

_BODY = "본문 문장이 이어지는 충분히 긴 설명 줄입니다"


def _revise(lines: list[tuple[str, float]], *blocks: dict[str, Any]) -> list[Any]:
    spans = [dict(text=t, x=60, y=100 + 20 * i, endX=60 + 9 * len(t), size=size) for i, (t, size) in enumerate(lines)]
    page = PdfPage.model_validate(
        dict(number=1, text="\n".join(t for t, _ in lines), spans=spans, layout=dict(width=600, height=900))
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


_LINES = [
    ("1. 지원동기", 10.5),
    ("[처음 만난 문제]", 10.0),
    (_BODY, 10.0),
    (_BODY, 10.0),
    ("2. 성장과정", 10.5),
    ("[자란 배경]", 10.0),
    (_BODY, 10.0),
    ("[다시 시작한 일]", 10.0),
    (_BODY, 10.0),
]


def test_untitled_question_sections_take_their_question_line_as_title_and_their_subsections() -> None:
    shaped = _revise(_LINES, _s(None, _p("1-4")), _s(None, _p("5")), _s(["6"], _p("7")), _s(["8"], _p("9")))
    assert shaped == [
        (["1"], [(["2"], [["3-4"]])]),
        (["5"], [(["6"], [["7"]]), (["8"], [["9"]])]),
    ]


def test_a_question_whose_text_another_block_holds_out_of_order_is_left_alone() -> None:
    """다음 문항 앞까지의 조각을 문항 밖의 다른 블록이 갖고 있으면(읽는 순서가 섞임) 손대지 않는다."""
    shaped = _revise(_LINES, _s(None, _p("1-2")), _s(None, _p("5-9")), _p("3-4"))
    assert shaped[0] == (None, [["1-2"]])


def test_an_untitled_section_inside_a_question_is_unwrapped() -> None:
    """쪽 경계에서 모델이 제목 없이 연 섹션이 문항 범위 안이면 그 내용을 문항에 바로 넣는다."""
    shaped = _revise(_LINES, _s(None, _p("1-4")), _s(None, _p("5-6")), _s(None, _p("7")), _s(["8"], _p("9")))
    assert shaped[1] == (["5"], [(["6"], [["7"]]), (["8"], [["9"]])])
