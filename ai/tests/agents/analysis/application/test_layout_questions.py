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


def test_a_long_sentence_used_as_a_section_title_goes_back_to_the_text() -> None:
    lines = [
        ("제가 기획한 두 프로젝트를 소개드립니다. 정책서와 화면 설계서는 첨부한 포트폴리오에 담았습니다.", 10.0),
        (_BODY, 10.0),
        ("[경력]", 10.0),
        ("※ 건축 설계 분야의 정규직 경력은 없고, 이전 직무는 게임 그래픽과 레벨 디자인이었습니다.", 10.0),
        (_BODY, 10.0),
    ]
    shaped = _revise(lines, _s(["1"], _p("2")), _s(["3", "4"], _p("5")))
    assert shaped == [["1"], ["2"], (["3"], [["4"], ["5"]])]


def test_a_body_line_cut_mid_sentence_used_as_a_title_goes_back_to_the_text() -> None:
    line = (
        "블로그 주소는 이력서에 적어 두었습니다. "
        "업무와 직접 관련된 내용은 아니지만 꾸준히 운영해 온 기록이라 보여 드릴 수 있"
    )
    shaped = _revise([(_BODY, 10.0), (line, 10.0), ("는 자료입니다.", 10.0)], _p("1"), _s(["2"], _p("3")))
    assert shaped == [["1"], ["2"], ["3"]]


def test_a_title_without_letters_or_digits_goes_back_to_the_text() -> None:
    shaped = _revise([(_BODY, 10.0), ("/**", 10.0), (" * 매일 실행하는 작업", 10.0)], _p("1"), _s(["2"], _p("3")))
    assert shaped == [["1"], ["2"], ["3"]]


def test_questions_the_model_nested_inside_the_previous_question_come_back_out_as_siblings() -> None:
    lines = [
        ("1. 지원동기", 10.5),
        (_BODY, 10.0),
        ("2. 성장과정", 10.5),
        (_BODY, 10.0),
        ("3. 나의 장점과 약점", 10.5),
        (_BODY, 10.0),
        ("위의 내용은 사실과 틀림없음을 확인합니다.", 9.5),
    ]
    nested = _s(None, _p("1"), _p("2"), _s(None, _p("3"), _p("4"), _s(None, _p("5"), _p("6"))), _p("7"))
    assert _revise(lines, nested) == [(["1"], [["2"]]), (["3"], [["4"]]), (["5"], [["6"]]), ["7"]]


def test_a_short_lead_line_larger_than_the_rest_of_its_section_becomes_the_title() -> None:
    lines = [("학력사항 (최종학력 : 대학교 졸업)", 9.5), (_BODY, 9.0), (_BODY, 9.0)]
    assert _revise(lines, _s(None, _p("1"), _p("2-3"))) == [(["1"], [["2-3"]])]
    same = [("학력사항 (최종학력 : 대학교 졸업)", 9.0), (_BODY, 9.0), (_BODY, 9.0)]
    assert _revise(same, _s(None, _p("1"), _p("2-3"))) == [(None, [["1"], ["2-3"]])]
