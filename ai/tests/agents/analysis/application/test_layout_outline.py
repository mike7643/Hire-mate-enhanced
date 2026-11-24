"""글자 크기로 제목 단계를 맞추는 규칙.

자유양식 문서에서 제목 단계는 대개 글자 크기로 보인다. 모델이 큰 제목 아래에 둘 작은 제목을 형제로 두거나, 같은 크기
제목을 서로 다른 깊이에 두면 글자 크기 순서로 다시 묶는다. 본문보다 큰 글자로 쓴 제목만 근거로 쓴다.
"""

from typing import Any
from uuid import uuid4

from hiremate_ai.agents.analysis.application.layout_structure import revise_structure
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, Segment, build_segments
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentType, PdfPage

_BODY = "#1b1b1b"
_LONG = "본문 문장이 이어집니다 충분히 긴 설명 문장입니다"


def _span(text: str, y: float, size: float = 10.0, x: float = 40, color: str = _BODY) -> dict[str, Any]:
    return dict(text=text, x=x, y=y, endX=x + len(text) * size * 0.9, size=size, color=color)


def _request(
    spans: list[dict[str, Any]], fills: list[dict[str, Any]] | None = None
) -> tuple[DocumentNormalizationRequest, list[Segment]]:
    layout = dict(width=600, height=900, fillRects=fills or [])
    page = PdfPage.model_validate(dict(number=1, text="\n".join(s["text"] for s in spans), spans=spans, layout=layout))
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=DocumentType.RESUME,
            pages=[page],
        )
    )
    return request, build_segments([page])


def _plan(*blocks: dict[str, Any]) -> DocumentPlan:
    return DocumentPlan.model_validate({"title": None, "blocks": list(blocks)})


def _p(*refs: str) -> dict[str, Any]:
    return {"type": "paragraph", "segments": list(refs)}


def _s(title: list[str] | None, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "section", "title": title, "blocks": list(blocks)}


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


def _revise(
    spans: list[dict[str, Any]], *blocks: dict[str, Any], fills: list[dict[str, Any]] | None = None
) -> list[Any]:
    request, segments = _request(spans, fills)
    return _shape(revise_structure(_plan(*blocks), segments, request).blocks)


def _resume() -> list[dict[str, Any]]:
    """1 경력(18), 2 본문, 3 회사(14), 4 본문, 5 자격(18), 6 정보처리기사(14), 7 본문, 8 SQLD(14), 9 본문."""
    rows = [("경력", 18), (_LONG, 10), ("한빛정보", 14), (_LONG, 10), ("자격", 18), ("정보처리기사", 14), (_LONG, 10)]
    rows += [("SQLD", 14), (_LONG, 10)]
    return [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]


def test_smaller_headings_left_beside_a_bigger_heading_go_under_it() -> None:
    shaped = _revise(
        _resume(),
        _s(["1"], _p("2"), _s(["3"], _p("4"))),
        _s(["5"]),
        _s(["6"], _p("7")),
        _s(["8"], _p("9")),
    )
    assert shaped == [(["1"], [["2"], (["3"], [["4"]])]), (["5"], [(["6"], [["7"]]), (["8"], [["9"]])])]


def test_a_bigger_heading_nested_under_a_smaller_one_is_lifted() -> None:
    shaped = _revise(
        _resume(),
        _s(["1"], _p("2"), _s(["3"], _p("4"), _s(["5"], _s(["6"], _p("7")), _s(["8"], _p("9"))))),
    )
    assert shaped == [(["1"], [["2"], (["3"], [["4"]])]), (["5"], [(["6"], [["7"]]), (["8"], [["9"]])])]


def test_nesting_that_already_follows_heading_sizes_is_left_alone() -> None:
    blocks = (_s(["1"], _p("2"), _s(["3"], _p("4"))), _s(["5"], _s(["6"], _p("7")), _s(["8"], _p("9"))))
    assert _revise(_resume(), *blocks) == [
        (["1"], [["2"], (["3"], [["4"]])]),
        (["5"], [(["6"], [["7"]]), (["8"], [["9"]])]),
    ]


def test_headings_in_body_size_keep_the_model_nesting() -> None:
    spans = [_span(text, 40 + 30 * i) for i, text in enumerate(["경력", _LONG, "회사", _LONG, "자격", _LONG])]
    shaped = _revise(spans, _s(["1"], _p("2")), _s(["3"], _p("4")), _s(["5"], _p("6")))
    assert shaped == [(["1"], [["2"]]), (["3"], [["4"]]), (["5"], [["6"]])]


def test_a_body_size_titled_section_stays_whole_under_the_current_heading() -> None:
    """1 인적사항(18), 2 본문, 3 Education(본문 크기), 4 본문, 5 자격(18), 6 정보처리기사(14), 7 본문."""
    rows = [("인적사항", 18), (_LONG, 10), ("Education", 10), (_LONG, 10), ("자격", 18), ("정보처리기사", 14)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate([*rows, (_LONG, 10)])]
    shaped = _revise(spans, _s(["1"], _p("2")), _s(["3"], _p("4")), _s(["5"]), _s(["6"], _p("7")))
    assert shaped == [(["1"], [["2"], (["3"], [["4"]])]), (["5"], [(["6"], [["7"]])])]


def test_a_paragraph_led_by_a_heading_size_line_becomes_a_section() -> None:
    """1 프로젝트(18), 2 배경(14), 3 본문, 4 원인(14), 5 본문, 6 결과(14), 7 본문. 원인과 결과는 문단 첫 줄에 붙었다."""
    rows = [("프로젝트", 18), ("배경", 14), (_LONG, 10), ("원인", 14), (_LONG, 10), ("결과", 14), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    shaped = _revise(spans, _s(["1"], _s(["2"], _p("3")), _p("4-5"), _p("6-7")))
    assert shaped == [(["1"], [(["2"], [["3"]]), (["4"], [["5"]]), (["6"], [["7"]])])]


def test_a_large_lead_line_is_not_a_heading_unless_that_size_titles_a_section() -> None:
    rows = [("프로젝트", 18), ("강조 문장", 14), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    assert _revise(spans, _s(["1"], _p("2-3"))) == [(["1"], [["2-3"]])]


def test_a_title_holding_a_link_or_address_goes_back_into_the_text() -> None:
    rows = [("연락처", 18), ("GITHUB : https://github.com/someone", 10), (_LONG, 10), ("자격", 18), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    shaped = _revise(spans, _s(["1"], _s(["2"], _p("3"))), _s(["4"], _p("5")))
    assert shaped == [(["1"], [["2"], ["3"]]), (["4"], [["5"]])]


def test_a_heading_earlier_in_reading_order_is_not_put_under_a_later_heading() -> None:
    """1 연락처(14), 2 본문, 3 경력(18), 4 회사(14), 5 본문, 6 학력(18), 7 본문. 모델이 앞 단의 연락처를 끝에 두었다."""
    rows = [("연락처", 14), (_LONG, 10), ("경력", 18), ("회사", 14), (_LONG, 10), ("학력", 18), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    shaped = _revise(spans, _s(["3"], _s(["4"], _p("5"))), _s(["6"], _p("7")), _s(["1"], _p("2")))
    assert shaped == [(["3"], [(["4"], [["5"]])]), (["6"], [["7"]]), (["1"], [["2"]])]


def test_a_sidebar_heading_is_not_put_under_a_main_column_heading() -> None:
    """오른쪽 사이드바. 1 경력(18), 2 회사(14), 3 본문, 4 학력(18), 5 본문은 본문 단.

    6 기술(14), 7-9 본문은 사이드바 안이다.
    """
    main = [("경력", 18), ("회사", 14), (_LONG, 10), ("학력", 18), (_LONG, 10)]
    side = [("기술", 14), ("파이썬", 10), ("자바", 10), ("SQL", 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(main)]
    spans += [_span(text, 40 + 30 * i, size, x=470) for i, (text, size) in enumerate(side)]
    fills = [dict(x1=460, y1=0, x2=600, y2=900, color="#eeeeee")]
    shaped = _revise(spans, _s(["1"], _s(["2"], _p("3"))), _s(["4"], _p("5")), _s(["6"], _p("7-9")), fills=fills)
    assert shaped == [(["1"], [(["2"], [["3"]])]), (["4"], [["5"]]), (["6"], [["7-9"]])]


def test_a_large_label_beside_a_form_row_is_not_made_a_heading() -> None:
    """1 프로필(18), 2 사진(14) 옆에 3 같은 행의 칸 글, 4 본문, 5 학력(14), 6 본문. 사진은 표 이름표다."""
    spans = [_span("프로필", 40, 18), _span("사진", 80, 14), _span("(한글) 홍길동 전화번호", 81.3, 10, x=200)]
    spans += [_span(_LONG, 120), _span("학력", 160, 14), _span(_LONG, 190)]
    shaped = _revise(spans, _s(["1"], _p("2-4"), _s(["5"], _p("6"))))
    assert shaped == [(["1"], [["2-4"], (["5"], [["6"]])])]


def test_a_shaded_box_after_a_heading_block_is_not_put_under_that_heading() -> None:
    """1 경력기술서(18), 2 회사A(14), 3 본문, 4 회사B(14), 5 본문, 6-7 회색 상자 안 프로젝트, 8 학력(18), 9 본문."""
    rows = [("경력기술서", 18), ("회사A", 14), (_LONG, 10), ("회사B", 14), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    spans += [_span("개인 프로젝트", 190, 10, x=70), _span(_LONG, 220, 10, x=70)]
    spans += [_span("학력", 250, 18), _span(_LONG, 280)]
    fills = [dict(x1=60, y1=176, x2=560, y2=232, color="#f2f2f2")]
    shaped = _revise(
        spans, _s(["1"], _s(["2"], _p("3"))), _s(["4"], _p("5")), _s(["6"], _p("7")), _s(["8"], _p("9")), fills=fills
    )
    assert shaped == [(["1"], [(["2"], [["3"]]), (["4"], [["5"]])]), (["6"], [["7"]]), (["8"], [["9"]])]


def test_a_heading_band_does_not_split_the_heading_from_its_text() -> None:
    """제목만 덮는 한 줄 높이의 띠는 상자가 아니다. 1 경력(18, 띠), 2 회사(14), 3 본문, 4 학력(18, 띠), 5 회사(14)."""
    rows = [("경력", 18), ("회사", 14), (_LONG, 10), ("학력", 18), ("학교", 14), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    fills = [dict(x1=30, y1=24, x2=570, y2=46, color="#dddddd"), dict(x1=30, y1=114, x2=570, y2=136, color="#dddddd")]
    shaped = _revise(spans, _s(["1"]), _s(["2"], _p("3")), _s(["4"]), _s(["5"], _p("6")), fills=fills)
    assert shaped == [(["1"], [(["2"], [["3"]])]), (["4"], [(["5"], [["6"]])])]


def test_a_fill_in_the_page_background_color_is_not_a_box() -> None:
    """회색 상자 대신 바탕과 같은 흰 채움이면 보이지 않으므로, 6-7은 앞 제목(회사B) 아래 내용이다."""
    rows = [("경력기술서", 18), ("회사A", 14), (_LONG, 10), ("회사B", 14), (_LONG, 10)]
    spans = [_span(text, 40 + 30 * i, size) for i, (text, size) in enumerate(rows)]
    spans += [_span("개인 프로젝트", 190, 10, x=70), _span(_LONG, 220, 10, x=70)]
    spans += [_span("학력", 250, 18), _span(_LONG, 280)]
    fills = [dict(x1=0, y1=0, x2=600, y2=900, color="#FFFFFF"), dict(x1=60, y1=176, x2=560, y2=232, color="#ffffff")]
    shaped = _revise(
        spans, _s(["1"], _s(["2"], _p("3"))), _s(["4"], _p("5")), _s(["6"], _p("7")), _s(["8"], _p("9")), fills=fills
    )
    assert shaped == [(["1"], [(["2"], [["3"]]), (["4"], [["5"], (["6"], [["7"]])])]), (["8"], [["9"]])]
