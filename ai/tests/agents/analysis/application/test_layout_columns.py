"""자유 양식 PDF에서 단과 목록 머리표를 그림 증거로 읽는 규칙."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest

from hiremate_ai.agents.analysis.application.layout_structure import revise_structure
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, build_segments
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentType, PdfPage


def _span(text: str, x: float, y: float, end_x: float, size: float = 9.3) -> dict[str, Any]:
    return dict(text=text, x=x, y=y, endX=end_x, size=size)


def _box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)


def _page(
    spans: list[dict[str, Any]], *, fills: Sequence[dict[str, Any]] = (), marks: Sequence[dict[str, float]] = ()
) -> PdfPage:
    layout = dict(width=600, height=800, fillRects=list(fills), markBoxes=list(marks))
    return PdfPage.model_validate(dict(number=1, text="\n".join(s["text"] for s in spans), spans=spans, layout=layout))


def _request(kind: DocumentType, page: PdfPage) -> DocumentNormalizationRequest:
    return DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=kind,
            pages=[page],
        )
    )


# 왼쪽 사이드바와 본문 줄이 세로 위치 순서로 번갈아 나오는 쪽.
_INTERLEAVED = [
    _span("Summary line one", 220, 90, 500),
    _span("CONTACT", 20, 100, 70),
    _span("Summary line two", 220, 110, 480),
    _span("010-0000-0000", 20, 120, 80),
    _span("Experience", 220, 130, 300),
    _span("someone@example.com", 20, 140, 130),
]


def _texts(page: PdfPage) -> list[str]:
    return [segment.text for segment in build_segments([page])]


def test_a_full_height_edge_panel_is_read_as_its_own_column_before_the_main_column() -> None:
    page = _page(_INTERLEAVED, fills=[dict(_box(0, 0, 190, 800), color="#222222")])
    assert _texts(page) == [
        "CONTACT",
        "010-0000-0000",
        "someone@example.com",
        "Summary line one",
        "Summary line two",
        "Experience",
    ]
    assert [segment.id for segment in build_segments([page])] == [1, 2, 3, 4, 5, 6]


def test_a_right_edge_panel_is_read_after_the_main_column() -> None:
    spans = [
        _span("Main one", 20, 90, 300),
        _span("SKILLS", 430, 100, 480),
        _span("Main two", 20, 110, 300),
        _span("Python", 430, 120, 470),
        _span("Main three", 20, 130, 300),
        _span("SQL", 430, 140, 460),
    ]
    page = _page(spans, fills=[dict(_box(410, 0, 600, 800), color="#eeeeee")])
    assert _texts(page) == ["Main one", "Main two", "Main three", "SKILLS", "Python", "SQL"]


@pytest.mark.parametrize(
    "fill",
    [
        _box(0, 0, 600, 150),  # 머리 띠: 쪽 높이를 덮지 않는다
        _box(200, 0, 390, 800),  # 쪽 가장자리에 붙지 않았다
        _box(0, 0, 400, 800),  # 쪽 너비의 절반 넘게 덮는다
        _box(0, 0, 190, 115),  # 높이가 모자란다
    ],
)
def test_reading_order_is_unchanged_without_a_column_panel(fill: dict[str, float]) -> None:
    page = _page(_INTERLEAVED, fills=[dict(fill, color="#222222")])
    assert _texts(page) == [span["text"] for span in _INTERLEAVED]


def test_reading_order_is_unchanged_when_the_panel_holds_too_few_lines() -> None:
    spans = [s for s in _INTERLEAVED if s["text"] not in ("010-0000-0000", "someone@example.com")]
    page = _page(spans, fills=[dict(_box(0, 0, 190, 800), color="#222222")])
    assert _texts(page) == [span["text"] for span in spans]


def _bullet(y: float, x: float = 223.5) -> dict[str, float]:
    # 글자 기준선(y)보다 조금 위, 줄 왼쪽에 그린 지름 3.8pt 원
    return _box(x, y - 5.2, x + 3.8, y - 1.4)


def _bulleted_page(*, gap_left: float = 0, extra: Sequence[dict[str, Any]] = ()) -> PdfPage:
    spans = [
        _span("Company · Engineer", 221, 100, 330, 10),
        _span("First achievement starts here and runs to the", 235, 122, 550),
        _span("end of the line", 235, 136.2, 320),
        _span("Second achievement on one line", 235, 150.5, 500),
        _span("Third achievement begins", 235, 165.5, 540),
        _span("and wraps", 235, 179.8, 300),
        *extra,
    ]
    marks = [_bullet(y, 223.5 - gap_left) for y in (122, 150.5, 165.5)]
    return _page(spans, marks=marks)


def _plan(*blocks: dict[str, Any]) -> DocumentPlan:
    return DocumentPlan.model_validate(dict(title=None, blocks=list(blocks)))


def _paragraph(*ids: str) -> dict[str, Any]:
    return dict(type="paragraph", segments=list(ids))


def _list(*items: list[str]) -> dict[str, Any]:
    return dict(type="list", items=[dict(segments=item, items=[]) for item in items])


@pytest.mark.parametrize("kind", [DocumentType.RESUME, DocumentType.PORTFOLIO, DocumentType.COVER_LETTER])
def test_drawn_bullets_split_a_merged_paragraph_into_list_items(kind: DocumentType) -> None:
    request = _request(kind, _bulleted_page())
    segments = build_segments(request.pages)
    original = _plan(_paragraph("1"), _paragraph("2-6"))

    corrected = revise_structure(original, segments, request)

    assert corrected == _plan(_paragraph("1"), _list(["2", "3"], ["4"], ["5", "6"]))
    assert revise_structure(corrected, segments, request) is corrected


def test_drawn_bullets_keep_the_lead_in_text_as_a_paragraph() -> None:
    request = _request(DocumentType.RESUME, _bulleted_page())
    segments = build_segments(request.pages)
    corrected = revise_structure(_plan(_paragraph("1-6")), segments, request)
    assert corrected == _plan(_paragraph("1"), _list(["2", "3"], ["4"], ["5", "6"]))


def test_drawn_bullets_fix_list_items_and_join_one_item_paragraphs() -> None:
    request = _request(DocumentType.RESUME, _bulleted_page())
    segments = build_segments(request.pages)
    merged_items = revise_structure(_plan(_paragraph("1"), _list(["2-4"], ["5-6"])), segments, request)
    split_paragraphs = revise_structure(
        _plan(_paragraph("1"), _paragraph("2-3"), _paragraph("4"), _paragraph("5-6")), segments, request
    )
    expected = _plan(_paragraph("1"), _list(["2", "3"], ["4"], ["5", "6"]))
    assert merged_items == expected
    assert split_paragraphs == expected


def test_a_less_indented_line_ends_the_bullet_item() -> None:
    # 머리표 없는 다음 줄이 항목 글 위치에 있어도, 그 사이에 덜 들여 쓴 줄이 있으면 항목이 끝난다.
    extra = [_span("Next company · Lead", 221, 187, 330, 10), _span("Lead text", 235, 194, 300)]
    request = _request(DocumentType.RESUME, _bulleted_page(extra=extra))
    segments = build_segments(request.pages)
    corrected = revise_structure(_plan(_paragraph("1"), _paragraph("2-6"), _paragraph("7-8")), segments, request)
    assert corrected == _plan(_paragraph("1"), _list(["2", "3"], ["4"], ["5", "6"]), _paragraph("7-8"))


@pytest.mark.parametrize("fault", ["no_marks", "far_mark", "interleaved", "keep_lines"])
def test_bullets_without_clear_evidence_leave_the_plan_alone(fault: str) -> None:
    if fault == "no_marks":
        page = _bulleted_page()
        layout = page.layout
        assert layout is not None
        page = page.model_copy(update={"layout": layout.model_copy(update={"mark_boxes": []})})
    else:
        page = _bulleted_page(gap_left=30 if fault == "far_mark" else 0)
    request = _request(DocumentType.RESUME, page)
    segments = build_segments(request.pages)
    if fault == "interleaved":
        original = _plan(_paragraph("1", "2", "4", "3", "5", "6"))
    elif fault == "keep_lines":
        original = _plan(_paragraph("1"), dict(type="paragraph", segments=["2-6"], keepLines=True))
    else:
        original = _plan(_paragraph("1"), _paragraph("2-6"))
    assert revise_structure(original, segments, request) is original


# 글자 모양으로 찾은 제목 후보가 제목 없는 섹션의 첫 줄이면 그 섹션의 제목이다.
_ENTRIES = [
    _span("Company · Engineer", 221, 100, 330, 10),
    _span("2023.01 ~ 현재", 503, 99.3, 561, 9),
    _span("Team | duties", 221, 115, 500, 9),
    _span("more duties", 221, 129, 400, 9),
    _span("학력", 221, 160, 241, 11),
    _span("School of Engineering", 221, 186, 334, 10),
    _span("졸업", 221, 200, 260, 9),
]


def _section(title: list[str] | None, *blocks: dict[str, Any]) -> dict[str, Any]:
    return dict(type="section", title=title, blocks=list(blocks))


def _entries_request() -> DocumentNormalizationRequest:
    spans = [
        dict(span, color=color)
        for span, color in zip(
            _ENTRIES, ["#1b1b1b", "#666666", "#555555", "#555555", "#222222", "#1b1b1b", "#555555"], strict=True
        )
    ]
    return _request(DocumentType.RESUME, _page(spans))


def test_an_untitled_section_takes_its_first_heading_line_as_title() -> None:
    request = _entries_request()
    segments = build_segments(request.pages)
    original = _plan(
        _section(None, _paragraph("1-2"), _paragraph("3-4")),
        _section(None, _paragraph("5-7")),
    )

    corrected = revise_structure(original, segments, request)

    # 같은 줄 오른쪽의 기간은 제목에 함께 들어간다. 섹션마다 제목은 하나만 올린다.
    assert corrected == _plan(
        _section(["1", "2"], _paragraph("3-4")),
        _section(["5"], _paragraph("6", "7")),
    )
    assert revise_structure(corrected, segments, request) is corrected


@pytest.mark.parametrize(
    "section",
    [
        _section(["3"], _paragraph("1-2"), _paragraph("4")),  # 모델이 단 제목은 바꾸지 않는다
        _section(None, _paragraph("3-4"), _paragraph("1-2")),  # 첫 줄이 제목 후보가 아니다
        _section(None, _paragraph("1-2")),  # 제목을 빼면 섹션이 빈다
        _section(None, dict(type="paragraph", segments=["1-4"], keepLines=True)),  # 줄을 지키라는 문단
    ],
)
def test_sections_without_a_clear_heading_line_keep_their_plan(section: dict[str, Any]) -> None:
    request = _entries_request()
    segments = build_segments(request.pages)
    original = _plan(section, _paragraph("5-7"))
    assert revise_structure(original, segments, request) is original


def test_headings_outside_sections_are_not_promoted() -> None:
    request = _entries_request()
    segments = build_segments(request.pages)
    original = _plan(_paragraph("1-2"), _paragraph("3-4"), _paragraph("5-7"))
    assert revise_structure(original, segments, request) is original


# 1쪽에서 가장 큰 글자 한 줄(이름, 서류 제목)은 모델이 문서 제목을 비워 두었을 때 문서 제목이다.
_HEADLINE = [
    _span("김민지 Kim Minji", 221, 68, 344, 12),
    dict(_span("전구체 공침부터 소성까지 이어서 보는", 221, 90, 559, 11), color="#e15759"),
    dict(_span("공정 연구 6년차", 221, 106.5, 293, 11), color="#e15759"),
    dict(_span("NCM811 전구체 연속 공침 공정 2년의", 221, 133.5, 563, 9.6), color="#333333"),
    dict(_span("경력을 가지고 있습니다.", 221, 148.5, 317, 9.6), color="#333333"),
]


def _headline_request(*extra: dict[str, Any]) -> DocumentNormalizationRequest:
    return _request(DocumentType.RESUME, _page([*_HEADLINE, *extra]))


def _titled(title: list[str] | None, *blocks: dict[str, Any]) -> DocumentPlan:
    return DocumentPlan.model_validate(dict(title=title, blocks=list(blocks)))


def test_the_largest_line_on_the_first_page_becomes_the_missing_document_title() -> None:
    request = _headline_request()
    segments = build_segments(request.pages)
    original = _titled(None, _section(None, _paragraph("1"), _paragraph("2-3"), _paragraph("4-5")))

    corrected = revise_structure(original, segments, request)

    assert corrected == _titled(["1"], _section(["2", "3"], _paragraph("4-5")))
    assert revise_structure(corrected, segments, request) is corrected


def test_a_document_title_from_the_model_is_kept() -> None:
    request = _headline_request()
    segments = build_segments(request.pages)
    original = _titled(["2-3"], _paragraph("1"), _paragraph("4-5"))
    assert revise_structure(original, segments, request) is original


@pytest.mark.parametrize(
    ("extra", "first"),
    [
        ((), _paragraph("1-3")),  # 가장 큰 줄이 긴 문단 안에 묶여 있다
        ((dict(_span("같은 크기의 다른 줄", 221, 400, 330, 12)),), _paragraph("1")),  # 가장 큰 글자가 한 줄뿐이 아니다
    ],
)
def test_the_document_title_needs_one_unique_largest_line_on_its_own(
    extra: tuple[dict[str, Any], ...], first: dict[str, Any]
) -> None:
    request = _headline_request(*extra)
    segments = build_segments(request.pages)
    rest = [_paragraph("2-3") if first["segments"] == ["1"] else None, _paragraph("4-5")]
    original = _titled(None, _section(None, first, *[b for b in rest if b]), *([_paragraph("6")] if extra else []))
    assert revise_structure(original, segments, request).title is None
