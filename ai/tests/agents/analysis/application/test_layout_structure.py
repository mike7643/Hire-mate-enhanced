from copy import deepcopy
from typing import Any
from uuid import uuid4

import pytest

from hiremate_ai.agents.analysis.application.layout_structure import ensure_visual_notices, revise_structure
from hiremate_ai.agents.analysis.application.normalize import (
    DocumentPlan,
    ParagraphPlan,
    SectionPlan,
    Segment,
    build_segments,
)
from hiremate_ai.contract import (
    DocumentNormalizationRequest,
    DocumentParagraph,
    DocumentSection,
    DocumentType,
    DocumentVisualNotice,
    NormalizedDocument,
    PdfPage,
)


def _span(text: str, x: float, y: float, end_x: float, size: float = 10) -> dict[str, Any]:
    return dict(text=text, x=x, y=y, endX=end_x, size=size)


def _box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return dict(x1=x1, y1=y1, x2=x2, y2=y2)


def _page(number: int, spans: list[dict[str, Any]], **evidence: Any) -> PdfPage:
    return PdfPage.model_validate(
        dict(number=number, text="\n".join(s["text"] for s in spans), spans=spans, **evidence)
    )


def _request(kind: DocumentType, *pages: PdfPage) -> tuple[DocumentNormalizationRequest, list[Segment]]:
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=kind,
            pages=list(pages),
        )
    )
    return request, build_segments(request.pages)


def _plan(*blocks: dict[str, Any], title: list[str] | None = None) -> DocumentPlan:
    return DocumentPlan.model_validate(dict(title=title, blocks=list(blocks)))


def _paragraph(*ids: str, keep_lines: bool = False) -> dict[str, Any]:
    return dict(type="paragraph", segments=list(ids), keepLines=keep_lines)


def _section(title: str, *blocks: dict[str, Any]) -> dict[str, Any]:
    return dict(type="section", title=[title], blocks=list(blocks))


def _form_page(*, fault: str = "", filled_rules: bool = False) -> PdfPage:
    spans = [_span("Title or preface", 40, 50, 120)]
    for top in (100, 220, 340):
        spans += [
            _span("A complete answer line with meaningful content", 116, top + 15, 345),
            _span("Another long line in the form's primary text column", 116, top + 30, 360),
            _span("Label", 48, top + 41, 85),
            _span("A third answer line aligned with the two above it", 116, top + 45, 372),
            _span("A separate paragraph after a visible blank line", 116, top + 85, 355),
        ]
    spans.append(_span("Closing signature", 40, 490, 122))
    rules = [_box(left, y, right, y) for y in (100, 220, 340, 460) for left, right in ((40, 250), (251, 560))]
    rules += [_box(102, 100, 102, 460)]
    if fault == "two_bands":
        del rules[:2]
    if fault == "crossed_divider":
        spans[1]["x"] = 95
    fills = []
    if filled_rules:
        for rule in rules:
            x1, y1, x2, y2 = (rule[k] for k in ("x1", "y1", "x2", "y2"))
            rect = _box(x1, y1 - 0.375, x2, y2 + 0.375) if y1 == y2 else _box(x1 - 0.375, y1, x2 + 0.375, y2)
            fills.append({**rect, "color": "#233345"})
        # A parser may report only the label-side grid lines but still expose all visible filled rectangles.
        rules = [_box(40, y, 102, y) for y in (100, 220, 340, 460)]
    layout = dict(width=600, height=800, fillRects=fills)
    return _page(1, spans, gridLines=rules, layout=None if fault == "no_layout" else layout)


@pytest.mark.parametrize("filled_rules", [False, True])
def test_repeated_ruled_cover_letter_forms_replace_partial_refs_and_split_by_line_pitch(filled_rules: bool) -> None:
    request, segments = _request(DocumentType.COVER_LETTER, _form_page(filled_rules=filled_rules))
    plan = _plan(
        _paragraph("1-3", keep_lines=True),
        dict(type="table", title=None, headers=[["4"], ["5-6"]], rows=[[["9"], ["7-8", "10-11"]]]),
        _paragraph("12-15", "17", keep_lines=True),  # 16 omitted by the model
    )
    before = deepcopy(plan)

    corrected = revise_structure(plan, segments, request)

    assert corrected == _plan(
        _paragraph("1", keep_lines=True),
        _section("4", _paragraph("2", "3", "5"), _paragraph("6")),
        _section("9", _paragraph("7", "8", "10"), _paragraph("11")),
        _section("14", _paragraph("12", "13", "15"), _paragraph("16")),
        _paragraph("17", keep_lines=True),
    )
    assert plan == before
    assert revise_structure(corrected, segments, request) == corrected


def test_form_repair_keeps_a_preceding_parent_title_and_model_notice_in_place() -> None:
    request, segments = _request(DocumentType.COVER_LETTER, _form_page())
    notice = _visual_plan(1)
    plan = _plan(_section("1", _paragraph("2-16"), notice, _paragraph("17")))

    corrected = revise_structure(plan, segments, request)

    assert corrected == _plan(
        _section(
            "1",
            _section("4", _paragraph("2", "3", "5"), _paragraph("6")),
            _section("9", _paragraph("7", "8", "10"), _paragraph("11")),
            _section("14", _paragraph("12", "13", "15"), _paragraph("16")),
            notice,
            _paragraph("17"),
        )
    )


@pytest.mark.parametrize("fault", ["no_layout", "two_bands", "crossed_divider"])
def test_does_not_rebuild_cover_letter_forms_without_three_unambiguous_bands(fault: str) -> None:
    request, segments = _request(DocumentType.COVER_LETTER, _form_page(fault=fault))
    plan = _plan(_paragraph("1-17"))

    assert revise_structure(plan, segments, request) is plan


def test_unnamed_section_is_titled_when_the_next_question_line_is_in_plain_text() -> None:
    """다음 문항을 모델이 섹션으로 세우지 않았어도, 번호와 글자 모양이 이어지는 다음 문항 줄이 범위를 정한다."""
    request, segments = _request(DocumentType.COVER_LETTER, _questions_page())
    original = _plan(dict(type="section", title=None, blocks=[_paragraph("2-6")]), _paragraph("7-10"), title=["1"])
    corrected = revise_structure(original, segments, request)
    expected = SectionPlan(
        type="section", title=["2"], blocks=[ParagraphPlan(type="paragraph", segments=["3", "4", "5", "6"])]
    )
    assert corrected.blocks[0] == expected


def test_does_not_replace_document_title_or_revise_an_unrelated_kind() -> None:
    page = _form_page()
    cover, segments = _request(DocumentType.COVER_LETTER, page)
    resume, _ = _request(DocumentType.RESUME, page)
    plan = _plan(_paragraph("1-3", "5-17"), title=["4", "9", "14"])

    assert revise_structure(plan, segments, cover) is plan
    assert revise_structure(plan, segments, resume) is plan


def _questions_page(*, fault: str = "") -> PdfPage:
    spans = [
        _span("Preface", 45, 50, 90),
        _span("1. Approach", 45, 105, 130, 10.5),
        _span("[A supported subtopic]", 45, 130, 180),
        _span("Opening lines of the missing answer.", 45, 147, 240),
        _span("[15]", 45, 164, 75),
        _span("The next lines are still part of the answer.", 45, 181, 270),
        _span("2) Delivery", 45, 215, 130, 10.5),
        _span("The model kept this answer.", 45, 241, 238),
        # 괄호 소제목 규칙(test_layout_brackets)과 섞이지 않게 대괄호를 쓰지 않는다.
        _span("(Unstyled prose label)", 45, 260, 170),
        _span("The model kept this paragraph as ordinary text.", 45, 277, 290),
    ]
    if fault == "size":
        spans[1]["size"] = 10.3
    if fault == "alignment":
        spans[2]["x"] = 49
    if fault == "gap":
        spans[2]["y"] = 117
    return _page(1, spans)


def test_missing_numbered_question_is_recovered_from_typography_without_page_layout() -> None:
    request, segments = _request(DocumentType.COVER_LETTER, _questions_page())
    retained = _section("7", _paragraph("8-10"))
    plan = _plan(retained, title=["1"])

    corrected = revise_structure(plan, segments, request)

    assert corrected == _plan(
        _section("2", _section("3", _paragraph("4", "5", "6"))),
        retained,
        title=["1"],
    )
    assert corrected.blocks[-1] is plan.blocks[-1]
    assert revise_structure(corrected, segments, request) is corrected


def test_unnamed_section_gets_its_numbered_title_without_rewriting_the_answer() -> None:
    request, segments = _request(DocumentType.COVER_LETTER, _questions_page())
    next_section = _section("7", _paragraph("8-10"))
    original = _plan(
        dict(type="section", title=None, blocks=[_paragraph("2-4"), _paragraph("5-6")]),
        next_section,
        title=["1"],
    )
    corrected = revise_structure(original, segments, request)
    assert corrected == _plan(_section("2", _paragraph("3", "4"), _paragraph("5-6")), next_section, title=["1"])
    assert revise_structure(corrected, segments, request) is corrected


async def test_unnamed_section_is_titled_when_the_model_left_the_rest_of_its_answer_unreferenced() -> None:
    """실제 Gemma 응답 모양: 문항 끝 조각을 계획에 넣지 않았다. 빠진 조각은 조립 때 같은 문항 안에 되살아난다."""
    from hiremate_ai.agents.analysis.application.normalize import assemble

    request, segments = _request(DocumentType.COVER_LETTER, _questions_page())
    next_section = _section("7", _paragraph("8-10"))
    original = _plan(dict(type="section", title=None, blocks=[_paragraph("2-4")]), next_section, title=["1"])

    corrected = revise_structure(original, segments, request)

    assert corrected == _plan(_section("2", _paragraph("3", "4")), next_section, title=["1"])
    assert revise_structure(corrected, segments, request) is corrected
    document, report = assemble(corrected, segments, request, 1)
    assert report.missing == [5, 6]
    first = document.blocks[0]
    assert isinstance(first, DocumentSection) and first.title == segments[1].text
    assert all(isinstance(block, DocumentParagraph) for block in first.blocks)
    assert [block.text for block in first.blocks if isinstance(block, DocumentParagraph)] == [
        f"{segments[2].text} {segments[3].text}",
        f"{segments[4].text} {segments[5].text}",
    ]


@pytest.mark.parametrize("fault", ["different_heading_style", "claimed_elsewhere"])
def test_unnamed_section_is_not_relabelled_without_scope_and_sibling_evidence(fault: str) -> None:
    from dataclasses import replace

    request, segments = _request(DocumentType.COVER_LETTER, _questions_page())
    if fault == "different_heading_style":
        segments[6] = replace(segments[6], size=12)
    body = _paragraph("2-5" if fault == "claimed_elsewhere" else "2-6")
    next_block = _paragraph("7-10") if fault == "no_neighbouring_section" else _section("7", _paragraph("8-10"))
    extra = [_paragraph("6")] if fault == "claimed_elsewhere" else []
    original = _plan(dict(type="section", title=None, blocks=[body]), next_block, *extra, title=["1"])
    assert revise_structure(original, segments, request) is original


@pytest.mark.parametrize("fault", ["size", "alignment", "gap", "heading_already_used", "body_already_used"])
def test_leaves_numbered_headings_without_repeated_typography_and_missing_text(fault: str) -> None:
    request, segments = _request(DocumentType.COVER_LETTER, _questions_page(fault=fault))
    retained = _section("7", _paragraph("8-10"))
    blocks = {
        "heading_already_used": [_paragraph("2"), retained],
        "body_already_used": [_paragraph("3-6"), retained],
    }.get(fault, [retained])
    plan = _plan(*blocks, title=["1"])

    assert revise_structure(plan, segments, request) is plan


def _visual_plan(*pages: int) -> dict[str, Any]:
    return dict(type="visual_notice", pages=list(pages), visualType="image", message="모델의 원래 안내")


def _caption_page(*, fault: str = "") -> PdfPage:
    spans = [
        _span("Project overview", 60, 100, 200),
        _span("Left figure caption", 90, 420, 215),
        _span("Right figure caption", 335, 420, 460),
        _span("Continued left caption", 90, 432, 205),
        _span("Different text style", 335, 434, 457, 12),
        _span("Too distant to be the next line", 335, 444, 490),
        _span("Unrelated prose", 60, 480, 210),
    ]
    # The placed images arrive out of left-to-right order.
    images = [_box(320, 210, 540, 410), _box(60, 210, 260, 410)]
    if fault == "one_image":
        images.pop()
    if fault == "not_unique":
        images.append(_box(62, 215, 265, 412))
    if fault == "not_below":
        spans[2]["y"] = 410
    layout = dict(width=600, height=800, imageBoxes=images)
    return _page(1, spans, imageCount=len(images), layout=None if fault == "no_layout" else layout)


@pytest.mark.parametrize("block_type", ["paragraph", "table"])
def test_splits_caption_groups_left_to_right_and_recovers_only_a_matching_missing_line(block_type: str) -> None:
    request, segments = _request(DocumentType.PORTFOLIO, _caption_page())
    combined = (
        _paragraph("3", "2")
        if block_type == "paragraph"
        else dict(type="table", title=["3"], headers=[], rows=[[["2"]]])
    )
    plan = _plan(_section("1", combined, _visual_plan(1), _paragraph("7")))
    before = deepcopy(plan)

    corrected = revise_structure(plan, segments, request)

    assert corrected == _plan(_section("1", _paragraph("2", "4"), _paragraph("3"), _visual_plan(1), _paragraph("7")))
    assert plan == before
    assert revise_structure(corrected, segments, request) is corrected


@pytest.mark.parametrize("fault", ["no_layout", "one_image", "not_unique", "not_below", "mixed_refs", "unknown_ref"])
def test_does_not_split_caption_blocks_without_exclusive_image_support(fault: str) -> None:
    request, segments = _request(DocumentType.PORTFOLIO, _caption_page(fault=fault))
    extra = {"mixed_refs": ["7"], "unknown_ref": ["8-9"]}.get(fault, [])
    plan = _plan(_section("1", _paragraph("2-3", *extra)))

    assert revise_structure(plan, segments, request) is plan


def test_caption_recovery_does_not_duplicate_a_line_already_in_the_model_plan() -> None:
    request, segments = _request(DocumentType.PORTFOLIO, _caption_page())
    referenced = _section("7", dict(type="list", items=[dict(segments=["4"])]))
    plan = _plan(_paragraph("2-3"), referenced, title=["1"])

    assert revise_structure(plan, segments, request) == _plan(_paragraph("2"), _paragraph("3"), referenced, title=["1"])


def _notice_page(number: int, caption: str = "그림 1 결과 개요", *, fault: str = "") -> PdfPage:
    body = "The longer prose above the chart establishes the page's normal body font size for comparison."
    image = _box(80, 180, 520, 410)
    spans = [
        _span(body, 80, 150, 500),
        _span(caption, 218, 428, 382, 8),
        _span("for row in data:", 110, 470, 300, 6),
        _span("    print(row)", 110, 482, 300, 6),
    ]
    if fault == "portrait":
        image = _box(190, 145, 410, 410)
    if fault == "small":
        image = _box(190, 365, 410, 410)
    if fault == "background":
        image = _box(0, -130, 600, 560)
        spans[1]["y"] = 576
    if fault == "not_centered":
        spans[1].update(x=91, endX=176)
    if fault == "too_far":
        spans[1]["y"] = 444
    if fault == "same_size":
        spans[1]["size"] = 10
    if fault == "no_body_above":
        spans[0]["y"] = 500  # Only code-size noise precedes the end of the page.
    if fault == "two_labels":
        spans.append(_span("그림 2 비교 결과", 238, 436, 376, 8))
    layout = dict(width=600, height=800, imageBoxes=[] if fault == "no_boxes" else [image])
    return _page(
        number,
        spans,
        imageCount=0 if fault == "no_image_objects" else 1,
        layout=None if fault == "no_layout" else layout,
    )


def _normalized(page_count: int, *blocks: dict[str, Any]) -> NormalizedDocument:
    return NormalizedDocument.model_validate(
        dict(document=dict(id=uuid4(), type="portfolio", title="Example project", pageCount=page_count), blocks=blocks)
    )


def test_appends_only_missing_visual_notice_pages_in_order_and_preserves_metadata() -> None:
    existing = {
        "id": "existing-section",
        "type": "section",
        "title": "Project evidence",
        "pages": [1, 2],
        "blocks": [
            {**_visual_plan(2), "id": "original-notice", "detection": "suspected"},
            dict(
                type="list", id="list", pages=[1], items=[dict(id="visual-notice-page-1", pages=[1], text="Unchanged")]
            ),
        ],
    }
    document = _normalized(3, existing)
    pages = [_notice_page(3), _notice_page(2), _notice_page(1, fault="two_labels")]
    segments = build_segments(pages)

    result = ensure_visual_notices(document, pages, segments)

    assert result.document is document.document
    assert result.schema_version == document.schema_version
    assert result.blocks[:1] == document.blocks
    assert result.blocks[0] is document.blocks[0]
    assert result.blocks[1:] == [
        DocumentVisualNotice(
            id=id_,
            type="visual_notice",
            pages=[page],
            visual_type="unknown",
            detection="suspected",
            message=f"원본 PDF {page}쪽의 시각 자료를 확인해 주세요.",
        )
        for id_, page in [("visual-notice-page-1-extra", 1), ("visual-notice-page-3", 3)]
    ]
    assert len(document.blocks) == 1
    assert ensure_visual_notices(result, pages, segments) is result


def test_appends_notice_for_smaller_measured_trend_below_wide_image() -> None:
    document = _normalized(1)
    page = _notice_page(1, "활성 사용자 비율이 12% 증가했습니다.")

    result = ensure_visual_notices(document, [page], build_segments([page]))

    assert len(result.blocks) == 1
    assert isinstance(result.blocks[0], DocumentVisualNotice) and result.blocks[0].pages == [1]


@pytest.mark.parametrize(
    ("caption", "fault"),
    [
        ("그림 1 결과 개요", "no_layout"),
        ("그림 1 결과 개요", "no_image_objects"),
        ("그림 1 결과 개요", "no_boxes"),
        ("그림 1 결과 개요", "portrait"),
        ("그림 1 결과 개요", "small"),
        ("그림 1 결과 개요", "background"),
        ("그림 1 결과 개요", "not_centered"),
        ("그림 1 결과 개요", "too_far"),
        ("그림 1 증명 사진", ""),
        ("A random string or decorative line", ""),
        ("원본의 코드 들여쓰기를 확인해 주세요.", ""),
        ("활성 사용자 비율이 증가했습니다.", ""),
        ("활성 사용자 비율이 12%입니다.", ""),
        ("활성 사용자 비율이 12% 증가했습니다.", "same_size"),
        ("활성 사용자 비율이 12% 증가했습니다.", "no_body_above"),
    ],
)
def test_does_not_add_notice_for_unsupported_page(caption: str, fault: str) -> None:
    document = _normalized(1)
    page = _notice_page(1, caption, fault=fault)

    assert ensure_visual_notices(document, [page], build_segments([page])) is document
