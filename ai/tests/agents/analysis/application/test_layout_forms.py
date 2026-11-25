"""칸 표 자기소개서(왼쪽 좁은 항목 칸, 오른쪽 본문 칸)의 항목을 섹션으로 세우는 규칙.

구인 사이트 표준 양식은 세로선으로 항목 이름 칸과 본문 칸을 나누고, 가로선으로 항목을 나눈다. 본문 칸이 쪽을 넘어
이어지면 다음 쪽 첫 칸에는 항목 이름이 없다. 모델은 항목 이름을 본문 문단에 붙이거나 쪽마다 제목 없는 섹션을
만들곤 한다. 선 좌표로 칸을 읽어 항목 이름을 제목으로, 그 칸의 본문(다음 쪽으로 이어진 부분 포함)을 섹션 내용으로
세운다. 모델 요청은 바꾸지 않는다.
"""

from typing import Any
from uuid import uuid4

from hiremate_ai.agents.analysis.application.layout_structure import revise_structure
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, build_segments
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentType, PdfPage

_LINE = "본문 칸에 이어 쓴 충분히 긴 자기소개 문장입니다 그리고 계속 이어집니다"


def _rect(x1: float, y1: float, x2: float, y2: float) -> dict[str, Any]:
    return dict(x1=x1, y1=y1, x2=x2, y2=y2, color="#000000")


def _frame(top: float, bottom: float, cuts: list[float], *, closed_top: bool) -> list[dict[str, Any]]:
    """세로선 셋(왼쪽 테두리, 항목 칸 경계, 오른쪽 테두리)과 가로선. 가로선은 칸 경계에서 두 조각이다."""
    rects = [_rect(43, top, 43.75, bottom), _rect(106, top, 106.75, bottom), _rect(553, top, 553.75, bottom)]
    for y in ([top] if closed_top else []) + cuts:
        rects += [_rect(43, y, 106, y + 0.75), _rect(106, y, 553, y + 0.75)]
    return rects


def _span(text: str, x: float, y: float, end_x: float, size: float = 10.0) -> dict[str, Any]:
    return dict(text=text, x=x, y=y, endX=end_x, size=size, color="#1b1b1b")


def _body(top: float, count: int, gap_after: int | None = None) -> list[dict[str, Any]]:
    spans, y = [], top
    for index in range(count):
        spans.append(_span(_LINE, 116, y, 540))
        y += 34 if index == gap_after else 17
    return spans


def _pages() -> list[PdfPage]:
    first = [_span("자기소개서", 250, 50, 330, 16.0), _span("성장과정", 53, 106, 93), *_body(106, 6, gap_after=2)]
    second = [
        *_body(60, 2),
        _span("지원", 59, 466, 88),
        _span("동기", 59, 483, 88),
        _span("및 포부", 57, 500, 90),
        *_body(466, 5),
    ]
    return [
        PdfPage.model_validate(
            dict(
                number=1,
                text="\n".join(s["text"] for s in first),
                spans=first,
                layout=dict(width=595, height=842, fillRects=_frame(86, 796, [], closed_top=True)),
            )
        ),
        PdfPage.model_validate(
            dict(
                number=2,
                text="\n".join(s["text"] for s in second),
                spans=second,
                layout=dict(width=595, height=842, fillRects=_frame(45, 796, [444], closed_top=False)),
            )
        ),
    ]


def _revise(plan: dict[str, Any]) -> tuple[list[Any], dict[int, str]]:
    pages = _pages()
    segments = build_segments(pages)
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=DocumentType.COVER_LETTER,
            pages=pages,
        )
    )
    revised = revise_structure(DocumentPlan.model_validate(plan), segments, request)
    return revised.blocks, {s.id: s.text.strip() for s in segments}


def _texts(refs: list[str], text: dict[int, str]) -> list[str]:
    out: list[str] = []
    for ref in refs:
        first, _, last = ref.partition("-")
        out += [text[n] for n in range(int(first), int(last or first) + 1)]
    return out


def test_form_items_become_sections_with_text_continued_on_the_next_page() -> None:
    # 조각: 1 문서 제목, 2 성장과정, 3-8 1쪽 본문, 9-10 2쪽 이어진 본문, 11-18 지원/동기/및 포부와 본문이 줄마다 섞임
    plan = {
        "title": ["1"],
        "blocks": [
            {"type": "section", "title": None, "blocks": [{"type": "paragraph", "segments": ["2-8"]}]},
            {"type": "section", "title": None, "blocks": [{"type": "paragraph", "segments": ["9-18"]}]},
        ],
    }
    blocks, text = _revise(plan)
    assert all(block.type == "section" for block in blocks)
    assert [_texts(block.title or [], text) for block in blocks if block.type == "section"] == [
        ["성장과정"],
        ["지원", "동기", "및 포부"],
    ]
    first, second = blocks
    assert first.type == "section" and second.type == "section"
    assert [len(_texts(p.segments, text)) for p in first.blocks if p.type == "paragraph"] == [3, 5]
    assert [len(_texts(p.segments, text)) for p in second.blocks if p.type == "paragraph"] == [5]


def test_a_form_without_a_narrow_item_column_is_left_alone() -> None:
    plan = {
        "title": ["1"],
        "blocks": [
            {"type": "section", "title": None, "blocks": [{"type": "paragraph", "segments": ["2-8"]}]},
            {"type": "section", "title": None, "blocks": [{"type": "paragraph", "segments": ["9-18"]}]},
        ],
    }
    pages = _pages()
    for page in pages:
        assert page.layout is not None
        page.layout.fill_rects[:] = [r for r in page.layout.fill_rects if not (105 < r.x1 < 107 and r.x2 < 108)]
    segments = build_segments(pages)
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=DocumentType.COVER_LETTER,
            pages=pages,
        )
    )
    original = DocumentPlan.model_validate(plan)
    first = revise_structure(original, segments, request).blocks[0]
    assert first.type == "section" and first.title is None


async def test_a_stacked_item_name_stays_the_whole_section_title_after_assembly() -> None:
    from hiremate_ai.agents.analysis.application.normalize import assemble

    pages = _pages()
    segments = build_segments(pages)
    request = DocumentNormalizationRequest.model_validate(
        dict(
            companyId=uuid4(),
            recruitmentId=uuid4(),
            applicationId=uuid4(),
            documentId=uuid4(),
            revision=1,
            documentType=DocumentType.COVER_LETTER,
            pages=pages,
        )
    )
    plan = DocumentPlan.model_validate(
        {
            "title": ["1"],
            "blocks": [
                {"type": "section", "title": None, "blocks": [{"type": "paragraph", "segments": ["2-8"]}]},
                {"type": "section", "title": None, "blocks": [{"type": "paragraph", "segments": ["9-18"]}]},
            ],
        }
    )
    revised = revise_structure(plan, segments, request)
    document, report = assemble(revised, segments, request, 2)
    assert [getattr(block, "title", None) for block in document.blocks] == ["성장과정", "지원 동기 및 포부"]
    assert report.split_titles == 0
