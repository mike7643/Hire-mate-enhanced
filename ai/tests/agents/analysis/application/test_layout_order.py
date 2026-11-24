"""모델이 읽는 순서를 어긴 구조를 PDF 조각 번호와 글자 모양으로 바로잡는 규칙.

조각 번호는 PDF를 읽는 순서다. 모델이 제목을 엉뚱한 자리에 두거나 본문 줄을 제목으로 쓰거나, 다른 쪽 글을 표에
넣으면 원문 글자는 그대로여도 읽는 구조가 틀린다. 여기의 규칙은 문서 종류나 특정 양식에 기대지 않는다.
"""

from typing import Any
from uuid import uuid4

from hiremate_ai.agents.analysis.application.layout_structure import revise_structure
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, Segment, build_segments
from hiremate_ai.contract import DocumentNormalizationRequest, DocumentType, PdfPage

_BODY = "#1b1b1b"


def _span(text: str, x: float, y: float, end_x: float, size: float = 10.0, color: str = _BODY) -> dict[str, Any]:
    return dict(text=text, x=x, y=y, endX=end_x, size=size, color=color)


def _page(number: int, spans: list[dict[str, Any]]) -> PdfPage:
    layout = dict(width=600, height=800)
    return PdfPage.model_validate(
        dict(number=number, text="\n".join(s["text"] for s in spans), spans=spans, layout=layout)
    )


def _request(
    *pages: PdfPage, kind: DocumentType = DocumentType.COVER_LETTER
) -> tuple[DocumentNormalizationRequest, list[Segment]]:
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
    return request, build_segments(list(pages))


def _plan(*blocks: dict[str, Any], title: list[str] | None = None) -> DocumentPlan:
    return DocumentPlan.model_validate({"title": title, "blocks": list(blocks)})


def _p(*refs: str) -> dict[str, Any]:
    return {"type": "paragraph", "segments": list(refs)}


def _s(title: list[str] | None, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "section", "title": title, "blocks": list(blocks)}


def _shape(blocks: list[Any]) -> list[Any]:
    """구조를 (제목, [자식]) 과 조각 번호 목록으로 줄여 비교한다."""
    shaped: list[Any] = []
    for block in blocks:
        if block.type == "section":
            shaped.append((block.title, _shape(block.blocks)))
        elif block.type == "paragraph":
            shaped.append(block.segments)
        elif block.type == "list":
            shaped.append([item.segments for item in block.items])
        elif block.type == "table":
            shaped.append(("table", block.headers, block.rows))
        else:
            shaped.append(block.type)
    return shaped


def _letter() -> list[PdfPage]:
    """대괄호 제목 셋과 본문 줄. 조각: 1 [동기], 2-3 본문, 4 [경험], 5-6 본문, 7 [포부], 8 본문."""
    lines = [
        ("[동기]", 120), ("동기 첫 줄입니다 이어지는 본문", 140), ("동기 끝 줄입니다.", 160),
        ("[경험]", 200), ("경험 첫 줄입니다 이어지는 본문", 220), ("경험 끝 줄입니다.", 240),
        ("[포부]", 280), ("포부 한 줄입니다.", 300),
    ]  # fmt: skip
    return [_page(1, [_span(text, 60, y, 60 + 8 * len(text)) for text, y in lines])]


def test_a_title_placed_after_earlier_text_takes_only_the_text_that_follows_it() -> None:
    # 모델이 앞 섹션의 끝 문단(3)을 다음 제목(4) 아래에 둔 경우: 그 문단은 앞 섹션 끝으로 돌아간다.
    request, segments = _request(*_letter())
    plan = _plan(_s(["1"], _p("2")), _s(["4"], _p("3"), _p("5-6")), _s(["7"], _p("8")))
    revised = revise_structure(plan, segments, request)
    assert _shape(revised.blocks) == [(["1"], [["2"], ["3"]]), (["4"], [["5-6"]]), (["7"], [["8"]])]


def test_shifted_titles_are_realigned_and_a_skipped_heading_line_becomes_a_title() -> None:
    # 모델이 첫 제목(1)을 둘째 섹션에 붙이고 둘째 제목(4)을 빠뜨려 제목이 한 칸씩 밀린 경우.
    request, segments = _request(*_letter())
    plan = _plan(_s(None, _p("2-3")), _s(["1"], _p("5-6")), _s(["7"], _p("8")))
    revised = revise_structure(plan, segments, request)
    assert _shape(revised.blocks) == [(["1"], [["2-3"]]), (["4"], [["5-6"]]), (["7"], [["8"]])]


def test_sections_already_in_reading_order_are_left_alone() -> None:
    request, segments = _request(*_letter())
    plan = _plan(_s(["1"], _p("2-3")), _s(["4"], _p("5-6")), _s(["7"], _p("8")))
    assert revise_structure(plan, segments, request) == plan


def test_titles_are_not_moved_when_the_model_reordered_the_content_itself() -> None:
    # 내용 블록끼리 순서가 바뀐 경우(사이드바를 뒤로 보낸 것처럼)는 모델 판단으로 보고 건드리지 않는다.
    request, segments = _request(*_letter())
    plan = _plan(_s(["4"], _p("5-6")), _s(["1"], _p("2-3")), _s(["7"], _p("8")))
    assert revise_structure(plan, segments, request) == plan


def test_side_labels_left_of_their_answers_keep_the_model_sections() -> None:
    # 왼쪽 칸 이름표 양식: 이름표가 답 옆 가운데쯤에 있어 번호가 답 첫 줄보다 뒤다. 번호만으로 옮기지 않는다.
    spans = [
        _span("성장 과정에서 배운 점을 적은 첫 줄", 116, 100, 530), _span("둘째 줄", 116, 118, 530),
        _span("셋째 줄", 116, 136, 400), _span("성장과정", 53, 140, 93, size=9.0),
        _span("지원 동기를 적은 첫 줄", 116, 200, 530), _span("둘째 줄입니다.", 116, 218, 300),
        _span("지원동기", 53, 222, 93, size=9.0),
    ]  # fmt: skip
    request, segments = _request(_page(1, spans))
    plan = _plan(_s(["4"], _p("1-3")), _s(["7"], _p("5-6")))
    assert revise_structure(plan, segments, request) == plan


def _marked_letter(*heads: str) -> list[PdfPage]:
    """기호로 시작하는 짧은 제목(본문보다 작은 글자)과 본문. 조각: 1 문서 제목, 그 뒤 제목마다 [제목, 본문, 본문]."""
    spans = [_span("자기소개서", 250, 60, 345, size=19)]
    y = 100.0
    for head in heads:
        spans.append(_span(head, 51, y, 51 + 9 * len(head), size=9.5))
        spans += [_span("본문 줄입니다 이어 쓰는 문장", 53, y + 18, 520), _span("본문 끝 줄입니다.", 53, y + 33, 200)]
        y += 70
    return [_page(1, spans)]


def test_short_lines_led_by_the_same_heading_mark_are_heading_lines() -> None:
    # "■ 성장과정"처럼 같은 기호로 시작하는 짧은 줄이 본문 앞에 되풀이되면 글자가 작아도 제목 줄이다.
    request, segments = _request(*_marked_letter("■ 성장과정", "■ 지원동기", "■ 입사 후 포부"))
    plan = _plan(_s(None, _p("3-4")), _s(["2"], _p("6-7")), _s(["8"], _p("9-10")), title=["1"])
    revised = revise_structure(plan, segments, request)
    assert _shape(revised.blocks) == [(["2"], [["3-4"]]), (["5"], [["6-7"]]), (["8"], [["9-10"]])]


def test_a_heading_mark_used_once_is_not_a_heading_line() -> None:
    request, segments = _request(*_marked_letter("■ 성장과정"))
    plan = _plan(_s(None, _p("2-4")), title=["1"])
    assert revise_structure(plan, segments, request) == plan


def test_consecutive_lines_led_by_the_same_mark_are_list_items_not_headings() -> None:
    spans = [_span("자기소개서", 250, 60, 345, size=19)]
    spans += [_span(text, 51, 100 + 15 * i, 150, size=9.5) for i, text in enumerate(["■ Python", "■ SQL", "■ Excel"])]
    spans += [_span("본문 줄입니다.", 53, 160, 200)]
    request, segments = _request(_page(1, spans))
    plan = _plan(_s(None, _p("2-5")), title=["1"])
    assert revise_structure(plan, segments, request) == plan


def _page_break_letter(*, last_line_end: float = 530) -> list[PdfPage]:
    """쪽이 바뀌며 문단이 이어지는 자기소개서.

    조각: 1 [동기], 2-3 본문(3은 1쪽 마지막 줄), 4-5 2쪽 본문, 6 [경험], 7 본문.
    """
    first = _page(
        1,
        [
            _span("[동기]", 60, 700, 110),
            _span("동기 첫 줄은 오른쪽 끝까지 이어 쓴 본문입니다", 60, 720, 532),
            _span("다음 쪽으로 넘어가는 본문 줄입니다", 60, 740, last_line_end),
        ],
    )
    second = _page(
        2,
        [
            _span("이어지는 문장이 여기서 끝납니다 그리고 한 줄 더", 60, 76, 528),
            _span("문단 끝 줄입니다.", 60, 95, 250),
            _span("[경험]", 60, 135, 110),
            _span("경험 본문입니다.", 60, 155, 250),
        ],
    )
    return [first, second]


def test_a_title_that_continues_the_previous_line_goes_back_into_the_text() -> None:
    # 모델이 다음 쪽 첫 줄(본문이 이어지는 줄)을 섹션 제목으로 쓴 경우: 앞 문단에 다시 붙인다.
    request, segments = _request(*_page_break_letter())
    plan = _plan(_s(["1"], _p("2-3")), _s(["4"], _p("4-5")), _s(["6"], _p("7")))
    revised = revise_structure(plan, segments, request)
    assert _shape(revised.blocks) == [(["1"], [["2-3", "4-5"]]), (["6"], [["7"]])]


def test_a_title_after_a_finished_paragraph_stays_a_title() -> None:
    # 앞 줄이 짧게 끝났으면 문단이 끝난 것이다. 본문과 글자 모양이 같은 제목일 수 있어 그대로 둔다.
    request, segments = _request(*_page_break_letter(last_line_end=250))
    plan = _plan(_s(["1"], _p("2-3")), _s(["4"], _p("5")), _s(["6"], _p("7")))
    assert revise_structure(plan, segments, request) == plan


def _resume() -> list[PdfPage]:
    """큰 제목 셋(12pt)과 그 아래 작은 제목.

    조각: 1 경력, 2 회사, 3 본문, 4 프로젝트, 5 프로젝트명, 6 본문, 7 학력, 8 본문.
    """
    lines = [
        ("경력 사항", 54, 100, 12.0), ("로지온 마케팅팀", 43, 125, 11.0), ("검색광고 운영", 57, 140, 9.5),
        ("주요 프로젝트", 54, 180, 12.0), ("리드 대시보드 구축", 53, 205, 10.0), ("주간 리포트 자동화", 53, 220, 9.5),
        ("학력 · 자격 · 교육", 54, 260, 12.0), ("해든대학교 경영학과", 48, 285, 9.3),
    ]  # fmt: skip
    return [_page(1, [_span(text, x, y, x + 9 * len(text), size=size) for text, x, y, size in lines])]


def test_a_title_styled_like_its_parent_title_is_lifted_to_the_parent_level() -> None:
    # 같은 모양의 큰 제목은 같은 단계다. 모델이 앞 큰 제목 아래에 넣었으면 그 뒤 형제로 올린다.
    request, segments = _request(*_resume(), kind=DocumentType.RESUME)
    plan = _plan(_s(["1"], _s(["2"], _p("3")), _s(["4"], _s(["5"], _p("6")), _s(["7"], _p("8")))))
    revised = revise_structure(plan, segments, request)
    assert _shape(revised.blocks) == [
        (["1"], [(["2"], [["3"]])]),
        (["4"], [(["5"], [["6"]])]),
        (["7"], [["8"]]),
    ]


def test_titles_styled_differently_from_their_parent_keep_their_nesting() -> None:
    request, segments = _request(*_resume(), kind=DocumentType.RESUME)
    plan = _plan(_s(["1"], _s(["2"], _p("3"))), _s(["4"], _s(["5"], _p("6"))), _s(["7"], _p("8")))
    assert revise_structure(plan, segments, request) == plan


def _forms() -> list[PdfPage]:
    """빈 양식 표 둘 뒤에 다음 쪽 경력기술서. 조각: 1 어학, 2-3 머리 칸, 4 교육, 5-6 머리 칸, 7 경력기술서, 8-9 본문."""
    first = _page(
        1,
        [
            _span("어학", 43, 100, 62, size=10.5), _span("기간", 77, 125, 93), _span("시험", 259, 125, 275),
            _span("교육", 43, 200, 62, size=10.5), _span("기간", 77, 225, 93), _span("과정명", 255, 225, 279),
        ],
    )  # fmt: skip
    second = _page(
        2,
        [
            _span("경력기술서", 238, 67, 351, size=18),
            _span("한빛정보시스템 | PMO지원팀 대리", 43, 110, 272, size=11.5),
            _span("공공기관 정보시스템 구축 사업 PMO 지원", 57, 144, 306),
        ],
    )
    return [first, second]


def _table(headers: list[str], *rows: list[str]) -> dict[str, Any]:
    return {"type": "table", "headers": [[n] for n in headers], "rows": [[[n] for n in row] for row in rows]}


def test_table_cells_past_a_later_section_title_leave_the_table() -> None:
    # 모델이 앞 쪽 빈 양식 표의 행에 다음 섹션 제목 너머(다른 쪽)의 줄을 넣은 경우: 그 칸을 표에서 뺀다.
    # 뒤 블록이 쓰는 조각은 그 블록에 맡기고, 아무도 쓰지 않는 조각은 읽는 자리로 되돌린다.
    request, segments = _request(*_forms(), kind=DocumentType.RESUME)
    plan = _plan(
        _s(["1"], _table(["2", "3"], ["7", "9"])),
        _s(["4"], _table(["5", "6"])),
        _s(["7"], _p("8")),
    )
    revised = revise_structure(plan, segments, request)
    assert _shape(revised.blocks) == [
        (["1"], [("table", [["2"], ["3"]], [])]),
        (["4"], [("table", [["5"], ["6"]], [])]),
        (["7"], [["8"], ["9"]]),
    ]


def test_a_table_continuing_on_the_next_page_keeps_its_cells() -> None:
    request, segments = _request(*_forms(), kind=DocumentType.RESUME)
    plan = _plan(_s(["1"], _table(["2", "3"]), _p("4"), _table(["5", "6"], ["8", "9"])), _p("7"))
    assert revise_structure(plan, segments, request) == plan
