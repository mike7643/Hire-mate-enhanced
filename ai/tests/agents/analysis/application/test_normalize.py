import json
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hiremate_ai.agents.analysis.application.chunks import document_chunks
from hiremate_ai.agents.analysis.application.normalize import (
    DocumentPlan,
    build_segments,
    join_segments,
    normalize_with_report,
)
from hiremate_ai.config import settings
from hiremate_ai.contract import (
    DocumentArtifact,
    DocumentList,
    DocumentNormalizationRequest,
    DocumentParagraph,
    DocumentSection,
    DocumentTable,
    DocumentType,
    DocumentVisualNotice,
    PdfPage,
)
from hiremate_ai.llm.fake import FakeLlm


def span(text: str, x: float, y: float, end_x: float | None = None) -> dict[str, Any]:
    return {"text": text, "x": x, "y": y, "endX": end_x if end_x is not None else x + 10 * len(text)}


def page(number: int, spans: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    text = "\n".join(item["text"] for item in spans)
    return {"number": number, "text": text, "issues": [], "spans": spans, "imageCount": 0, "formCount": 0, **extra}


def request(*pages: dict[str, Any]) -> DocumentNormalizationRequest:
    return DocumentNormalizationRequest.model_validate(
        {
            "companyId": str(uuid4()),
            "recruitmentId": str(uuid4()),
            "applicationId": str(uuid4()),
            "documentId": str(uuid4()),
            "revision": 1,
            "documentType": DocumentType.RESUME,
            "pages": list(pages),
        }
    )


def plan(data: dict[str, Any]) -> DocumentPlan:
    return DocumentPlan.model_validate({"title": None, **data})


async def test_local_model_retries_one_wholly_missing_page_with_original_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_profile", "local")
    first = plan({"blocks": [{"type": "paragraph", "segments": ["1-2"]}, {"type": "paragraph", "segments": ["5"]}]})
    retry = plan({"title": ["3"], "blocks": [{"type": "paragraph", "segments": ["4"]}]})
    llm = FakeLlm(first, retry)
    pages = [
        page(1, [span("Opening text", 20, 20), span("Next line", 20, 40)]),
        page(2, [span("Restored heading", 20, 20), span("Restored body", 20, 40)]),
        page(3, [span("Referenced text", 20, 20), span("Partly missed text", 20, 40)]),
        page(4, []),
    ]
    document, report = await normalize_with_report(llm, request(*pages))
    assert [call.agent for call in llm.calls] == ["application.normalize", "application.normalize.retry.page_2"]
    original, scoped = (json.loads(c.user.split("\n", 1)[1].rsplit("\n", 1)[0]) for c in llm.calls)
    assert scoped["page_count"] == 1 and scoped["segment_count"] == 2 and scoped["pages"] == [original["pages"][1]]
    assert not llm.calls[1].images and llm.calls[1].system == llm.calls[0].system
    assert document.document.title is None
    assert [b.text for b in document.blocks if isinstance(b, DocumentParagraph) and b.pages == [2]] == [
        "Restored heading",
        "Restored body",
    ]
    assert report.missing == [6]


@pytest.mark.parametrize("foreign_ref", ["1", "4-2", "9000"])
async def test_local_page_retry_discards_out_of_scope_plan(monkeypatch: pytest.MonkeyPatch, foreign_ref: str) -> None:
    monkeypatch.setattr(settings, "llm_profile", "local")
    llm = FakeLlm(
        plan({"blocks": [{"type": "paragraph", "segments": ["1-2"]}]}),
        plan({"blocks": [{"type": "paragraph", "segments": ["3", foreign_ref]}]}),
    )
    document, report = await normalize_with_report(
        llm,
        request(
            page(1, [span("Existing first line", 20, 20), span("Existing second line", 20, 40)]),
            page(2, [span("Missed line", 20, 20)]),
        ),
    )
    assert len(llm.calls) == 2
    assert report.missing == [3] and not report.duplicated and not report.unknown
    assert [b.text for b in document.blocks if isinstance(b, DocumentParagraph)] == [
        "Existing first line Existing second line",
        "Missed line",
    ]


@pytest.mark.parametrize("profile,budget", [("local", 1), ("cloud", 8)])
async def test_no_page_retry_without_local_call_budget(
    monkeypatch: pytest.MonkeyPatch, profile: str, budget: int
) -> None:
    monkeypatch.setattr(settings, "llm_profile", profile)
    monkeypatch.setattr(settings, "llm_agent_max_calls", budget)
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-2"]}]}))
    _, report = await normalize_with_report(
        llm,
        request(
            page(1, [span("Existing first line", 20, 20), span("Existing second line", 20, 40)]),
            page(2, [span("Missed line", 20, 20)]),
        ),
    )
    assert len(llm.calls) == 1 and report.missing == [3]


def test_reference_schema_uses_explicit_ascii_digit_ranges_for_local_grammar() -> None:
    schema = DocumentPlan.model_json_schema(by_alias=True)
    segments = schema["$defs"]["ParagraphPlan"]["properties"]["segments"]
    assert segments["items"]["pattern"] == "^[0-9]+(-[0-9]+)?$"


@pytest.mark.parametrize("ref", ["", "재학기간", "１２", "12-"])
def test_reference_rejects_invalid_segment_ids(ref: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        plan({"blocks": [{"type": "paragraph", "segments": [ref]}]})


def test_reference_accepts_ascii_segment_ids_and_ranges() -> None:
    assert plan({"blocks": [{"type": "paragraph", "segments": ["1", "12-14"]}]}).blocks


@pytest.mark.parametrize("present", ["broken_table", "partial", "missing"])
async def test_aligned_grid_is_used_when_model_split_or_misplaced_cells(present: str) -> None:
    xs, ys = (40, 120, 200, 280), (80, 100, 120, 140)
    fill_rects = [{"x1": 40, "y1": y - 0.25, "x2": 280, "y2": y + 0.25, "color": "#555555"} for y in ys]
    fill_rects += [{"x1": x - 0.25, "y1": 80, "x2": x + 0.25, "y2": 140, "color": "#555555"} for x in xs]
    fill_rects += [
        {"x1": first, "y1": 80, "x2": last, "y2": 100, "color": "#dadee5"}
        for first, last in zip(xs[:-1], xs[1:], strict=True)
    ]
    cells = [["Role", "Action", "Result"], ["Editor", "Improve", "Draft"], ["Tester", "Review", "Report"]]
    spans = [span("Evaluation section", 45, 50), span("Before grid", 45, 65)]
    spans += [
        {**span(value, x + 5, y + 15, x + 70), "color": "#444488" if y == 80 else "#111111"}
        for y, row in zip(ys[:-1], cells, strict=True)
        for x, value in zip(xs[:-1], row, strict=True)
    ]
    spans += [span("After grid", 45, 150), span("Other report", 45, 165)]
    blocks: list[dict[str, Any]] = [{"type": "paragraph", "segments": ["2"]}]
    if present == "broken_table":
        blocks += [
            {"type": "table", "headers": [], "rows": [[["3", "6", "9"], ["4", "7", "10"], ["5", "8", "11"]]]},
        ]
    elif present == "partial":
        blocks += [{"type": "paragraph", "segments": ["3-6", "10"]}]
    blocks += [{"type": "paragraph", "segments": ["12"]}, {"type": "paragraph", "segments": ["13"]}]
    llm = FakeLlm(plan({"blocks": [{"type": "section", "title": ["1"], "blocks": blocks}]}))
    document, report = await normalize_with_report(
        llm,
        request(
            page(
                1,
                spans,
                layout={"width": 600, "height": 800, "fillRects": fill_rects},
            )
        ),
    )
    (section,) = document.blocks
    assert isinstance(section, DocumentSection) and section.title == "Evaluation section"
    before, table, after, other = section.blocks
    assert isinstance(table, DocumentTable) and table.headers == cells[0] and table.rows == cells[1:]
    assert [b.text for b in (before, after, other) if isinstance(b, DocumentParagraph)] == [
        "Before grid",
        "After grid",
        "Other report",
    ]
    assert not report.missing and not report.duplicated


@pytest.mark.parametrize(
    "case", ["missing", "open_grid", "shared_block", "already_used", "reversed", "address", "styled_address"]
)
async def test_restores_only_wholly_unreferenced_closed_grid_at_clear_boundary(case: str) -> None:
    lines = [{"x1": x, "y1": 30, "x2": x, "y2": 90} for x in (40, 120, 200)]
    lines += [{"x1": 40, "y1": y, "x2": 200, "y2": y} for y in (30, 50, 70, 90)]
    if case == "open_grid":
        lines.pop(1)
    spans = [span("Before", 40, 20), span("Role", 45, 45), span("Task", 125, 45)]
    spans += [span("Dev", 45, 65), span("Build", 125, 65), span("Ops", 45, 85), span("Run", 125, 85)]
    spans += [span("After", 40, 100)]
    if case in ("address", "styled_address"):
        spans[1:3] = [span("주소", 45, 45), span("가상 라온길 178", 125, 45, 195)]
    for i in range(1, 7):
        spans[i]["color"] = "#775599" if i < 3 and case in ("missing", "styled_address") else "#111111"
    blocks = [{"type": "paragraph", "segments": ["1"]}, {"type": "paragraph", "segments": ["8"]}]
    if case == "shared_block":
        blocks = [{"type": "paragraph", "segments": ["1", "8"]}]
    if case == "already_used":
        blocks.insert(1, {"type": "paragraph", "segments": ["2"]})
    if case == "reversed":
        blocks.reverse()
    blocks.append({"type": "paragraph", "segments": ["9-14"]})
    llm = FakeLlm(plan({"blocks": blocks}))
    following = page(2, [span("More text", 40, y) for y in range(20, 80, 10)])
    document, report = await normalize_with_report(llm, request(page(1, spans, gridLines=lines), following))
    if case == "missing":
        before, table, after, _ = document.blocks
        assert isinstance(before, DocumentParagraph) and before.text == "Before"
        assert isinstance(after, DocumentParagraph) and after.text == "After"
        assert isinstance(table, DocumentTable) and table.headers == ["Role", "Task"]
        assert table.rows == [["Dev", "Build"], ["Ops", "Run"]]
        assert not report.missing
    elif case in ("address", "styled_address"):
        assert isinstance(document.blocks[1], DocumentTable)
        if case == "address":
            assert document.blocks[1].headers == []
        for blind in (False, True):
            chunks = document_chunks(document, blind=blind, applicant_name="")
            assert all("라온길" not in chunk.text for chunk in chunks)
    else:
        assert not any(isinstance(block, DocumentTable) for block in document.blocks)


# 2단 이력서 한 쪽: 왼쪽 단(연락처), 오른쪽 단(소개 두 줄), 아래 쪽 번호
TWO_COLUMNS = page(
    1,
    [
        span("C", 20, 100, 25),
        span("O", 26, 100, 32),
        span("N", 33, 100, 39),
        span("소개 첫 줄의 끝 ", 220, 100),
        span("010-0000-0000", 20, 120),
        span("둘째 줄·마무리", 220, 120),
        span("1", 300, 800),
    ],
)


def test_segments_merge_letter_spaced_heading_and_keep_columns_apart() -> None:
    segments = build_segments([PdfPage.model_validate(TWO_COLUMNS)])
    assert [segment.text for segment in segments] == [
        "CON",
        "소개 첫 줄의 끝 ",
        "010-0000-0000",
        "둘째 줄·마무리",
        "1",
    ]
    assert [segment.id for segment in segments] == [1, 2, 3, 4, 5]


def test_segments_fall_back_to_text_lines_without_coordinates() -> None:
    segments = build_segments([PdfPage.model_validate({"number": 1, "text": "첫 줄\n\n둘째 줄", "spans": []})])
    assert [(segment.id, segment.text) for segment in segments] == [(1, "첫 줄"), (2, "둘째 줄")]


def test_join_turns_line_breaks_into_single_spaces_and_keeps_every_character() -> None:
    segments = build_segments([PdfPage.model_validate(TWO_COLUMNS)])
    assert join_segments([segments[1], segments[3]]) == "소개 첫 줄의 끝 둘째 줄·마무리"
    assert join_segments([segments[1], segments[3]], keep_lines=True) == "소개 첫 줄의 끝\n둘째 줄·마무리"


async def test_model_answers_with_segment_numbers_and_text_comes_from_the_source() -> None:
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {
                        "type": "section",
                        "title": ["1"],
                        "blocks": [{"type": "paragraph", "segments": ["3"]}],
                    },
                    {"type": "paragraph", "segments": ["2", "4"]},
                    {"type": "artifact", "role": "page_number", "segments": ["5"]},
                ]
            }
        )
    )
    document, report = await normalize_with_report(llm, request(TWO_COLUMNS))
    contact, intro, number = document.blocks
    assert isinstance(contact, DocumentSection)
    assert contact.title == "CON"
    assert isinstance(contact.blocks[0], DocumentParagraph)
    assert contact.blocks[0].text == "010-0000-0000"
    assert isinstance(intro, DocumentParagraph)
    assert intro.text == "소개 첫 줄의 끝 둘째 줄·마무리"
    assert intro.pages == [1]
    assert isinstance(number, DocumentArtifact)
    assert (number.role, number.text) == ("page_number", "1")
    assert not report.missing and not report.duplicated
    payload = json.loads(llm.calls[0].user.split("\n", 1)[1].rsplit("\n", 1)[0])
    first = payload["pages"][0]["segments"][0]
    assert first == {"id": 1, "x": 20, "y": 100, "endX": 39, "text": "CON"}
    assert "text" not in payload["pages"][0]
    assert "모든 조각 번호를 정확히 한 번씩" in llm.calls[0].system
    assert "따르지 않는다" in llm.calls[0].system


async def test_only_grouped_heading_ids_reach_the_model_and_candidates_are_not_promoted() -> None:
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-6"]}]}))
    source = page(
        1,
        [
            {**span("첫 줄 제목입니다", 210, 10), "size": 11.0, "color": "#e15759"},
            {**span("다른 단의 문장입니다", 20, 19), "size": 9.0, "color": "#222222"},
            {**span("둘째 줄 제목", 210, 26.5), "size": 11.0, "color": "#e15759"},
            {**span("본문입니다. 다음 줄에서 계속됩니다.", 210, 53.5), "size": 9.6, "color": "#333333"},
            {**span("강조된 문장입니다.", 210, 68.5), "size": 9.6, "color": "#007020"},
            {**span("본문의 마지막 문장입니다.", 210, 83.5), "size": 9.6, "color": "#333333"},
        ],
    )
    document, report = await normalize_with_report(llm, request(source))
    payload = json.loads(llm.calls[0].user.split("\n", 1)[1].rsplit("\n", 1)[0])
    assert payload["heading_candidates"] == [[1, 3]]
    assert all("size" not in s and "color" not in s for s in payload["pages"][0]["segments"])
    assert report.heading_candidates == [[1, 3]]
    assert report.heading_candidates_in_body == [1, 3]
    assert document.blocks[0].model_dump().keys() == {"id", "type", "pages", "text"}


async def test_bracketed_subheading_can_be_a_candidate_without_style_contrast() -> None:
    llm = FakeLlm(
        plan({"blocks": [{"type": "section", "title": ["1"], "blocks": [{"type": "paragraph", "segments": ["2-3"]}]}]})
    )
    source = page(
        1,
        [
            {**span("[배운 점]", 30, 10), "size": 10, "color": "#222222"},
            {**span("일반 문단입니다.", 30, 26), "size": 10, "color": "#222222"},
            {**span("[123]", 30, 42), "size": 10, "color": "#222222"},
        ],
    )
    document, report = await normalize_with_report(llm, request(source))
    assert report.heading_candidates == [[1]]
    assert report.heading_candidates_in_body == []
    assert isinstance(document.blocks[0], DocumentSection)


def _indented_under_heading(marks: list[dict[str, float]]) -> dict[str, Any]:
    return page(
        1,
        [
            {**span("CERTIFICATES", 20, 100), "size": 9.0, "color": "#ffb3b4"},
            {**span("화공기사 (2018.11)", 31, 121.7), "size": 8.6, "color": "#f2f2f2"},
            {**span("위험물산업기사 (2019.05)", 31, 135.3), "size": 8.6, "color": "#f2f2f2"},
        ],
        layout={"width": 600, "height": 800, "markBoxes": marks},
    )


async def test_a_heading_over_bulleted_lines_indented_by_their_bullets_is_a_candidate() -> None:
    # 사이드바 제목 아래 항목 글이 그려진 머리표만큼 들여 쓰여 있어도 같은 단의 내용이다.
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-3"]}]}))
    bullets = [{"x1": 20.2, "y1": y - 5.0, "x2": 23.2, "y2": y - 2.0} for y in (121.7, 135.3)]
    _, report = await normalize_with_report(llm, request(_indented_under_heading(bullets)))
    assert report.heading_candidates == [[1]]


async def test_a_line_over_indented_lines_without_bullets_is_not_a_candidate() -> None:
    # 표 머리 칸 아래 값이나 코드 앞 문장처럼, 머리표 없이 들여 쓴 줄은 다른 단으로 본다.
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-3"]}]}))
    _, report = await normalize_with_report(llm, request(_indented_under_heading([])))
    assert report.heading_candidates == []


async def test_a_two_line_sentence_over_bulleted_lines_is_not_a_candidate() -> None:
    # 목록 앞 두 줄 문장은 제목이 아니다. 들여 쓴 머리표 줄과는 한 줄 제목만 비교한다.
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-4"]}]}))
    source = page(
        1,
        [
            {**span("봉사 활동에서 자판이 지워진 노트북으로 공부하", 48, 100), "size": 10.0, "color": "#1b1b1b"},
            {**span("던 학생이 떠올라 접근성을 목표로 삼았습니다.", 48, 115), "size": 10.0, "color": "#1b1b1b"},
            {**span("키보드만으로 모든 기능 사용", 62, 133), "size": 9.2, "color": "#333333"},
            {**span("화면 낭독기 이름표 점검", 62, 147), "size": 9.2, "color": "#333333"},
        ],
        layout={
            "width": 600,
            "height": 800,
            "markBoxes": [{"x1": 50, "y1": y - 5.0, "x2": 53, "y2": y - 2.0} for y in (133, 147)],
        },
    )
    _, report = await normalize_with_report(llm, request(source))
    assert report.heading_candidates == []


async def test_a_line_followed_only_by_another_column_is_not_a_candidate() -> None:
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-2"]}]}))
    source = page(
        1,
        [
            {**span("CONTACT", 20, 100), "size": 9.0, "color": "#ffb3b4"},
            {**span("본문 단의 문장입니다.", 220, 121.7), "size": 8.6, "color": "#333333"},
        ],
    )
    _, report = await normalize_with_report(llm, request(source))
    assert report.heading_candidates == []


async def test_last_lines_of_a_long_uniform_paragraph_are_not_heading_candidates() -> None:
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-4"]}]}))
    source = page(
        1,
        [
            {**span(text, 30, y), "size": size, "color": "#222222"}
            for text, y, size in [
                ("문단 첫 줄", 10, 10),
                ("문단 둘째 줄", 25, 10),
                ("문단 마지막 줄", 40, 10),
                ("각주", 58, 8),
            ]
        ],
    )
    _, report = await normalize_with_report(llm, request(source))
    assert report.heading_candidates == []


async def test_missing_segments_return_next_to_their_neighbours_and_duplicates_keep_the_first_place() -> None:
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "section", "title": ["1"], "blocks": []},
                    {"type": "paragraph", "segments": ["2", "2", "99"]},
                    {"type": "artifact", "role": "page_number", "segments": ["5"]},
                ]
            }
        )
    )
    document, report = await normalize_with_report(llm, request(TWO_COLUMNS))
    section, intro, restored, number = document.blocks
    assert isinstance(section, DocumentSection) and not section.blocks
    assert isinstance(intro, DocumentParagraph) and intro.text == "소개 첫 줄의 끝"
    assert isinstance(restored, DocumentParagraph) and restored.text == "010-0000-0000 둘째 줄·마무리"
    assert isinstance(number, DocumentArtifact)
    assert report.missing == [3, 4]
    assert report.duplicated == [2]
    assert report.unknown == ["99"]


async def test_segments_missing_right_after_a_title_return_to_the_start_of_that_section() -> None:
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "section", "title": ["1"], "blocks": [{"type": "paragraph", "segments": ["4"]}]},
                    {"type": "paragraph", "segments": ["3", "5"]},
                ]
            }
        )
    )
    document, report = await normalize_with_report(llm, request(TWO_COLUMNS))
    section = document.blocks[0]
    assert isinstance(section, DocumentSection)
    assert [block.text for block in section.blocks if isinstance(block, DocumentParagraph)] == [
        "소개 첫 줄의 끝",
        "둘째 줄·마무리",
    ]
    assert report.missing == [2]


async def test_artifacts_that_hold_content_return_to_paragraphs() -> None:
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "artifact", "role": "header", "segments": ["1"]},
                    {"type": "artifact", "role": "decoration", "segments": ["2"]},
                    {"type": "artifact", "role": "page_number", "segments": ["3"]},
                    {"type": "paragraph", "segments": ["4"]},
                    {"type": "artifact", "role": "page_number", "segments": ["5"]},
                ]
            }
        )
    )
    document, report = await normalize_with_report(llm, request(TWO_COLUMNS))
    kinds = [(block.type, getattr(block, "text", "")) for block in document.blocks]
    assert kinds == [
        ("paragraph", "CON"),
        ("paragraph", "소개 첫 줄의 끝"),
        ("paragraph", "010-0000-0000"),
        ("paragraph", "둘째 줄·마무리"),
        ("artifact", "1"),
    ]
    assert report.demoted_artifacts == [1, 2, 3]


async def test_repeated_headers_stay_visible_while_bullet_marks_and_page_numbers_stay_hidden() -> None:
    # 글이 든 줄은 쪽마다 반복되는 머리글이어도 숨기지 않는다. 숨기는 것은 쪽 번호, 구분선, 머리표 기호뿐이다.
    first = page(1, [span("김민지 이력서", 20, 10), span("•", 20, 100, 25), span("본문 하나", 30, 100)])
    second = page(2, [span("김민지 이력서", 20, 10), span("본문 둘", 30, 100), span("- 2 -", 300, 800)])
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "artifact", "role": "header", "segments": ["1", "4"]},
                    {"type": "artifact", "role": "decoration", "segments": ["2"]},
                    {"type": "paragraph", "segments": ["3", "5"]},
                    {"type": "artifact", "role": "page_number", "segments": ["6"]},
                ]
            }
        )
    )
    document, report = await normalize_with_report(llm, request(first, second))
    hidden = [block.text for block in document.blocks if isinstance(block, DocumentArtifact)]
    shown = [block.text for block in document.blocks if isinstance(block, DocumentParagraph)]
    assert hidden == ["•", "- 2 -"]
    assert shown.count("김민지 이력서") == 2
    assert report.demoted_artifacts == [1, 4]


@pytest.mark.parametrize("role", ["header", "footer"])
async def test_text_repeated_at_the_same_page_edge_is_never_hidden(role: str) -> None:
    # 포트폴리오 바닥글처럼 모든 쪽 같은 자리에 반복되는 글도 원문이라 보이게 둔다.
    edge_y = 10 if role == "header" else 800
    pages = [
        page(i, [span("김주원 | SPACE, DRAWN TO SCALE", 45, edge_y), span(f"본문 {i}", 45, 300)]) for i in (1, 2, 3)
    ]
    edges = [str(1 + 2 * i) for i in range(3)]
    blocks = [{"type": "artifact", "role": role, "segments": edges}]
    blocks += [{"type": "paragraph", "segments": [str(2 + 2 * i)]} for i in range(3)]
    document, report = await normalize_with_report(FakeLlm(plan({"blocks": blocks})), request(*pages))
    assert not [block for block in document.blocks if isinstance(block, DocumentArtifact)]
    assert report.demoted_artifacts == [1, 3, 5]


@pytest.mark.parametrize("role", ["header", "footer"])
async def test_repeated_body_text_is_not_treated_as_a_margin_artifact(role: str) -> None:
    pages = [
        page(i, [span(f"Heading {i}", 20, 20), span("Cut latency by half", 20, 150), span(f"Body {i}", 20, 250)])
        for i in (1, 2)
    ]
    blocks = [
        {"type": "paragraph", "segments": [str(i)]}
        if i not in (2, 5)
        else {"type": "artifact", "role": role, "segments": [str(i)]}
        for i in range(1, 7)
    ]
    document, report = await normalize_with_report(FakeLlm(plan({"blocks": blocks})), request(*pages))
    assert document.blocks[1].type == document.blocks[4].type == "paragraph"
    assert report.demoted_artifacts == [2, 5]


async def test_hidden_real_content_returns_in_reading_order_with_its_headings() -> None:
    # 모델이 사이드바 제목과 항목을 머리글·장식으로 숨겨 문서 끝에 몰아 둔 경우: 원래 읽는 자리에 제목과 함께 되살린다.
    style = {"size": 8.6, "color": "#f2f2f2"}
    heading = {"size": 9.0, "color": "#ffb3b4"}
    sidebar = page(
        1,
        [
            {**span("CONTACT", 20, 100), **heading},
            {**span("010-0000-0000", 20, 122), **style},
            {**span("someone@example.com", 20, 136), **style},
            {**span("SKILLS", 20, 166), **heading},
            {**span("Python", 31, 188), **style},
            {**span("SQL", 31, 202), **style},
            {**span("본문 문장입니다.", 20, 260), "size": 9.6, "color": "#333333"},
        ],
        layout={
            "width": 600,
            "height": 800,
            "markBoxes": [{"x1": 20.2, "y1": y - 5.0, "x2": 23.2, "y2": y - 2.0} for y in (188, 202)],
        },
    )
    hidden = [("page_number", "1"), ("header", "1"), ("decoration", "2"), ("decoration", "3"), ("header", "4")]
    hidden += [("decoration", "5"), ("decoration", "6")]
    blocks = [{"type": "paragraph", "segments": ["7"]}]
    blocks += [{"type": "artifact", "role": role, "segments": [ref]} for role, ref in hidden]
    document, report = await normalize_with_report(FakeLlm(plan({"blocks": blocks})), request(sidebar))

    contact, skills, body = document.blocks
    assert isinstance(contact, DocumentSection) and isinstance(skills, DocumentSection)
    assert contact.title == "CONTACT"
    assert [getattr(block, "text", None) for block in contact.blocks] == ["010-0000-0000", "someone@example.com"]
    assert skills.title == "SKILLS"
    assert [block.type for block in skills.blocks] == ["list"]
    assert [item.text for item in skills.blocks[0].items] == ["Python", "SQL"]  # type: ignore[union-attr]
    assert getattr(body, "text", None) == "본문 문장입니다."
    assert report.demoted_artifacts == [1, 2, 3, 4, 5, 6]
    assert report.missing == []


async def test_hidden_text_already_in_reading_order_stays_where_the_model_put_it() -> None:
    # 마지막 문항 뒤 확인 문구를 바닥글로 숨긴 경우: 문항 안으로 옮기지 않고 그 자리(문서 맨 끝)에서 보인다.
    letter = page(
        1,
        [
            span("1. 지원 동기", 20, 100),
            span("교육 현장의 경험을 살리고 싶습니다.", 20, 130),
            span("위의 내용은 사실과 틀림없음을 확인합니다.", 20, 400),
        ],
    )
    blocks = [
        {"type": "section", "title": ["1"], "blocks": [{"type": "paragraph", "segments": ["2"]}]},
        {"type": "artifact", "role": "footer", "segments": ["3"]},
    ]
    document, report = await normalize_with_report(FakeLlm(plan({"blocks": blocks})), request(letter))
    assert [block.type for block in document.blocks] == ["section", "paragraph"]
    assert report.demoted_artifacts == [3]


async def test_personal_information_is_kept_even_when_the_model_hides_it() -> None:
    form = page(
        1, [span("주소", 20, 100), span("(우: 40653) 인천 부평구 라온길 178", 80, 100), span("나이 30세", 20, 120)]
    )
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "artifact", "role": "decoration", "segments": ["1-2"]},
                    {"type": "artifact", "role": "header", "segments": ["3"]},
                ]
            }
        )
    )
    document, _ = await normalize_with_report(llm, request(form))
    assert [getattr(block, "text", "") for block in document.blocks] == [
        "주소 (우: 40653) 인천 부평구 라온길 178",
        "나이 30세",
    ]
    assert all(block.type == "paragraph" for block in document.blocks)


async def test_fails_when_the_model_drops_most_of_the_document() -> None:
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1"]}]}))
    with pytest.raises(ValueError, match="절반"):
        await normalize_with_report(llm, request(TWO_COLUMNS))


async def test_document_with_only_artifacts_has_no_content() -> None:
    marks = page(1, [span("────────", 20, 100), span("=====", 20, 120), span("1", 300, 800)])
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "artifact", "role": "decoration", "segments": ["1-2"]},
                    {"type": "artifact", "role": "page_number", "segments": ["3"]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="내용이 없다"):
        await normalize_with_report(llm, request(marks))


async def test_meaningful_symbols_marked_as_decoration_stay_visible() -> None:
    form = page(
        1,
        [
            span("기간", 20, 100),
            span("2021.03", 80, 100),
            span("~", 170, 100, 176),
            span("2023.02", 200, 100),
            span("/", 290, 100, 296),
            span("●●●○○", 20, 130),
            span("※", 20, 160, 26),
            span("50%", 120, 160),
            span("자격증", 20, 190),
            span("-", 200, 190, 205),
        ],
    )
    hidden = {"3", "5", "6", "7", "10"}
    blocks = [
        {"type": "artifact", "role": "decoration", "segments": [str(i)]}
        if str(i) in hidden
        else {"type": "paragraph", "segments": [str(i)]}
        for i in range(1, 11)
    ]
    document, report = await normalize_with_report(FakeLlm(plan({"blocks": blocks})), request(form))
    assert all(block.type == "paragraph" for block in document.blocks)
    assert report.demoted_artifacts == [3, 5, 6, 7, 10]


async def test_separator_lines_and_list_bullets_stay_hidden() -> None:
    lines = page(
        1, [span("────────", 20, 100), span("•", 20, 130, 25), span("API 설계", 30, 130), span("=====", 20, 160)]
    )
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "artifact", "role": "decoration", "segments": ["1"]},
                    {"type": "artifact", "role": "decoration", "segments": ["2"]},
                    {"type": "paragraph", "segments": ["3"]},
                    {"type": "artifact", "role": "decoration", "segments": ["4"]},
                ]
            }
        )
    )
    document, report = await normalize_with_report(llm, request(lines))
    assert [block.type for block in document.blocks] == ["artifact", "artifact", "paragraph", "artifact"]
    assert not report.demoted_artifacts


async def test_number_inside_the_body_is_not_hidden_as_a_page_number() -> None:
    body = page(
        1, [span("프로젝트 수", 20, 100), span("3", 20, 120), span("팀 규모 5명", 20, 140), span("1", 300, 800)]
    )
    blocks = [
        {"type": "paragraph", "segments": ["1"]},
        {"type": "artifact", "role": "page_number", "segments": ["2"]},
        {"type": "paragraph", "segments": ["3"]},
        {"type": "artifact", "role": "page_number", "segments": ["4"]},
    ]
    document, report = await normalize_with_report(FakeLlm(plan({"blocks": blocks})), request(body))
    assert [block.type for block in document.blocks] == ["paragraph", "paragraph", "paragraph", "artifact"]
    assert report.demoted_artifacts == [2]


@pytest.mark.parametrize(
    ("kind", "pages", "message"),
    [
        ("image", [1], "원본 PDF 1쪽에 이미지가 있습니다. 내용은 원본에서 확인해 주세요."),
        ("chart", [1, 2], "원본 PDF 1, 2쪽에 차트가 있습니다. 수치는 원본에서 확인해 주세요."),
        ("diagram", [2], "원본 PDF 2쪽에 도식이 있습니다. 구성은 원본에서 확인해 주세요."),
        ("unknown", [1], "원본 PDF 1쪽의 시각 자료를 확인해 주세요."),
    ],
)
async def test_visual_notice_text_is_written_by_code_not_the_model(kind: str, pages: list[int], message: str) -> None:
    notice = {"type": "visual_notice", "pages": pages, "visualType": kind, "message": "매출이 30% 늘어난 차트"}
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1-2"]}, notice]}))
    document, _ = await normalize_with_report(
        llm, request(page(1, [span("본문", 20, 100)]), page(2, [span("둘째", 20, 100)]))
    )
    notices = [block for block in document.blocks if isinstance(block, DocumentVisualNotice)]
    assert [(n.visual_type, n.pages, n.message) for n in notices] == [(kind, pages, message)]


async def test_visual_notice_without_a_real_page_is_dropped() -> None:
    notice = {"type": "visual_notice", "pages": [9], "visualType": "image", "message": "원본 확인"}
    llm = FakeLlm(plan({"blocks": [{"type": "paragraph", "segments": ["1"]}, notice]}))
    document, _ = await normalize_with_report(llm, request(page(1, [span("본문", 20, 100)])))
    assert [block.type for block in document.blocks] == ["paragraph"]


@pytest.mark.parametrize(
    ("change", "missing", "added", "segments"),
    [(lambda text: text[:-1], 1, 0, [2]), (lambda text: text + "X", 0, 1, [])],
)
async def test_output_that_differs_from_the_source_text_fails(
    monkeypatch: pytest.MonkeyPatch, change: Any, missing: int, added: int, segments: list[int]
) -> None:
    from hiremate_ai.agents.analysis.application import normalize as module

    original = module.ensure_visual_notices

    def tamper(document: Any, pages: Any, found: Any) -> Any:
        document = original(document, pages, found)
        last = document.blocks[-1]
        return document.model_copy(
            update={"blocks": [*document.blocks[:-1], last.model_copy(update={"text": change(last.text)})]}
        )

    monkeypatch.setattr(module, "ensure_visual_notices", tamper)
    llm = FakeLlm(
        plan({"blocks": [{"type": "paragraph", "segments": ["1"]}, {"type": "paragraph", "segments": ["2"]}]})
    )
    with pytest.raises(module.NormalizationIntegrityError) as caught:
        await normalize_with_report(llm, request(page(1, [span("가나다", 20, 100), span("라마바", 20, 130)])))
    error = caught.value
    assert isinstance(error, ValueError)
    assert (error.missing_chars, error.added_chars, error.segments) == (missing, added, segments)
    assert "라마바" not in str(error) and "가나다" not in str(error)


async def test_table_cells_join_wrapped_fragments_and_drop_rows_without_values() -> None:
    form = page(
        1,
        [
            span("근무회사", 20, 100),
            span("부서", 200, 100),
            span("모먼트애드", 20, 120),
            span("퍼포먼스마", 200, 120),
            span("케팅팀", 200, 132),
            span("~", 20, 150),
        ],
    )
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {
                        "type": "table",
                        "title": None,
                        "headers": [["1"], ["2"]],
                        "rows": [[["3"], ["4", "5"]], [[], []]],
                    },
                    {"type": "artifact", "role": "decoration", "segments": ["6"]},
                ]
            }
        )
    )
    document, _ = await normalize_with_report(llm, request(form))
    table = document.blocks[0]
    assert isinstance(table, DocumentTable)
    assert table.headers == ["근무회사", "부서"]
    assert table.rows == [["모먼트애드", "퍼포먼스마 케팅팀"]]


async def test_lists_ranges_and_pages_come_from_segments() -> None:
    llm = FakeLlm(
        plan(
            {
                "title": ["1"],
                "blocks": [
                    {"type": "list", "items": [{"segments": ["2"], "items": [{"segments": ["3-4"]}]}]},
                    {
                        "type": "visual_notice",
                        "pages": [1, 9],
                        "visualType": "unknown",
                        "message": "원본 PDF를 확인해 주세요.",
                    },
                    {"type": "artifact", "role": "page_number", "segments": ["5"]},
                ],
            }
        )
    )
    document, _ = await normalize_with_report(llm, request(TWO_COLUMNS))
    assert document.document.title == "CON"
    listing, notice, _ = document.blocks
    assert isinstance(listing, DocumentList)
    assert listing.items[0].text == "소개 첫 줄의 끝"
    assert listing.items[0].items[0].text == "010-0000-0000 둘째 줄·마무리"
    assert isinstance(notice, DocumentVisualNotice)
    assert notice.pages == [1]


async def test_body_put_into_a_section_title_moves_to_a_paragraph() -> None:
    lines = page(1, [span("01 결과", 20, 100), span("성공률 50%", 20, 120), span("→ 88%", 20, 140)])
    llm = FakeLlm(plan({"blocks": [{"type": "section", "title": ["1-3"], "blocks": []}]}))
    document, report = await normalize_with_report(llm, request(lines))
    section = document.blocks[0]
    assert isinstance(section, DocumentSection)
    assert section.title == "01 결과"
    assert isinstance(section.blocks[0], DocumentParagraph)
    assert section.blocks[0].text == "성공률 50% → 88%"
    assert report.split_titles == 1


async def test_rejects_documents_without_text_or_with_failed_pages_before_calling_the_model() -> None:
    llm = FakeLlm()
    with pytest.raises(ValueError, match="텍스트"):
        await normalize_with_report(llm, request({"number": 1, "text": " ", "issues": [], "spans": []}))
    with pytest.raises(ValueError, match="페이지"):
        await normalize_with_report(llm, request({**TWO_COLUMNS, "issues": ["PAGE_EXTRACTION_FAILED"]}))
    assert not llm.calls


async def test_block_ids_are_unique_across_blocks_and_items() -> None:
    llm = FakeLlm(
        plan(
            {
                "blocks": [
                    {"type": "section", "title": ["1"], "blocks": [{"type": "paragraph", "segments": ["2"]}]},
                    {"type": "list", "items": [{"segments": ["3"]}, {"segments": ["4"]}]},
                    {"type": "artifact", "role": "page_number", "segments": ["5"]},
                ]
            }
        )
    )
    document, _ = await normalize_with_report(llm, request(TWO_COLUMNS))
    data = document.model_dump(mode="json", by_alias=True)
    ids: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "id" in value and isinstance(value["id"], str):
                ids.append(value["id"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(data["blocks"])
    assert len(ids) == len(set(ids)) == 6
