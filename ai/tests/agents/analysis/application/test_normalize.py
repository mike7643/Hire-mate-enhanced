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


async def test_repeated_header_and_bullet_marks_stay_artifacts() -> None:
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
    assert [block.type for block in document.blocks] == ["artifact", "artifact", "paragraph", "artifact"]
    assert not report.demoted_artifacts


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
    marks = page(1, [span("•", 20, 100, 25), span("~", 20, 120, 25), span("1", 300, 800)])
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
