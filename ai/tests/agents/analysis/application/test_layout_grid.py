from collections.abc import Sequence

import pytest

from hiremate_ai.agents.analysis.application.layout_grid import GeometryTable, detect_tables
from hiremate_ai.agents.analysis.application.normalize import Segment
from hiremate_ai.contract import PdfPage


def _page(
    horizontal: Sequence[tuple[float, float, float]],
    vertical: Sequence[tuple[float, float, float]],
    *,
    number: int = 1,
    fills: Sequence[tuple[float, float, float, float, str]] = (),
) -> PdfPage:
    return PdfPage.model_validate(
        {
            "number": number,
            "text": "",
            "gridLines": [
                *[dict(x1=left, y1=y, x2=right, y2=y) for y, left, right in horizontal],
                *[dict(x1=x, y1=top, x2=x, y2=bottom) for x, top, bottom in vertical],
            ],
            "layout": {
                "width": 600,
                "height": 800,
                "fillRects": [dict(x1=x1, y1=y1, x2=x2, y2=y2, color=color) for x1, y1, x2, y2, color in fills],
                "imageBoxes": [],
            },
        }
    )


def _segment(
    ref: int,
    x: float,
    y: float,
    *,
    page: int = 1,
    end: float | None = None,
    text: str = "field",
    color: str | None = None,
) -> Segment:
    return Segment(ref, page, x, y, x + 16 if end is None else end, text, color=color)


def _dots(left: int, right: int, ys: tuple[int, ...]) -> list[tuple[float, float, float, float, str]]:
    return [(x, y, x + 1, y + 0.5, "#666666") for y in ys for x in [*range(left, right - 1, 3), right - 1]]


def test_complete_unstyled_grid_keeps_every_row_as_data() -> None:
    page = _page([(y, 50, 290) for y in (50, 80, 120)], [(x, 50, 120) for x in (50, 130, 210, 290)])
    segments = [
        _segment(8, 68, 68, text="Address"),
        _segment(7, 148, 68),
        _segment(6, 228, 68),
        _segment(3, 68, 92),
        _segment(4, 148, 92),
        _segment(2, 148, 107),
        _segment(5, 228, 92),
        _segment(9, 305, 100),
    ]

    assert detect_tables([page], list(reversed(segments))) == [
        GeometryTable(1, "rectangular", [], [[[8], [7], [6]], [[3], [4, 2], [5]]], [])
    ]


def test_native_cell_background_distinguishes_header_in_ruled_open_sides() -> None:
    xs, ys = (60, 140, 230, 310), (80, 110, 140, 170, 200, 230)
    fills = [
        *[(a, y, b, y + 0.75, "#555555") for y in ys for a, b in zip(xs, xs[1:], strict=False)],
        *[(x, y + 0.75, x + 0.75, z, "#555555") for x in xs[1:-1] for y, z in zip(ys, ys[1:], strict=False)],
        *[(a, 80.75, b, 110, "#ededed") for a, b in zip(xs, xs[1:], strict=False)],
    ]
    segments = [
        _segment(ref, x + 7, y + 16) for ref, (y, x) in enumerate(((y, x) for y in ys[:-1] for x in xs[:-1]), 1)
    ]

    assert detect_tables([_page([], [], fills=fills)], segments) == [
        GeometryTable(
            1,
            "rectangular",
            [[1], [2], [3]],
            [[[4], [5], [6]], [[7], [8], [9]], [[10], [11], [12]], [[13], [14], [15]]],
            [],
        )
    ]


def test_dotted_row_separators_keep_wrapped_cell_fragments() -> None:
    fills = [(60, 100, 140, 121, "#aaceef"), (140, 100, 230, 121, "#aaceef"), *_dots(60, 230, (121, 147, 171))]
    segments = [
        _segment(1, 70, 113),
        _segment(2, 150, 113),
        _segment(5, 72, 136),
        _segment(3, 152, 130),
        _segment(4, 153, 142),
        _segment(6, 74, 161, text="~"),
        _segment(7, 152, 161, text="/"),
        _segment(8, 70, 184),
    ]

    assert detect_tables([_page([], [], fills=fills)], segments) == [
        GeometryTable(1, "dotted", [[1], [2]], [[[5], [3, 4]]], [6, 7])
    ]


def test_colored_label_bands_split_physical_rows_into_label_value_pairs() -> None:
    horizontal = [(51, 80, 360)]
    vertical = [(147, 52, 105), (280, 52, 78)]
    fills = [
        (80, 52, 147, 78, "#b6d5f4"),
        (213, 52, 280, 78, "#b6d5f4"),
        (80, 78, 147, 105, "#b6d5f4"),
        *_dots(80, 360, (78, 105)),
    ]
    segments = [
        _segment(7, 90, 66),
        _segment(8, 155, 66),
        _segment(3, 220, 66),
        _segment(4, 286, 66),
        _segment(5, 90, 95),
        _segment(2, 155, 95),
        _segment(9, 69, 95, end=71),
    ]

    assert detect_tables([_page(horizontal, vertical, fills=fills)], segments) == [
        GeometryTable(1, "pairs", [], [[[7], [8]], [[3], [4]], [[5], [2]]], [])
    ]


def test_header_only_form_excludes_marks_drawn_in_the_body() -> None:
    fills = [(60, 100, 140, 122, "#aaceef"), (140, 100, 230, 122, "#aaceef"), *_dots(60, 230, (122, 145, 168, 192))]
    segments = [
        _segment(4, 70, 114),
        _segment(5, 152, 114),
        _segment(6, 75, 134, text="~"),
        _segment(7, 75, 155, text="／"),
        _segment(8, 75, 178, text="/"),
    ]

    assert detect_tables([_page([], [], fills=fills)], segments) == [
        GeometryTable(1, "rectangular", [[4], [5]], [], [6, 7, 8])
    ]


def test_adjacent_open_grids_keep_independent_row_boundaries() -> None:
    page = _page(
        [(y, 55, 235) for y in (100, 130, 161)] + [(y, 266, 511) for y in (100, 148, 203)],
        [(130, 100, 161), (343, 100, 203), (422, 100, 203)],
    )
    segments = [
        _segment(7, 70, 120),
        _segment(8, 165, 119),
        _segment(1, 280, 120),
        _segment(2, 365, 127),
        _segment(3, 450, 120),
        _segment(4, 280, 161),
        _segment(5, 365, 174),
        _segment(6, 450, 172),
        _segment(9, 70, 149),
        _segment(10, 165, 148),
        _segment(11, 242, 143, end=251),
    ]

    assert detect_tables([page], segments) == [
        GeometryTable(1, "adjacent-open-grid", [], [[[7], [8]], [[9], [10]]], []),
        GeometryTable(1, "adjacent-open-grid", [], [[[1], [2], [3]], [[4], [5], [6]]], []),
    ]


def test_late_grid_matches_only_the_open_top_row_of_the_next_page() -> None:
    first = _page(
        [(610, 70, 492), *[(y, 115, 492) for y in (580, 650, 686, 722, 758)]],
        [(x, 580, 758) for x in (115, 220, 370, 492)],
    )
    next_page = _page([(67, 70, 492)], [(x, 40, 67) for x in (115, 220, 370)], number=2)
    segments = [
        _segment(1, 80, 674),
        _segment(4, 150, 634),
        _segment(5, 261, 634),
        _segment(6, 405, 634),
        _segment(7, 261, 674),
        _segment(8, 405, 674),
        _segment(9, 261, 709),
        _segment(10, 405, 709),
        _segment(11, 261, 745),
        _segment(12, 405, 745),
        _segment(13, 261, 56, page=2),
        _segment(14, 405, 56, page=2),
        _segment(15, 261, 85, page=2),
    ]

    assert detect_tables([first, next_page], segments) == [
        GeometryTable(
            1,
            "continued-open-grid",
            [[4], [5], [6]],
            [[[], [7], [8]], [[], [9], [10]], [[], [11], [12]], [[], [13], [14]]],
            [],
            continuation_page=2,
        )
    ]


def test_repeated_native_rule_seams_make_an_unstyled_label_value_grid() -> None:
    ys = (92, 117, 142, 167, 192)
    fills = [(a, y, b, y + 0.75, "#aaaaaa") for y in ys for a, b in [(50, 160), (160, 470)]]
    segments = [
        _segment(ref, x, y - 9, text="Address" if ref == 1 else "field")
        for ref, (y, x) in enumerate(((y, x) for y in ys for x in (55, 165)), 1)
    ]

    assert detect_tables([_page([], [], fills=fills)], segments) == [
        GeometryTable(1, "rectangular", [], [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]], [[9], [10]]], [])
    ]


def test_ruled_prompt_boxes_with_paragraphs_are_not_data_grids() -> None:
    page = _page([(y, 50, 530) for y in (80, 200, 330)], [(x, 80, 330) for x in (50, 112, 530)])
    sentence = "A full line of prose, rather than a short field value."
    segments = [
        _segment(1, 65, 142),
        *[_segment(i, 135, 90 + 16 * i, text=sentence, end=475) for i in (2, 3, 4)],
        _segment(5, 65, 260),
        *[_segment(i, 135, 120 + 16 * i, text=sentence, end=475) for i in (6, 7, 8)],
    ]

    assert detect_tables([page], segments) == []


@pytest.mark.parametrize("last_row_present", [True, False])
def test_shared_caps_define_the_full_grid_next_to_a_merged_margin(last_row_present: bool) -> None:
    rows = (100, 130, 160, 193, 233)
    page = _page(
        [(100, 40, 510), (233, 40, 510), *[(y, 120, 510) for y in rows[1:-1]]],
        [(x, 100, 233) for x in (40, 120, 215, 510)],
        fills=[(120, a, 215, b, "#eeeeee") for a, b in zip(rows, rows[1:], strict=False)],
    )
    segments = [
        _segment(1, 65, 180),
        *[
            _segment(ref, x, y + 19)
            for ref, (y, x) in enumerate(((y, x) for y in rows[:-1] for x in (145, 242)), 10)
            if last_row_present or ref < 16
        ],
    ]
    expected = [GeometryTable(1, "rectangular", [], [[[10], [11]], [[12], [13]], [[14], [15]], [[16], [17]]], [])]

    assert detect_tables([page], segments) == (expected if last_row_present else [])


@pytest.mark.parametrize("color, header", [("#55218F", True), ("not a PDF color", False)])
def test_only_valid_native_text_contrast_independently_labels_a_header(color: str, header: bool) -> None:
    page = _page([(y, 50, 240) for y in (80, 110, 141)], [(x, 80, 141) for x in (50, 130, 240)])
    segments = [
        _segment(1, 65, 99, color=color),
        _segment(2, 153, 99, color=color),
        _segment(3, 65, 127, color="#171717"),
        _segment(4, 153, 127, color="#171717"),
    ]
    first, second = [[1], [2]], [[3], [4]]

    assert detect_tables([page], segments) == [
        GeometryTable(1, "rectangular", first if header else [], [second] if header else [first, second], [])
    ]


@pytest.mark.parametrize("connected", [False, True])
def test_floating_glyph_stroke_does_not_act_like_an_attached_partial_divider(connected: bool) -> None:
    page = _page(
        [(y, 50, 251) for y in (80, 110, 141)],
        [*[(x, 80, 141) for x in (50, 130, 251)], (101, 80 if connected else 89, 98)],
    )
    segments = [
        _segment(1, 65, 99),
        _segment(2, 153, 99),
        _segment(3, 65, 127),
        _segment(4, 153, 127),
    ]

    expected = [GeometryTable(1, "rectangular", [], [[[1], [2]], [[3], [4]]], [])]
    assert detect_tables([page], segments) == ([] if connected else expected)
