"""검증된 PDF 격자로 모델 표를 바꿀 때 표 제목(캡션) 처리."""

from typing import Any

from hiremate_ai.agents.analysis.application.layout_grid import GeometryTable
from hiremate_ai.agents.analysis.application.normalize import DocumentPlan, Segment
from hiremate_ai.agents.analysis.application.plan_tables import apply_geometry_tables

_TEXTS = ["표 1. 개선 전후 비교", "지표", "개선 후", "대기 시간", "4.1초", "중복 매칭", "0건", "다음 문단"]
SEGMENTS = [Segment(i, 1, 50.0, 100.0 + 15 * i, 200.0, text, 10.0, None) for i, text in enumerate(_TEXTS, start=1)]
GRID = GeometryTable(page=1, mode="rectangular", headers=[[2], [3]], rows=[[[4], [5]], [[6], [7]]], marks=[])
EXPECTED_CELLS: dict[str, Any] = {"headers": [["2"], ["3"]], "rows": [[["4"], ["5"]], [["6"], ["7"]]]}


def _plan(*blocks: dict[str, Any]) -> DocumentPlan:
    return DocumentPlan.model_validate({"title": None, "blocks": list(blocks)})


def test_a_replaced_table_keeps_the_caption_the_model_gave_it() -> None:
    # 모델이 칸을 한 줄에 몰아 적은 표: 격자 표로 바꾸되 캡션은 새 표의 제목으로 옮긴다. 빈 표를 남기지 않는다.
    wrong = {"type": "table", "title": ["1"], "headers": [], "rows": [[["2", "3"], ["4", "5"], ["6", "7"]]]}
    plan = _plan({"type": "section", "title": None, "blocks": [wrong, {"type": "paragraph", "segments": ["8"]}]})

    revised = apply_geometry_tables(plan, SEGMENTS, [GRID])

    assert revised == _plan(
        {
            "type": "section",
            "title": None,
            "blocks": [{"type": "table", "title": ["1"], **EXPECTED_CELLS}, {"type": "paragraph", "segments": ["8"]}],
        }
    )


def test_a_caption_stays_put_when_the_old_table_keeps_other_cells() -> None:
    # 옛 표에 격자 밖 칸이 남으면 어느 표의 캡션인지 알 수 없으므로 옮기지 않는다.
    wrong = {"type": "table", "title": ["1"], "headers": [], "rows": [[["2", "3"], ["4", "5"], ["6", "7"], ["8"]]]}
    revised = apply_geometry_tables(_plan(wrong), SEGMENTS, [GRID])
    assert revised == _plan(
        {"type": "table", "title": None, **EXPECTED_CELLS},
        {"type": "table", "title": ["1"], "headers": [], "rows": [[[], [], [], ["8"]]]},
    )
