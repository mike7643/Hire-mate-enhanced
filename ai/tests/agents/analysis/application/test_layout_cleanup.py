"""모델이 겹쳐 넣거나 따로 떼어 놓은 묶음, 겹친 원본 확인 안내를 정리하는 규칙. 모델 요청은 바꾸지 않는다."""

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
            documentType=DocumentType.RESUME,
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
        elif block.type == "visual_notice":
            shaped.append((block.visual_type, block.pages))
        else:
            shaped.append(block.type)
    return shaped


def _p(*refs: str) -> dict[str, Any]:
    return {"type": "paragraph", "segments": list(refs)}


def _s(title: list[str] | None, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"type": "section", "title": title, "blocks": list(blocks)}


def _notice(kind: str, page: int) -> dict[str, Any]:
    return {"type": "visual_notice", "pages": [page], "visual_type": kind, "message": "원본을 확인해 주세요."}


def test_untitled_groups_nested_one_inside_the_next_come_out_as_siblings() -> None:
    lines = [(_BODY, 10.0)] * 3
    nested = _s(None, _p("1"), _s(None, _p("2"), _s(None, _p("3"))))
    assert _revise(lines, nested) == [(None, [["1"]]), (None, [["2"]]), (None, [["3"]])]


def test_an_untitled_group_inside_a_titled_section_stays_inside() -> None:
    lines = [("경력", 12.0), (_BODY, 10.0), (_BODY, 10.0)]
    titled = _s(["1"], _p("2"), _s(None, _p("3")))
    assert _revise(lines, titled) == [(["1"], [["2"], (None, [["3"]])])]


def test_an_empty_titled_section_takes_the_untitled_group_right_after_it() -> None:
    lines = [("프로젝트", 12.0), (_BODY, 10.0), (_BODY, 10.0)]
    assert _revise(lines, _s(["1"]), _s(None, _p("2-3"))) == [(["1"], [["2-3"]])]


def test_notices_for_the_same_page_collapse_into_the_first_one_with_the_known_kind() -> None:
    lines = [(_BODY, 10.0), (_BODY, 10.0)]
    shaped = _revise(lines, _p("1"), _notice("unknown", 1), _s(None, _p("2"), _notice("image", 1), _notice("image", 1)))
    assert shaped == [["1"], ("image", [1]), (None, [["2"]])]
