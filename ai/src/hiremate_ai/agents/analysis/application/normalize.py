"""원문 글 조각에 구조를 입혀 정제 문서를 만든다.

모델은 조각 번호로 구조(제목, 문단, 목록, 표, 잡음)만 정하고 글은 쓰지 않는다. 글은 코드가 파서의 원문
조각을 이어 붙여 만든다. 원문과 달라질 수 있는 것은 공백과 줄바꿈뿐이다. 모델이 빠뜨린 조각은 앞 조각
가까이에 문단으로 되살리되, 닫힌 격자와 위치가 명확한 누락 표는 표로 되살린다. 두 번 쓴 조각은 처음 자리만 남긴다.
"""

import json
import logging
import math
import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from hiremate_ai.agents.analysis.application.grid import closed_grids
from hiremate_ai.agents.analysis.application.layout_grid import detect_tables
from hiremate_ai.agents.analysis.application.layout_structure import (
    bulleted_lines,
    ensure_visual_notices,
    revise_structure,
    visual_notice_message,
)
from hiremate_ai.agents.analysis.application.normalization_prompt import system_prompt
from hiremate_ai.config import settings
from hiremate_ai.contract import (
    DocumentArtifact,
    DocumentBlock,
    DocumentList,
    DocumentListItem,
    DocumentMetadata,
    DocumentNormalizationRequest,
    DocumentParagraph,
    DocumentSection,
    DocumentTable,
    NormalizedDocument,
    PdfPage,
)
from hiremate_ai.llm.chat import LlmFailureReason, LlmOutputError, model_call_budget
from hiremate_ai.llm.client import StructuredLlmClient
from hiremate_ai.llm.prompt import data_block

logger = logging.getLogger(__name__)

_SPACE = re.compile(r"\s+")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
# 같은 줄로 보는 세로 좌표 차이와, 한 글자씩 벌려 쓴 제목을 한 조각으로 합치는 가로 간격.
_SAME_LINE = 1.0
# 제목 후보 아래 같은 단으로 보는 들여쓰기 한도(pt). 머리표 들여쓰기는 보통 10~20pt다.
_HEADING_INDENT = 24.0
_LETTER_GAP = 12.0
_MARGIN_GAP = 12.0
# 모델이 이보다 많은 조각을 빠뜨리면 구조를 믿을 수 없어 실패로 둔다. 실패한 서류는 PDF 추출 글로 읽는다.
_MAX_MISSING_RATIO = 0.5
# 쪽 안에서 모델이 이만큼 넘게 연달아 빠뜨린 조각은 그 구간만 다시 묻는다(대개 답을 중간에 멈춘 경우). 더 짧은 구간은
# 조립이 읽는 자리에 문단으로 되살린다.
_RETRY_RUN = 8
# 조각 번호는 ASCII이며, 로컬 JSON 문법 생성기에서 \d 대신 명시적 범위를 쓴다.
_REF = r"^[0-9]+(-[0-9]+)?$"
_BRACKET_HEADING = re.compile(r"\[([^\[\]\n]{1,80})\]")
# 쪽 번호로 받는 모양: 1, - 1 -, 1/3, p.2, 3쪽
_PAGE_NUMBER = re.compile(r"(?i)[\W_]*(?:page|p\.?)?\s*\d{1,3}\s*(?:쪽|페이지)?(?:[\W_]+\d{1,3})?[\W_]*")
# 장식으로 숨겨도 되는 기호는 둘뿐이다. 같은 선 문자가 세 개 이상 이어진 구분선과, 줄 첫머리에서 바로 오른쪽에 글이
# 붙은 목록 머리표 한 글자. 물결표, 빗금, 백분율, 평점 표시처럼 뜻이 있을 수 있는 기호는 숨기지 않는다.
_LINE_CHARS = frozenset("-=_*―—–")
_BULLETS = frozenset("•◦▪▫■□●○◆◇▶►▷‣⁃∙◼◻-*")
_BULLET_GAP = 15.0
# 글 뒤에 놓이면 선택 표시(병역 ■ 군필 □ 미필)로 보는 기호. 이런 기호가 글 뒤에 있는 줄은 기호를 하나도 숨기지 않는다.
_OPTION_MARKS = frozenset("■□●○◆◇◼◻▪▫")
# 쪽 번호와 같은 줄에 글이 있으면, 그 글과 쪽 글 영역 너비의 1/3 이상 떨어져 있을 때만 쪽 번호로 본다. 바닥글
# "이름 | 제목 ... 3"은 70% 넘게 떨어지고, 마지막 줄의 점수("TOEIC 2023.03 760")는 5% 안팎이다.
_PAGE_NUMBER_APART = 1 / 3
# 사이드바로 보는 채움 영역: 쪽 높이의 90% 이상, 쪽 너비의 45% 이하, 가장자리에서 1pt 안, 안팎에 각각 3줄 이상.
_PANEL_HEIGHT = 0.9
_PANEL_WIDTH = 0.45
_PANEL_EDGE = 1.0
_PANEL_LINES = 3


def _is_separator(text: str) -> bool:
    """같은 선 문자 세 개 이상, 또는 상자 그리기 문자만으로 된 구분선."""
    if len(text) < 3:
        return False
    return all("\u2500" <= char <= "\u257f" for char in text) or (len(set(text)) == 1 and text[0] in _LINE_CHARS)


class NormalizationLimitError(ValueError):
    """서류가 정제 한도(쪽 수)를 넘는다. 다시 해도 같으므로 정제 실패로 두고 PDF 추출 글을 쓴다."""


class NormalizationCoverageError(ValueError):
    """다시 물은 뒤에도 모델 답이 원문 조각의 반 넘게 빠뜨렸다.

    같은 입력엔 같은 답이므로 정제 실패로 두고 PDF 추출 글을 쓴다.
    """


class NormalizationIntegrityError(ValueError):
    """정제 결과의 글자가 원문 조각과 다르다. 다시 해도 같으므로 정제 실패로 두고 PDF 추출 글을 쓴다.

    지원자 글은 담지 않는다. 글자 수와, 결과에서 찾을 수 없는 조각 번호만 남긴다.
    """

    def __init__(self, *, missing_chars: int, added_chars: int, segments: list[int]) -> None:
        self.missing_chars = missing_chars
        self.added_chars = added_chars
        self.segments = segments
        shown = ",".join(str(number) for number in segments[:50])
        super().__init__(
            f"정제 결과가 원문 글자와 다르다 missing={missing_chars} added={added_chars} segments=[{shown}]"
        )


@dataclass(frozen=True)
class Segment:
    """파서가 뽑은 원문 조각. text는 원문 그대로다."""

    id: int
    page: int
    x: float
    y: float
    end_x: float
    text: str
    size: float | None = None
    color: str | None = None


def _has_text(segment: Segment) -> bool:
    return any(char.isalnum() for char in segment.text)


@dataclass
class NormalizationReport:
    """모델 구조를 원문 조각과 맞추며 고친 내용. 실험과 로그에서 구조 품질을 본다."""

    segments: int = 0
    missing: list[int] = field(default_factory=list)
    duplicated: list[int] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    split_titles: int = 0
    # 잡음이라고 했지만 코드 확인을 통과하지 못해 문단으로 되돌린 조각. 잡음은 화면과 분석에서 빠지므로
    # 내용을 숨기지 못하게 한다.
    demoted_artifacts: list[int] = field(default_factory=list)
    heading_candidates: list[list[int]] = field(default_factory=list)
    heading_candidates_in_body: list[int] = field(default_factory=list)


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, alias_generator=lambda name: _camel(name))


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


Refs = list[Annotated[str, Field(pattern=_REF)]]


class ItemPlan(PlanModel):
    segments: Refs
    items: list["ItemPlan"] = Field(default_factory=list)


class ParagraphPlan(PlanModel):
    type: Literal["paragraph"]
    segments: Refs
    keep_lines: bool = False


class ListPlan(PlanModel):
    type: Literal["list"]
    items: list[ItemPlan]


class TablePlan(PlanModel):
    type: Literal["table"]
    title: Refs | None = None
    headers: list[Refs]
    rows: list[list[Refs]]


class VisualNoticePlan(PlanModel):
    type: Literal["visual_notice"]
    pages: list[int]
    visual_type: Literal["image", "chart", "diagram", "unknown"]
    message: str


class ArtifactPlan(PlanModel):
    type: Literal["artifact"]
    role: Literal["page_number", "header", "footer", "decoration"]
    segments: Refs


class SectionPlan(PlanModel):
    type: Literal["section"]
    title: Refs | None
    blocks: list["BlockPlan"]


BlockPlan = Annotated[
    SectionPlan | ParagraphPlan | ListPlan | TablePlan | VisualNoticePlan | ArtifactPlan,
    Field(discriminator="type"),
]
SectionPlan.model_rebuild()


class DocumentPlan(PlanModel):
    title: Refs | None
    blocks: list[BlockPlan]


def _plan_refs(node: DocumentPlan | BlockPlan | ItemPlan) -> Iterator[str]:
    if isinstance(node, (DocumentPlan, SectionPlan, TablePlan)):
        yield from node.title or []
    if isinstance(node, (DocumentPlan, SectionPlan)):
        for child in node.blocks:
            yield from _plan_refs(child)
    if isinstance(node, (ParagraphPlan, ArtifactPlan, ItemPlan)):
        yield from node.segments
    if isinstance(node, (ListPlan, ItemPlan)):
        for item in node.items:
            yield from _plan_refs(item)
    if isinstance(node, TablePlan):
        for row in (node.headers, *node.rows):
            for cell in row:
                yield from cell


def _ref_ids(ref: str, count: int) -> range:
    start, _, end = ref.partition("-")
    try:
        first, last = int(start), int(end or start)
    except ValueError:
        return range(0)
    return range(first, last + 1) if 1 <= first <= last <= count else range(0)


def build_segments(pages: list[PdfPage]) -> list[Segment]:
    """페이지 순서, 파서 순서대로 조각에 1부터 번호를 붙인다.

    같은 줄에서 한두 글자씩 벌려 쓴 조각(C O N T A C T, 자간으로 생긴 AT)은 하나로 합친다. 좌표가 없는 페이지는
    줄마다 조각이다.
    """
    result: list[Segment] = []
    for page in pages:
        pieces: list[tuple[float, float, float, str, float | None, str | None]] = []
        if page.spans:
            last_single = False
            for span in page.spans:
                text = unicodedata.normalize("NFC", span.text)
                if not text.strip():
                    continue
                single = len(text.strip()) <= 2 and text.strip().isalnum()
                if pieces and single and last_single:
                    x, y, end_x, previous, size, color = pieces[-1]
                    if abs(span.y - y) < _SAME_LINE and 0 <= span.x - end_x < _LETTER_GAP:
                        pieces[-1] = (
                            x,
                            y,
                            span.end_x,
                            previous.strip() + text.strip(),
                            size if size == span.size else None,
                            color if color == span.color else None,
                        )
                        continue
                pieces.append((span.x, span.y, span.end_x, text, span.size, span.color))
                last_single = single
        else:
            for row, line in enumerate(unicodedata.normalize("NFC", page.text).splitlines()):
                if line.strip():
                    pieces.append((0.0, float(row), 0.0, line, None, None))
        if page.spans:
            pieces = _column_order(page, pieces)
        for x, y, end_x, text, size, color in pieces:
            result.append(Segment(len(result) + 1, page.number, x, y, end_x, text, size, color))
    return result


_Piece = tuple[float, float, float, str, float | None, str | None]


def _column_order(page: PdfPage, pieces: list[_Piece]) -> list[_Piece]:
    """쪽 높이를 거의 다 덮고 한쪽 가장자리에 붙은 채움 영역(사이드바)이 있으면, 그 안의 줄을 따로 한 단으로 읽는다.

    단은 왼쪽부터 읽는다. 세로 위치 순서로 두 단의 줄이 번갈아 나오면 모델이 다른 단의 줄을 한 블록에 섞기 때문이다.
    영역이 하나가 아니거나 안팎 줄이 모자라면 파서 순서를 그대로 둔다.
    """
    layout = page.layout
    if layout is None:
        return pieces
    panels = [
        rect
        for rect in layout.fill_rects
        if rect.y2 - rect.y1 >= _PANEL_HEIGHT * layout.height
        and rect.x2 - rect.x1 <= _PANEL_WIDTH * layout.width
        and (rect.x1 <= _PANEL_EDGE or rect.x2 >= layout.width - _PANEL_EDGE)
    ]
    if len(panels) != 1:
        return pieces
    panel = panels[0]
    inside = [p for p in pieces if panel.x1 - 0.5 <= p[0] < p[2] <= panel.x2 + 0.5]
    outside = [p for p in pieces if not panel.x1 - 0.5 <= p[0] < p[2] <= panel.x2 + 0.5]
    if len(inside) < _PANEL_LINES or len(outside) < _PANEL_LINES:
        return pieces
    return [*inside, *outside] if panel.x1 <= _PANEL_EDGE else [*outside, *inside]


def _heading_candidates(by_page: dict[int, list[Segment]], bulleted: AbstractSet[int] = frozenset()) -> list[list[int]]:
    """같은 단에서 다음 줄과 글자 모양이 대비되는 줄을 찾는다. 조각 번호는 바꾸지 않는다.

    bulleted는 머리표 그림으로 시작하는 줄이다. 그런 줄은 머리표만큼 들여 써 있어도 같은 단의 다음 줄로 본다.
    """
    candidates: list[list[int]] = []
    for segments in by_page.values():
        lines = sorted(segments, key=lambda segment: (segment.y, segment.x))
        reserved: set[int] = set()
        for index, head in enumerate(lines):
            if head.id in reserved:
                continue
            bracket = _BRACKET_HEADING.fullmatch(head.text.strip())
            if bracket and any(char.isalpha() for char in bracket[1]):
                candidates.append([head.id])
                continue
            if head.size is None or head.size < 7 or not any(char.isalpha() for char in head.text):
                continue
            # 긴 본문을 두 줄짜리 제목 후보로 잘못 잘라 내지 않는다.
            previous = _preceding_aligned_line(lines, index, head, 1.7 * head.size)
            if previous is not None and (previous.size, previous.color) == (head.size, head.color):
                continue
            group = [head.id]
            last_y = head.y
            for i in range(index + 1, len(lines)):
                following = lines[i]
                gap = following.y - last_y
                if gap > 4 * head.size:
                    break
                offset = following.x - head.x
                aligned = abs(offset) <= 8
                indented_item = following.id in bulleted and 0 < offset <= _HEADING_INDENT and len(group) == 1
                if gap < _SAME_LINE or not (aligned or indented_item) or following.size is None:
                    continue
                if aligned and gap <= 1.7 * head.size and (head.size, head.color) == (following.size, following.color):
                    group.append(following.id)
                    last_y = following.y
                    if len(group) > 2:
                        break
                    continue
                larger = head.size - following.size >= 0.7 and head.size >= 1.08 * following.size
                different_color = (
                    head.color is not None
                    and following.color is not None
                    and head.color != following.color
                    and head.size >= following.size
                    and gap >= 1.8 * head.size
                )
                if larger or different_color:
                    candidates.append(group)
                    reserved.update(group)
                break
    return sorted(candidates, key=lambda group: group[0])


def _preceding_aligned_line(lines: list[Segment], index: int, head: Segment, max_gap: float) -> Segment | None:
    for i in range(index - 1, -1, -1):
        previous = lines[i]
        gap = head.y - previous.y
        if gap > max_gap:
            break
        if gap >= _SAME_LINE and abs(previous.x - head.x) <= 8:
            return previous
    return None


def join_segments(segments: list[Segment], *, keep_lines: bool = False) -> str:
    """조각을 이어 붙인다. 같은 줄은 공백 하나로, 줄이 바뀌면 공백 하나(keep_lines면 줄바꿈)로 잇는다."""
    parts: list[str] = []
    previous: Segment | None = None
    for segment in segments:
        if previous is not None:
            same_line = previous.page == segment.page and abs(previous.y - segment.y) < _SAME_LINE
            parts.append("\n" if keep_lines and not same_line else " ")
        parts.append(segment.text)
        previous = segment
    text = "".join(parts)
    if keep_lines:
        return _TRAILING.sub("", text).strip("\n")
    return _SPACE.sub(" ", text).strip()


async def normalize_document(llm: StructuredLlmClient, request: DocumentNormalizationRequest) -> NormalizedDocument:
    document, report = await normalize_with_report(llm, request)
    if report.heading_candidates_in_body:
        logger.info(
            "제목 후보가 문단이나 목록에 남았다 document=%s count=%d",
            request.document_id,
            len(report.heading_candidates_in_body),
        )
    if report.missing or report.duplicated or report.unknown or report.split_titles or report.demoted_artifacts:
        logger.warning(
            "정제 구조를 원문 조각에 맞춰 고쳤다 document=%s missing=%d duplicated=%d unknown=%d split_titles=%d"
            " demoted_artifacts=%d",
            request.document_id,
            len(report.missing),
            len(report.duplicated),
            len(report.unknown),
            report.split_titles,
            len(report.demoted_artifacts),
        )
    return document


async def normalize_with_report(
    llm: StructuredLlmClient, request: DocumentNormalizationRequest
) -> tuple[NormalizedDocument, NormalizationReport]:
    if not request.pages or [page.number for page in request.pages] != list(range(1, len(request.pages) + 1)):
        raise ValueError("정제할 페이지 정보가 없거나 순서가 올바르지 않다")
    if any("PAGE_EXTRACTION_FAILED" in page.issues for page in request.pages):
        raise ValueError("페이지 추출에 실패했다")
    if not any(page.text.strip() for page in request.pages):
        raise ValueError("추출된 텍스트가 없다")
    if len(request.pages) > settings.normalization_max_pages:
        raise NormalizationLimitError(
            f"정제 쪽 수 한도({settings.normalization_max_pages}쪽)를 넘는 {len(request.pages)}쪽 서류다"
        )
    segments = build_segments(request.pages)
    if not segments:
        raise ValueError("추출된 텍스트가 없다")
    page_count = len(request.pages)
    by_page: dict[int, list[Segment]] = {}
    for segment in segments:
        by_page.setdefault(segment.page, []).append(segment)
    heading_candidates = _heading_candidates(by_page, bulleted_lines(request.pages, by_page))
    payload = {
        "document_type": request.document_type.value.lower(),
        "page_count": page_count,
        "segment_count": len(segments),
        "heading_candidates": heading_candidates,
        "pages": [
            {
                "page": page.number,
                "parser_issues": page.issues,
                "image_objects": page.image_count,
                "form_objects": page.form_count,
                "segments": [
                    {
                        "id": segment.id,
                        "x": round(segment.x, 1),
                        "y": round(segment.y, 1),
                        "endX": round(segment.end_x, 1),
                        "text": segment.text.strip(),
                    }
                    for segment in by_page.get(page.number, [])
                ],
            }
            for page in request.pages
        ],
    }
    tables = detect_tables(request.pages, segments)
    # 선과 칸으로 찾은 표는 모델 답과 상관없이 코드가 세운다. 모델이 그 조각을 빠뜨려도 다시 묻지 않는다.
    drawn = {n for table in tables for row in [table.headers, *table.rows] for cell in row for n in cell}
    if settings.llm_profile == "local":
        with model_call_budget(settings.normalization_max_calls):
            plan = await _ask_local(llm, payload, by_page, drawn)
    else:
        with model_call_budget(settings.llm_agent_max_calls):
            plan = await _ask(llm, "application.normalize", payload)
    if tables:
        from hiremate_ai.agents.analysis.application.plan_tables import apply_geometry_tables

        plan = apply_geometry_tables(plan, segments, tables)
    plan, released = release_false_artifacts(plan, segments, page_count)
    plan = revise_structure(plan, segments, request, released=released)
    document, report = assemble(plan, segments, request, page_count, heading_candidates=heading_candidates)
    report.demoted_artifacts = sorted({*report.demoted_artifacts, *(n for group in released for n in group)})
    document = ensure_visual_notices(document, request.pages, segments)
    verify_source_text(document, segments)
    return document, report


def release_false_artifacts(
    plan: DocumentPlan, segments: list[Segment], page_count: int
) -> tuple[DocumentPlan, list[list[int]]]:
    """숨김 규칙에 맞지 않는 artifact 블록을 계획에서 빼고 그 조각 묶음을 돌려준다.

    모델이 제목이나 항목을 머리글·장식으로 잘못 숨겨 읽는 순서와 다른 자리(대개 문서 끝)에 두었으면, 그 자리에
    되살리는 대신 읽는 순서 자리에 구조와 함께 다시 넣도록 revise_structure에 넘긴다. 이미 읽는 순서 자리에 있는
    잘못된 숨김은 조립이 그 자리에서 문단으로 되살린다. 내용 블록이 쓴 조각과 규칙에 맞는 숨김은 그대로 둔다.
    """
    count = len(segments)
    builder = _Builder({segment.id: segment for segment in segments}, page_count)
    content = {n for ref in plan.title or [] for n in _ref_ids(ref, count)}
    # 계획을 앞에서부터 읽은 순서: 내용 조각 번호 묶음, 또는 artifact 블록.
    sequence: list[list[int] | ArtifactPlan] = [sorted(content)]

    def visit(blocks: list[BlockPlan]) -> None:
        for block in blocks:
            if isinstance(block, ArtifactPlan):
                sequence.append(block)
            elif isinstance(block, SectionPlan):
                sequence.append([n for ref in block.title or [] for n in _ref_ids(ref, count)])
                visit(block.blocks)
            else:
                sequence.append([n for ref in _plan_refs(block) for n in _ref_ids(ref, count)])

    visit(plan.blocks)
    content.update(n for entry in sequence if isinstance(entry, list) for n in entry)
    seen: set[int] = set()
    released: list[list[int]] = []
    removed: set[int] = set()
    for position, artifact in enumerate(sequence):
        if not isinstance(artifact, ArtifactPlan):
            continue
        fresh = [n for ref in artifact.segments for n in _ref_ids(ref, count) if n not in content and n not in seen]
        seen.update(fresh)
        before = max((n for entry in sequence[:position] if isinstance(entry, list) for n in entry), default=0)
        after = min(
            (n for entry in sequence[position + 1 :] if isinstance(entry, list) for n in entry), default=count + 1
        )
        in_order = all(before < n < after for n in fresh)
        if fresh and not in_order and not builder._is_artifact(artifact.role, fresh):
            # 쪽마다 반복되는 머리글처럼 떨어진 조각을 한 블록에 모았으면, 이어지는 번호끼리 나눠
            # 각자 읽는 자리로 보낸다.
            runs: list[list[int]] = []
            for number in sorted(fresh):
                if (
                    runs
                    and runs[-1][-1] == number - 1
                    and builder._segments[number - 1].page == builder._segments[number].page
                ):
                    runs[-1].append(number)
                else:
                    runs.append([number])
            released.extend(runs)
            removed.add(id(artifact))
        elif not fresh and any(
            n in seen for ref in artifact.segments for n in _ref_ids(ref, count) if n not in content
        ):
            # 앞의 잘못된 숨김이 이미 내놓은 조각을 다시 숨기려는 블록이다.
            removed.add(id(artifact))
    if not removed:
        return plan, []

    def strip(blocks: list[BlockPlan]) -> list[BlockPlan]:
        kept: list[BlockPlan] = []
        for block in blocks:
            if id(block) in removed:
                continue
            if isinstance(block, SectionPlan):
                block = block.model_copy(update={"blocks": strip(block.blocks)})
            kept.append(block)
        return kept

    return plan.model_copy(update={"blocks": strip(plan.blocks)}), sorted(released)


def _compact(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFC", text) if not char.isspace())


def _output_strings(document: NormalizedDocument) -> Iterator[str]:
    """결과에 든 원문 글. 시각 자료 안내는 코드가 쓴 문장이라 뺀다."""

    def items(entries: list[DocumentListItem]) -> Iterator[str]:
        for item in entries:
            yield item.text
            yield from items(item.items)

    def walk(blocks: list[DocumentBlock]) -> Iterator[str]:
        for block in blocks:
            if isinstance(block, DocumentSection):
                yield block.title or ""
                yield from walk(block.blocks)
            elif isinstance(block, (DocumentParagraph, DocumentArtifact)):
                yield block.text
            elif isinstance(block, DocumentList):
                yield from items(block.items)
            elif isinstance(block, DocumentTable):
                yield block.title or ""
                yield from block.headers
                for row in block.rows:
                    yield from row

    yield document.document.title or ""
    yield from walk(document.blocks)


def verify_source_text(document: NormalizedDocument, segments: list[Segment]) -> None:
    """결과의 글자가 원문 조각의 글자와 하나도 빠짐없이, 더함 없이 같은지 확인한다. 공백과 줄바꿈 차이는 보지 않는다."""
    source = Counter(char for segment in segments for char in _compact(segment.text))
    texts = [_compact(text) for text in _output_strings(document)]
    output = Counter(char for text in texts for char in text)
    if source == output:
        return
    joined = "".join(texts)
    lost = [segment.id for segment in segments if (text := _compact(segment.text)) and text not in joined]
    raise NormalizationIntegrityError(
        missing_chars=sum((source - output).values()), added_chars=sum((output - source).values()), segments=lost
    )


async def _ask(llm: StructuredLlmClient, agent: str, payload: dict[str, Any]) -> DocumentPlan:
    return await llm.structured(
        DocumentPlan,
        agent=agent,
        system=system_prompt(),
        user=data_block("pdf_segments", json.dumps(payload, ensure_ascii=False)),
    )


async def _ask_local(
    llm: StructuredLlmClient,
    payload: dict[str, Any],
    by_page: dict[int, list[Segment]],
    drawn: set[int] | None = None,
) -> DocumentPlan:
    """로컬 모델에 묻는다. 길면 쪽 묶음으로 나눠 묻고, 답이 빠뜨리거나 실패한 쪽은 쪽마다 다시 묻는다.

    한도 안의 서류는 서류 전체를 한 번에 묻는다(요청이 이전과 같다). 여러 쪽 서류에서 그 호출이 실패하면 서류를 버리지
    않고 쪽마다 다시 묻는다. 한 쪽짜리 서류의 실패는 그대로 올려 작업을 다시 시도하게 한다.
    """
    parts = _page_parts(payload)
    if len(parts) == 1:
        try:
            plan = await _ask(llm, "application.normalize", payload)
        except (LlmOutputError, httpx.HTTPError, TimeoutError) as error:
            if payload["page_count"] == 1 or (
                isinstance(error, LlmOutputError) and error.reason == LlmFailureReason.BUDGET
            ):
                raise
            logger.warning(
                "정제 호출 실패, 쪽마다 다시 묻는다 pages=%d reason=%s", payload["page_count"], _reason(error)
            )
            plan = DocumentPlan(title=None, blocks=[])
    else:
        plans: list[DocumentPlan] = []
        for pages in parts:
            allowed = {segment.id for page in pages for segment in by_page.get(page["page"], [])}
            first, last = pages[0]["page"], pages[-1]["page"]
            try:
                part = await _ask(llm, f"application.normalize.part_{first}_{last}", _scoped(payload, pages, allowed))
            except (LlmOutputError, httpx.HTTPError, TimeoutError) as error:
                if isinstance(error, LlmOutputError) and error.reason == LlmFailureReason.BUDGET:
                    break
                logger.warning("정제 쪽 묶음 호출 실패 pages=%d-%d reason=%s", first, last, _reason(error))
                continue
            if not _within(part, allowed, payload["segment_count"]):
                logger.warning("정제 쪽 묶음 답의 조각 범위가 올바르지 않다 pages=%d-%d", first, last)
                continue
            plans.append(part)
        plan = _merge_parts(plans)
    return await _retry_missing(llm, plan, payload, by_page, drawn or set())


def _reason(error: Exception) -> str:
    return str(error.reason) if isinstance(error, LlmOutputError) else type(error).__name__


def _page_parts(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """요청 글자 수가 한도 안이면 전체 한 묶음, 넘으면 연속한 쪽을 비슷한 크기의 묶음으로 나눈다.

    묶음 수는 한도로 나눈 올림이고, 묶음마다 평균 크기를 넘기 전에 끊는다. 한도를 꽉 채운 큰 묶음과 작은 묶음으로
    나누면 큰 묶음이 다시 모델이 멈추는 길이에 가까워진다. 어떤 묶음도 한도를 넘지 않는다
    (한 쪽이 한도보다 크면 그 쪽만 한 묶음이다).
    """
    limit = settings.normalization_part_chars
    if len(json.dumps(payload, ensure_ascii=False)) <= limit:
        return [payload["pages"]]
    lengths = [len(json.dumps(page, ensure_ascii=False)) for page in payload["pages"]]
    target = sum(lengths) / math.ceil(sum(lengths) / limit)
    parts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for page, length in zip(payload["pages"], lengths, strict=True):
        if current and (size + length > limit or size + length / 2 > target):
            parts.append(current)
            current, size = [], 0
        current.append(page)
        size += length
    if current:
        parts.append(current)
    return parts


def _scoped(payload: dict[str, Any], pages: list[dict[str, Any]], allowed: set[int]) -> dict[str, Any]:
    """일부 쪽만 묻는 요청. 조각 번호는 서류 전체 번호를 그대로 쓴다."""
    return {
        **payload,
        "page_count": len(pages),
        "segment_count": len(allowed),
        "heading_candidates": [group for group in payload["heading_candidates"] if allowed.issuperset(group)],
        "pages": pages,
    }


def _within(plan: DocumentPlan, allowed: set[int], count: int) -> bool:
    refs = [_ref_ids(ref, count) for ref in _plan_refs(plan)]
    return bool(refs) and all(ids and allowed.issuperset(ids) for ids in refs)


def _merge_parts(plans: list[DocumentPlan]) -> DocumentPlan:
    """쪽 묶음 답을 읽는 순서대로 잇는다.

    다음 묶음이 제목 없는 글로 시작하면 앞 묶음 마지막 섹션이 쪽을 넘어 이어진 것이므로 그 섹션 끝에 붙인다. 묶음에
    제목이 있으면 그 제목으로 섹션을 연다. 문서 제목은 첫 묶음의 제목이다.
    """
    if not plans:
        return DocumentPlan(title=None, blocks=[])
    blocks: list[BlockPlan] = list(plans[0].blocks)
    for part in plans[1:]:
        if part.title:
            blocks.append(SectionPlan(type="section", title=part.title, blocks=list(part.blocks)))
            continue
        rest = list(part.blocks)
        lead: list[BlockPlan] = []
        while rest and not (isinstance(rest[0], SectionPlan) and rest[0].title):
            head = rest.pop(0)
            lead.extend(head.blocks if isinstance(head, SectionPlan) else [head])
        blocks = [*_continue_last(blocks, lead), *rest]
    return DocumentPlan(title=plans[0].title, blocks=blocks)


def _continue_last(blocks: list[BlockPlan], lead: list[BlockPlan]) -> list[BlockPlan]:
    if not lead:
        return blocks
    if blocks and isinstance(blocks[-1], SectionPlan):
        last = blocks[-1]
        return [*blocks[:-1], last.model_copy(update={"blocks": _continue_last(list(last.blocks), lead)})]
    return [*blocks, *lead]


async def _retry_missing(
    llm: StructuredLlmClient,
    plan: DocumentPlan,
    payload: dict[str, Any],
    by_page: dict[int, list[Segment]],
    drawn: set[int],
) -> DocumentPlan:
    """정상 응답이 빠뜨린 곳을 원래 조각 번호로 한 번 더 묻는다.

    통째로 건너뛴 쪽은 그 쪽 전체를, 쪽 안에서 길게 이어 빠뜨린 구간(_RETRY_RUN 넘게)은 그 구간만 묻는다. 다시 묻기가
    실패하면 그 자리는 조립이 문단으로 되살린다. 쪽 안의 구간을 셀 때는 선과 칸으로 찾은 표 후보 조각(drawn)을 빠진
    것으로 보지 않는다(모델이 표를 건너뛰어도 코드가 표로 세운다).
    """
    count = payload["segment_count"]
    covered = {number for ref in _plan_refs(plan) for number in _ref_ids(ref, count)}
    asks: list[tuple[str, dict[str, Any], set[int]]] = []
    for page in payload["pages"]:
        number = page["page"]
        allowed = {segment.id for segment in by_page.get(number, [])}
        if not allowed:
            continue
        if not allowed & covered:
            # 표 후보는 계획과 맞을 때만 표가 되므로, 통째로 건너뛴 쪽은 표 후보가 있어도 다시 묻는다.
            asks.append((f"application.normalize.retry.page_{number}", page, allowed))
            continue
        for run in _runs(sorted(allowed - covered - drawn)):
            if len(run) > _RETRY_RUN:
                part = {**page, "segments": [segment for segment in page["segments"] if segment["id"] in run]}
                asks.append((f"application.normalize.retry.page_{number}_{run[0]}_{run[-1]}", part, set(run)))
    extra: list[tuple[int, list[BlockPlan]]] = []
    for agent, page, allowed in asks:
        try:
            repair = await _ask(llm, agent, _scoped(payload, [page], allowed))
        except LlmOutputError as error:
            if error.reason == LlmFailureReason.BUDGET:
                break
            logger.warning("정제 재요청 실패 agent=%s reason=%s", agent, error.reason)
            continue
        except (httpx.HTTPError, TimeoutError) as error:
            logger.warning("정제 재요청 실패 agent=%s reason=%s", agent, type(error).__name__)
            continue
        if not _within(repair, allowed, count):
            logger.warning("정제 재요청 답의 조각 범위가 올바르지 않다 agent=%s", agent)
            continue
        made: list[BlockPlan] = [ParagraphPlan(type="paragraph", segments=repair.title)] if repair.title else []
        extra.append((min(allowed), [*made, *repair.blocks]))
    blocks = list(plan.blocks)
    for first, made in extra:
        blocks = _place_page(blocks, made, first, count)
    return plan.model_copy(update={"blocks": blocks}) if extra else plan


def _runs(numbers: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for number in numbers:
        if runs and runs[-1][-1] == number - 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return runs


def _place_page(blocks: list[BlockPlan], made: list[BlockPlan], first: int, count: int) -> list[BlockPlan]:
    """다시 물어 얻은 쪽(또는 구간)의 블록을 읽는 자리에 넣는다. 조각 번호가 곧 읽는 순서다.

    그 쪽보다 뒤 번호로 시작하는 첫 블록 앞에 넣는다. 앞 번호로 시작해 뒤 번호까지 걸친 섹션이면 그 섹션 안에서
    같은 방법으로 넣는다.
    """
    for position, block in enumerate(blocks):
        ids = [number for ref in _plan_refs(block) for number in _ref_ids(ref, count)]
        if not ids:
            continue
        if min(ids) > first:
            return [*blocks[:position], *made, *blocks[position:]]
        if max(ids) > first and isinstance(block, SectionPlan):
            inner = _place_page(list(block.blocks), made, first, count)
            return [*blocks[:position], block.model_copy(update={"blocks": inner}), *blocks[position + 1 :]]
    return [*blocks, *made]


def assemble(
    plan: DocumentPlan,
    segments: list[Segment],
    request: DocumentNormalizationRequest,
    page_count: int,
    *,
    heading_candidates: list[list[int]] | None = None,
) -> tuple[NormalizedDocument, NormalizationReport]:
    """모델 구조에 원문 조각을 채워 정제 문서를 만든다. 정답 정제본도 같은 조립을 쓴다."""
    builder = _Builder({segment.id: segment for segment in segments}, page_count)
    title = builder.take(plan.title)
    blocks = [node for block in plan.blocks if (node := builder.block(block)) is not None]
    builder.restore_grids(blocks, request.pages)
    builder.restore_missing(blocks, title)
    report = builder.report
    if heading_candidates:
        report.heading_candidates = heading_candidates
        body = _body_ids(blocks)
        report.heading_candidates_in_body = sorted({n for group in heading_candidates for n in group if n in body})
    if len(report.missing) > len(segments) * _MAX_MISSING_RATIO:
        raise NormalizationCoverageError("정제 구조가 원문 조각의 절반 넘게 빠뜨렸다")
    data = [rendered for node in blocks if (rendered := builder.render(node)) is not None]
    if not any(block["type"] != "artifact" for block in data):
        raise ValueError("정제된 문서에 내용이 없다")
    counts = {"block": 0, "item": 0}
    for block in data:
        _assign_ids(block, counts)
    document = NormalizedDocument(
        document=DocumentMetadata(
            id=request.document_id,
            type=request.document_type.value.lower(),
            title=join_segments(builder.segments(title)) or None,
            page_count=page_count,
        ),
        blocks=TypeAdapter(list[DocumentBlock]).validate_python(data),
    )
    return document, report


Node = dict[str, Any]


def _body_ids(nodes: list[Node]) -> set[int]:
    ids: set[int] = set()
    for node in nodes:
        if node.get("type") in (None, "paragraph"):
            ids.update(node["ids"])
        ids.update(_body_ids(node.get("blocks", [])))
        ids.update(_body_ids(node.get("items", [])))
    return ids


class _Builder:
    """조각 번호를 한 번씩만 쓰게 하면서 모델 구조를 중간 형태로 옮긴다."""

    def __init__(self, segments: dict[int, Segment], page_count: int) -> None:
        self._segments = segments
        self._page_count = page_count
        self._used: set[int] = set()
        # 조각 번호가 놓인 블록과 그 블록이 든 목록. 빠진 조각을 그 뒤에 되살린다. 블록이 None이면 목록 맨 앞이다.
        self._owner: dict[int, tuple[list[Node], Node | None]] = {}
        self.report = NormalizationReport(segments=len(segments))
        by_page: dict[int, list[Segment]] = {}
        # 쪽의 첫 줄이나 마지막 줄에 떨어져 있는 조각. 쪽 번호는 여기 있을 때만 숨긴다.
        self._margin_ids: set[int] = set()
        for segment in segments.values():
            if segment.x < segment.end_x:
                by_page.setdefault(segment.page, []).append(segment)
        self._by_page = by_page
        self._text_width = {
            page: max(s.end_x for s in group) - min(s.x for s in group) for page, group in by_page.items()
        }
        for group in by_page.values():
            lines = sorted({segment.y for segment in group})
            if len(lines) < 2:
                continue
            for segment in group:
                if segment.y == lines[0] and lines[1] - segment.y >= _MARGIN_GAP:
                    self._margin_ids.add(segment.id)
                if segment.y == lines[-1] and segment.y - lines[-2] >= _MARGIN_GAP:
                    self._margin_ids.add(segment.id)

    def take(self, refs: list[str] | None) -> list[int]:
        found: list[int] = []
        for ref in refs or []:
            start, _, end = ref.partition("-")
            first, last = int(start), int(end or start)
            if first > last or first not in self._segments or last not in self._segments:
                self.report.unknown.append(ref)
                continue
            for number in range(first, last + 1):
                if number in self._used:
                    self.report.duplicated.append(number)
                    continue
                self._used.add(number)
                found.append(number)
        return found

    def segments(self, ids: list[int]) -> list[Segment]:
        return [self._segments[number] for number in ids]

    def block(self, plan: Any) -> Node | None:
        if isinstance(plan, SectionPlan):
            title = self.take(plan.title)
            children = [node for child in plan.blocks if (node := self.block(child)) is not None]
            title, moved = self._single_title(title)
            if moved:
                self.report.split_titles += 1
                children.insert(0, {"type": "paragraph", "ids": moved, "keep_lines": False})
            return {"type": "section", "title": title, "blocks": children}
        if isinstance(plan, ParagraphPlan):
            return {"type": "paragraph", "ids": self.take(plan.segments), "keep_lines": plan.keep_lines}
        if isinstance(plan, ListPlan):
            return {"type": "list", "items": [self._item(item) for item in plan.items]}
        if isinstance(plan, TablePlan):
            return {
                "type": "table",
                "title": self.take(plan.title),
                "headers": [self.take(cell) for cell in plan.headers],
                "rows": [[self.take(cell) for cell in row] for row in plan.rows],
            }
        if isinstance(plan, VisualNoticePlan):
            pages = sorted({page for page in plan.pages if 1 <= page <= self._page_count})
            if not pages:
                return None
            # 모델이 쓴 문장은 내보내지 않는다. 종류와 쪽만 받고 안내문은 코드가 쓴다.
            message = visual_notice_message(plan.visual_type, pages)
            return {"type": "visual_notice", "pages": pages, "visual_type": plan.visual_type, "message": message}
        if isinstance(plan, ArtifactPlan):
            ids = self.take(plan.segments)
            if self._is_artifact(plan.role, ids):
                return {"type": "artifact", "role": plan.role, "ids": ids}
            self.report.demoted_artifacts.extend(ids)
            return {"type": "paragraph", "ids": ids, "keep_lines": False}
        return None

    def _is_artifact(self, role: str, ids: list[int]) -> bool:
        """글이 든 줄은 숨기지 않는다. 숨기는 것은 쪽 가장자리 줄의 쪽 번호, 구분선, 목록 머리표 기호뿐이다.

        쪽마다 반복되는 머리글과 바닥글도 원문이라 보이게 둔다. 모델이 붙인 header, footer 표시는 따르지 않는다.
        """
        if role not in ("page_number", "decoration"):
            return False
        for segment in self.segments(ids):
            text = _SPACE.sub("", segment.text)
            if role == "page_number" and (
                not _PAGE_NUMBER.fullmatch(text)
                or segment.id not in self._margin_ids
                or not self._apart_from_line_text(segment, ids)
            ):
                return False
            if role == "decoration" and not (_is_separator(text) or self._is_bullet(segment, text)):
                return False
        return True

    def _is_bullet(self, segment: Segment, text: str) -> bool:
        """줄 첫머리에 있고 바로 오른쪽에 글이 붙은 한 글자 기호만 머리표다.

        글 뒤에 놓인 기호(기간 사이 붙임표, 선택 표시)는 뜻이 있으므로 숨기지 않는다. 같은 줄에 글 뒤에 놓인 선택 표시가
        하나라도 있으면 그 줄은 선택지 줄이라 첫머리 기호도 숨기지 않는다.
        """
        if len(text) != 1 or text not in _BULLETS:
            return False
        line = self._line_of(segment)

        def after_text(mark: Segment) -> bool:
            return any(other.end_x <= mark.x and _has_text(other) for other in line if other.id != mark.id)

        if after_text(segment):
            return False
        if any(_SPACE.sub("", other.text) in _OPTION_MARKS and after_text(other) for other in line):
            return False
        return any(0 <= other.x - segment.end_x <= _BULLET_GAP and _has_text(other) for other in line)

    def _apart_from_line_text(self, segment: Segment, ids: list[int]) -> bool:
        """같은 줄의 다른 글이 없거나, 모두 쪽 글 영역 너비의 1/3 이상 떨어져 있다."""
        apart = self._text_width.get(segment.page, 0) * _PAGE_NUMBER_APART
        return all(
            other.id in ids or not _has_text(other) or max(other.x - segment.end_x, segment.x - other.end_x) >= apart
            for other in self._line_of(segment)
        )

    def _line_of(self, segment: Segment) -> list[Segment]:
        """같은 쪽, 같은 줄에 있는 다른 조각."""
        return [
            other
            for other in self._by_page.get(segment.page, [])
            if other.id != segment.id and abs(other.y - segment.y) <= _SAME_LINE
        ]

    def _item(self, plan: ItemPlan) -> Node:
        return {"ids": self.take(plan.segments), "items": [self._item(child) for child in plan.items]}

    def _single_title(self, title: list[int]) -> tuple[list[int], list[int]]:
        """section 제목이 세 줄 이상이면 본문이 섞인 것으로 보고 첫 줄만 제목으로 남긴다."""
        lines: list[tuple[int, float]] = []
        for segment in self.segments(title):
            key = (segment.page, segment.y)
            if not any(page == key[0] and abs(y - key[1]) < _SAME_LINE for page, y in lines):
                lines.append(key)
        if len(lines) < 3:
            return title, []
        page, y = lines[0]
        head = [n for n in title if self._segments[n].page == page and abs(self._segments[n].y - y) < _SAME_LINE]
        return head, [n for n in title if n not in head]

    def restore_grids(self, blocks: list[Node], pages: list[PdfPage]) -> None:
        for page in pages:
            # 예전 대기 작업은 그리기 좌표 단서 없이 격자선만 전달한다.
            if page.layout is not None or not page.grid_lines:
                continue
            segments = [s for s in self._segments.values() if s.page == page.number]
            for xs, ys in closed_grids(page.grid_lines):
                cells = _grid_cells(segments, xs, ys)
                ids = {n for row in cells for cell in row for n in cell}
                if not ids or ids & self._used or len(ids) != max(ids) - min(ids) + 1:
                    continue
                first, last = min(ids), max(ids)
                neighbours = [s for s in segments if s.id in self._used]
                before = next((s for s in reversed(neighbours) if s.id < first), None)
                after = next((s for s in neighbours if s.id > last), None)
                if before is None or after is None or not before.y < ys[0] < ys[-1] < after.y:
                    continue
                first_row_colors = {self._segments[n].color for cell in cells[0] for n in cell}
                body_colors = {self._segments[n].color for row in cells[1:] for cell in row for n in cell}
                # 격자만으로 첫 행이 머리글인지는 모른다. 글자 색이 나머지 행과 일관되게 다를 때만 분리한다.
                has_header = len(first_row_colors) == len(body_colors) == 1 and not (
                    None in (first_row_colors | body_colors) or first_row_colors == body_colors
                )
                table = {
                    "type": "table",
                    "title": [],
                    "headers": cells[0] if has_header else [],
                    "rows": cells[1:] if has_header else cells,
                }
                if self._insert_grid(blocks, before.id, after.id, first, last, table):
                    self._used.update(ids)

    def _insert_grid(self, blocks: list[Node], before: int, after: int, first: int, last: int, table: Node) -> bool:
        refs = [self._subtree_ids(block) for block in blocks]
        left = next((i for i, ids in enumerate(refs) if before in ids), None)
        right = next((i for i, ids in enumerate(refs) if after in ids), None)
        if left is None or right is None:
            return False
        if left == right and blocks[left]["type"] == "section":
            return self._insert_grid(blocks[left]["blocks"], before, after, first, last, table)
        if left >= right or any(refs[left + 1 : right]) or max(refs[left]) >= first or min(refs[right]) <= last:
            return False
        blocks.insert(right, table)
        return True

    def _subtree_ids(self, block: Node) -> set[int]:
        if block["type"] != "section":
            return set(self._ids(block))
        ids: set[int] = set(block["title"])
        for child in block["blocks"]:
            ids.update(self._subtree_ids(child))
        return ids

    def restore_missing(self, blocks: list[Node], title: list[int]) -> None:
        self._index(blocks)
        missing = sorted(set(self._segments) - self._used)
        self.report.missing = missing
        runs: list[list[int]] = []
        for number in missing:
            if runs and runs[-1][-1] == number - 1 and self._segments[number - 1].page == self._segments[number].page:
                runs[-1].append(number)
            else:
                runs.append([number])
        # 뒤의 묶음부터 넣어야 같은 블록 뒤에 넣은 묶음들이 번호 순서를 지킨다.
        for run in reversed(runs):
            node: Node = {"type": "paragraph", "ids": run, "keep_lines": False}
            anchor = next((n for n in range(run[0] - 1, 0, -1) if n in self._owner), None)
            if anchor is None:
                blocks.insert(0, node)
                continue
            siblings, owner = self._owner[anchor]
            position = 0 if owner is None else next(i for i, sibling in enumerate(siblings) if sibling is owner) + 1
            siblings.insert(position, node)

    def _index(self, blocks: list[Node]) -> None:
        for block in blocks:
            for number in self._ids(block):
                self._owner[number] = (blocks, block)
            if block["type"] == "section":
                # 제목 바로 뒤에 빠진 조각은 그 section의 맨 앞에 되살린다.
                for number in block["title"]:
                    self._owner[number] = (block["blocks"], None)
                self._index(block["blocks"])

    def _ids(self, block: Node) -> list[int]:
        if block["type"] in ("paragraph", "artifact"):
            return list(block["ids"])
        if block["type"] == "list":
            return [number for item in block["items"] for number in self._item_ids(item)]
        if block["type"] == "table":
            cells = [block["title"], *block["headers"], *(cell for row in block["rows"] for cell in row)]
            return [number for cell in cells for number in cell]
        return []

    def _item_ids(self, item: Node) -> list[int]:
        return [*item["ids"], *(number for child in item["items"] for number in self._item_ids(child))]

    def _pages(self, ids: list[int]) -> list[int]:
        return sorted({self._segments[number].page for number in ids})

    def _text(self, ids: list[int], keep_lines: bool = False) -> str:
        return join_segments(self.segments(ids), keep_lines=keep_lines)

    def render(self, node: Node) -> dict[str, Any] | None:
        kind = node["type"]
        if kind == "section":
            children = [rendered for child in node["blocks"] if (rendered := self.render(child)) is not None]
            if not node["title"] and not children:
                return None
            pages = set(self._pages(node["title"]))
            for child in children:
                pages.update(child["pages"])
            return {
                "type": "section",
                "title": self._text(node["title"]) or None,
                "pages": sorted(pages),
                "blocks": children,
            }
        if kind in ("paragraph", "artifact"):
            if not node["ids"]:
                return None
            text = self._text(node["ids"], node.get("keep_lines", False))
            if kind == "artifact":
                return {"type": "artifact", "pages": self._pages(node["ids"]), "role": node["role"], "text": text}
            return {"type": "paragraph", "pages": self._pages(node["ids"]), "text": text}
        if kind == "list":
            items = [rendered for item in node["items"] if (rendered := self._render_item(item)) is not None]
            if not items:
                return None
            return {"type": "list", "pages": self._pages(self._ids(node)), "items": items}
        if kind == "table":
            ids = self._ids(node)
            if not ids:
                return None
            return {
                "type": "table",
                "pages": self._pages(ids),
                "title": self._text(node["title"]) or None,
                "headers": [self._text(cell) for cell in node["headers"]],
                "rows": [[self._text(cell) for cell in row] for row in node["rows"] if any(row)],
                "reconstruction": "inferred",
            }
        if kind == "visual_notice":
            return {
                "type": "visual_notice",
                "pages": node["pages"],
                "visualType": node["visual_type"],
                "detection": "suspected",
                "message": node["message"],
            }
        return None

    def _render_item(self, item: Node) -> dict[str, Any] | None:
        children = [rendered for child in item["items"] if (rendered := self._render_item(child)) is not None]
        if not item["ids"] and not children:
            return None
        pages = set(self._pages(item["ids"]))
        for child in children:
            pages.update(child["pages"])
        return {"text": self._text(item["ids"]), "pages": sorted(pages), "items": children}


def _grid_cells(segments: list[Segment], xs: list[float], ys: list[float]) -> list[list[list[int]]]:
    cells: list[list[list[int]]] = [[[] for _ in xs[1:]] for _ in ys[1:]]
    for segment in segments:
        if not (ys[0] < segment.y < ys[-1] and xs[0] < segment.end_x and segment.x < xs[-1]):
            continue
        row = bisect_right(ys, segment.y) - 1
        column = bisect_right(xs, segment.x) - 1
        if column < 0 or not xs[column] < segment.x < segment.end_x < xs[column + 1] or segment.y == ys[row]:
            return []
        cells[row][column].append(segment.id)
    # 빈 칸과 병합 셀을 구분할 수 없는 격자는 기존 문단 복구에 맡긴다.
    return cells if all(cell for row in cells for cell in row) else []


def _assign_ids(value: dict[str, Any], counts: dict[str, int], *, item: bool = False) -> None:
    kind = "item" if item else "block"
    counts[kind] += 1
    value["id"] = f"{kind}-{counts[kind]:04d}"
    for child in value.get("blocks", []):
        _assign_ids(child, counts)
    for child in value.get("items", []):
        _assign_ids(child, counts, item=True)
