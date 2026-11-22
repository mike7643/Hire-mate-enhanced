"""원문 글 조각에 구조를 입혀 정제 문서를 만든다.

모델은 조각 번호로 구조(제목, 문단, 목록, 표, 잡음)만 정하고 글은 쓰지 않는다. 글은 코드가 파서의 원문
조각을 이어 붙여 만든다. 원문과 달라질 수 있는 것은 공백과 줄바꿈뿐이다. 모델이 빠뜨린 조각은 앞 조각
가까이에 문단으로 되살리되, 닫힌 격자와 위치가 명확한 누락 표는 표로 되살린다. 두 번 쓴 조각은 처음 자리만 남긴다.
"""

import json
import logging
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from hiremate_ai.agents.analysis.application.grid import closed_grids
from hiremate_ai.agents.analysis.application.normalization_prompt import system_prompt
from hiremate_ai.config import settings
from hiremate_ai.contract import (
    DocumentBlock,
    DocumentMetadata,
    DocumentNormalizationRequest,
    NormalizedDocument,
    PdfPage,
)
from hiremate_ai.llm.chat import model_call_budget
from hiremate_ai.llm.client import LlmClient
from hiremate_ai.llm.prompt import data_block

logger = logging.getLogger(__name__)

_SPACE = re.compile(r"\s+")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
# 같은 줄로 보는 세로 좌표 차이와, 한 글자씩 벌려 쓴 제목을 한 조각으로 합치는 가로 간격.
_SAME_LINE = 1.0
_LETTER_GAP = 12.0
_MARGIN_GAP = 12.0
_MARGIN_ALIGNMENT = 6.0
# 모델이 이보다 많은 조각을 빠뜨리면 구조를 믿을 수 없어 실패로 둔다. 실패한 서류는 PDF 추출 글로 읽는다.
_MAX_MISSING_RATIO = 0.5
# 조각 번호는 ASCII이며, 로컬 JSON 문법 생성기에서 \d 대신 명시적 범위를 쓴다.
_REF = r"^[0-9]+(-[0-9]+)?$"
_BRACKET_HEADING = re.compile(r"\[([^\[\]\n]{1,80})\]")
# 쪽 번호로 받는 모양: 1, - 1 -, 1/3, p.2, 3쪽
_PAGE_NUMBER = re.compile(r"(?i)[\W_]*(?:page|p\.?)?\s*\d{1,3}\s*(?:쪽|페이지)?(?:[\W_]+\d{1,3})?[\W_]*")


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
        for x, y, end_x, text, size, color in pieces:
            result.append(Segment(len(result) + 1, page.number, x, y, end_x, text, size, color))
    return result


def _heading_candidates(by_page: dict[int, list[Segment]]) -> list[list[int]]:
    """같은 단에서 다음 줄과 글자 모양이 대비되는 줄을 찾는다. 조각 번호는 바꾸지 않는다."""
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
                if gap < _SAME_LINE or abs(following.x - head.x) > 8 or following.size is None:
                    continue
                if gap <= 1.7 * head.size and head.size == following.size and head.color == following.color:
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


async def normalize_document(llm: LlmClient, request: DocumentNormalizationRequest) -> NormalizedDocument:
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
    llm: LlmClient, request: DocumentNormalizationRequest
) -> tuple[NormalizedDocument, NormalizationReport]:
    if not request.pages or [page.number for page in request.pages] != list(range(1, len(request.pages) + 1)):
        raise ValueError("정제할 페이지 정보가 없거나 순서가 올바르지 않다")
    if any("PAGE_EXTRACTION_FAILED" in page.issues for page in request.pages):
        raise ValueError("페이지 추출에 실패했다")
    if not any(page.text.strip() for page in request.pages):
        raise ValueError("추출된 텍스트가 없다")
    segments = build_segments(request.pages)
    if not segments:
        raise ValueError("추출된 텍스트가 없다")
    page_count = len(request.pages)
    by_page: dict[int, list[Segment]] = {}
    for segment in segments:
        by_page.setdefault(segment.page, []).append(segment)
    heading_candidates = _heading_candidates(by_page)
    payload = json.dumps(
        {
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
        },
        ensure_ascii=False,
    )
    with model_call_budget(settings.llm_agent_max_calls):
        plan = await llm.structured(
            DocumentPlan,
            agent="application.normalize",
            system=system_prompt(),
            user=data_block("pdf_segments", payload),
        )
    return assemble(plan, segments, request, page_count, heading_candidates=heading_candidates)


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
        raise ValueError("정제 구조가 원문 조각의 절반 넘게 빠뜨렸다")
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
        self._margin_text: dict[tuple[str, str], list[Segment]] = {}
        for segment in segments.values():
            if segment.x < segment.end_x:
                by_page.setdefault(segment.page, []).append(segment)
        for group in by_page.values():
            lines = sorted({segment.y for segment in group})
            if len(lines) < 2:
                continue
            for segment in group:
                text = _SPACE.sub("", segment.text)
                if segment.y == lines[0] and lines[1] - segment.y >= _MARGIN_GAP:
                    self._margin_text.setdefault(("header", text), []).append(segment)
                if segment.y == lines[-1] and segment.y - lines[-2] >= _MARGIN_GAP:
                    self._margin_text.setdefault(("footer", text), []).append(segment)

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
            return {"type": "visual_notice", "pages": pages, "visual_type": plan.visual_type, "message": plan.message}
        if isinstance(plan, ArtifactPlan):
            ids = self.take(plan.segments)
            if self._is_artifact(plan.role, ids):
                return {"type": "artifact", "role": plan.role, "ids": ids}
            self.report.demoted_artifacts.extend(ids)
            return {"type": "paragraph", "ids": ids, "keep_lines": False}
        return None

    def _is_artifact(self, role: str, ids: list[int]) -> bool:
        """머리글과 바닥글은 두 쪽 이상에서 같은 쪽 가장자리 위치에 있어야 한다."""
        for segment in self.segments(ids):
            text = _SPACE.sub("", segment.text)
            if role == "page_number" and not _PAGE_NUMBER.fullmatch(text):
                return False
            if role == "decoration" and any(char.isalnum() for char in text):
                return False
            if role in ("header", "footer"):
                matching = self._margin_text.get((role, text), [])
                if segment not in matching or not any(
                    other.page != segment.page
                    and abs(other.x - segment.x) <= _MARGIN_ALIGNMENT
                    and abs(other.y - segment.y) <= _MARGIN_ALIGNMENT
                    for other in matching
                ):
                    return False
        return True

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
            if not page.grid_lines:
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
