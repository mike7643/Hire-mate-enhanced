"""글자 크기로 제목 단계를 맞춘다.

자유양식 서류에서 제목의 단계는 대개 글자 크기로 드러난다. 모델이 큰 제목 아래에 둘 작은 제목을 형제로 두거나, 작은
제목 아래에 큰 제목을 넣거나, 제목 줄을 문단 첫머리에 붙여 두면 읽는 구조가 틀린다. 여기서는 본문보다 큰 글자로 쓴
제목만 근거로, 조각 번호(읽는 순서)를 지키며 제목 단계를 다시 묶는다. 문서 종류나 특정 양식에 기대지 않는다.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hiremate_ai.agents.analysis.application.normalize import BlockPlan, Segment
    from hiremate_ai.contract import PdfFillRect, PdfPage

# 이 차이 안의 글자 크기는 같은 크기로 본다(14.94와 15).
_SIZE_STEP = 0.6
# 본문보다 이만큼 이상 커야 크기로 구분한 제목으로 본다. 본문 크기의 굵은 제목은 크기로 단계를 알 수 없다.
_OVER_BODY = 1.0
# 문단 첫 줄을 제목으로 떼어 낼 때 그 줄의 최대 글자 수.
_LEAD_CHARS = 80
_SAME_LINE = 1.0
# 떼어 낸 제목 다음 글은 제목 아래 줄에서 시작해야 한다(제목 크기에 대한 비율). 옆 칸 글이면 제목이 아니라 표 이름표다.
_NEXT_LINE = 0.6
# 상자로 보는 채움 영역: 글자 크기의 이 배수 이상 높고, 쪽 너비의 이 비율 이상 넓다.
_BOX_LINES = 3.0
_BOX_WIDTH = 0.2
# 쪽 넓이의 이 비율 이상을 덮는 채움은 쪽 바탕이다.
_PAGE_COVER = 0.9
# 주소, 전자우편, 전화번호. 이런 줄은 제목이 아니라 연락처 본문이다.
_CONTACT = re.compile(r"https?://|www\.|[\w.+-]+@[\w-]+\.[A-Za-z]{2,}|\b0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}\b")


def refine_outline(blocks: list[BlockPlan], source: dict[int, Segment], pages: list[PdfPage]) -> list[BlockPlan]:
    """연락처 줄 제목을 본문으로 돌리고, 제목 크기 줄을 문단에서 떼어 낸 뒤, 크기 순서와 어긋난 단계를 다시 묶는다."""
    body = _body_size(source.values())
    blocks = _demote_contact_titles(blocks, source)
    if body is None:
        return blocks
    sizes = _title_sizes(blocks, source, body)
    if not sizes:
        return blocks
    blocks = _promote_lead_lines(blocks, source, body, sizes)
    ranks = _Ranks(_title_sizes(blocks, source, body), blocks, source, body)
    places = _Places(source, pages)
    if not _conflicts(blocks, ranks, places, None):
        return blocks
    return _rebuild(blocks, ranks, places)


def _body_size(segments: Iterable[Segment]) -> float | None:
    """글자 수가 가장 많은 크기를 본문 크기로 본다."""
    weights: Counter[float] = Counter()
    for segment in segments:
        if segment.size:
            weights[round(segment.size, 1)] += len(segment.text.strip())
    return weights.most_common(1)[0][0] if weights else None


def _numbers(ref: str, source: dict[int, Segment]) -> list[int]:
    start, _, end = ref.partition("-")
    try:
        first, last = int(start), int(end or start)
    except ValueError:
        return []
    return [n for n in range(first, last + 1) if n in source] if first <= last else []


def _ids(refs: list[str] | None, source: dict[int, Segment]) -> list[int]:
    return [n for ref in refs or [] for n in _numbers(ref, source)]


def _refs(ids: list[int]) -> list[str]:
    runs: list[list[int]] = []
    for n in ids:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return [str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs]


def _title_size(block: BlockPlan, source: dict[int, Segment]) -> float | None:
    if block.type != "section":
        return None
    ids = _ids(block.title, source)
    return source[ids[0]].size if ids else None


def _sections(blocks: list[BlockPlan]) -> Iterator[BlockPlan]:
    for block in blocks:
        if block.type == "section":
            yield block
            yield from _sections(block.blocks)


def _title_sizes(blocks: list[BlockPlan], source: dict[int, Segment], body: float) -> list[float]:
    """본문보다 큰 제목 크기들. 가까운 크기는 하나로 묶고 큰 순서로 돌려준다."""
    found = sorted(
        {size for block in _sections(blocks) if (size := _title_size(block, source)) and size >= body + _OVER_BODY},
        reverse=True,
    )
    clusters: list[float] = []
    for size in found:
        if not clusters or clusters[-1] - size > _SIZE_STEP:
            clusters.append(size)
    return clusters


def _demote_contact_titles(blocks: list[BlockPlan], source: dict[int, Segment]) -> list[BlockPlan]:
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan

    result: list[BlockPlan] = []
    for block in blocks:
        if block.type != "section":
            result.append(block)
            continue
        children = _demote_contact_titles(block.blocks, source)
        title = " ".join(source[n].text for n in _ids(block.title, source))
        if title and _CONTACT.search(title):
            result.extend([ParagraphPlan(type="paragraph", segments=list(block.title or [])), *children])
        elif children != block.blocks:
            result.append(block.model_copy(update={"blocks": children}))
        else:
            result.append(block)
    return result


def _promote_lead_lines(
    blocks: list[BlockPlan], source: dict[int, Segment], body: float, sizes: list[float]
) -> list[BlockPlan]:
    """문단 첫 줄이 이 문서의 제목 크기이고 다음 줄이 그보다 작으면, 그 줄을 제목으로 떼어 섹션을 만든다."""
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan, SectionPlan

    result: list[BlockPlan] = []
    for block in blocks:
        if block.type == "section":
            children = _promote_lead_lines(block.blocks, source, body, sizes)
            result.append(block.model_copy(update={"blocks": children}) if children != block.blocks else block)
            continue
        if block.type != "paragraph":
            result.append(block)
            continue
        ids = _ids(block.segments, source)
        lead = _lead_line(ids, source)
        rest = ids[len(lead) :]
        size = source[ids[0]].size if ids else None
        if (
            not lead
            or not rest
            or size is None
            or size < body + _OVER_BODY
            or not any(abs(size - known) <= _SIZE_STEP for known in sizes)
            or any(abs((source[n].size or 0) - size) > _SIZE_STEP for n in lead)
            or (source[rest[0]].size or size) > size - _SIZE_STEP
            or source[rest[0]].page == source[lead[0]].page
            and source[rest[0]].y - source[lead[0]].y < _NEXT_LINE * size
            or len(" ".join(source[n].text.strip() for n in lead)) > _LEAD_CHARS
        ):
            result.append(block)
            continue
        paragraph = ParagraphPlan(type="paragraph", segments=_refs(rest), keep_lines=block.keep_lines)
        result.append(SectionPlan(type="section", title=_refs(lead), blocks=[paragraph]))
    return result


def _lead_line(ids: list[int], source: dict[int, Segment]) -> list[int]:
    if not ids:
        return []
    first = source[ids[0]]
    lead: list[int] = []
    for n in ids:
        segment = source[n]
        if segment.page != first.page or abs(segment.y - first.y) > _SAME_LINE:
            break
        lead.append(n)
    return lead


class _Ranks:
    """제목 크기의 단계. 0이 가장 큰 제목이다.

    가장 큰 크기의 제목이 문서 첫 제목 하나뿐이면(이름이나 문서 제목) 다른 제목을 품는 단계로 쓰지 않는다.
    """

    def __init__(self, sizes: list[float], blocks: list[BlockPlan], source: dict[int, Segment], body: float) -> None:
        self._source = source
        self._body = body
        self._sizes = sizes
        sized = [block for block in _sections(blocks) if self._raw(block) is not None]
        self._skip: int | None = None
        if len(sizes) > 1 and sized and sum(self._raw(block) == 0 for block in sized) == 1 and self._raw(sized[0]) == 0:
            self._skip = id(sized[0])

    def _raw(self, block: BlockPlan) -> int | None:
        size = _title_size(block, self._source)
        if size is None or size < self._body + _OVER_BODY:
            return None
        return next((rank for rank, known in enumerate(self._sizes) if abs(size - known) <= _SIZE_STEP), None)

    def of(self, block: BlockPlan) -> int | None:
        if id(block) == self._skip:
            return None
        return self._raw(block)


class _Places:
    """블록이 어느 제목 아래에 들어갈 수 있는지 읽는 순서와 담긴 상자로 판단한다.

    제목 아래 내용은 읽는 순서(조각 번호)로 제목 뒤에 오고, 제목과 같은 상자에 있다. 상자는 줄 여러 개를 덮는 채움
    영역이다(사이드바, 회색 프로젝트 상자). 제목 띠처럼 한 줄만 덮는 채움은 상자로 보지 않는다.
    """

    def __init__(self, source: dict[int, Segment], pages: list[PdfPage]) -> None:
        self._source = source
        fills: dict[int, tuple[list[PdfFillRect], float]] = {}
        for page in pages:
            if page.layout is None:
                continue
            area = page.layout.width * page.layout.height
            covers = [r for r in page.layout.fill_rects if (r.x2 - r.x1) * (r.y2 - r.y1) >= _PAGE_COVER * area]
            ground = max(covers, key=lambda r: (r.x2 - r.x1) * (r.y2 - r.y1)).color.lower() if covers else "#ffffff"
            # 바탕과 같은 색의 채움은 보이지 않으므로 상자가 아니다.
            visible = [r for r in page.layout.fill_rects if r.color.lower() != ground]
            fills[page.number] = (visible, page.layout.width)
        self._box: dict[int, tuple[int, int, int, int] | None] = {}
        for n, segment in source.items():
            rects, width = fills.get(segment.page, ([], 0.0))
            tall = _BOX_LINES * (segment.size or 10.0)
            boxes = [
                rect
                for rect in rects
                if rect.y2 - rect.y1 >= tall
                and rect.x2 - rect.x1 >= _BOX_WIDTH * width
                and rect.x1 - 0.5 <= segment.x < segment.end_x <= rect.x2 + 0.5
                and rect.y1 <= segment.y <= rect.y2
            ]
            box = min(boxes, key=lambda rect: (rect.x2 - rect.x1) * (rect.y2 - rect.y1), default=None)
            self._box[n] = None if box is None else (round(box.x1), round(box.y1), round(box.x2), round(box.y2))

    def lead(self, block: BlockPlan | _Node) -> int | None:
        if isinstance(block, _Node):
            return block.lead
        if block.type == "section" and block.title:
            ids = _ids(block.title, self._source)
            if ids:
                return ids[0]
        ids = [n for refs in _refs_of(block) for n in _ids(refs, self._source)]
        return min(ids) if ids else None

    def inside(self, head: int | None, block: BlockPlan) -> bool:
        lead = self.lead(block)
        if head is None or lead is None:
            return True
        return lead > head and self._box.get(lead) == self._box.get(head)


def _refs_of(block: BlockPlan) -> Iterator[list[str]]:
    if block.type in ("paragraph", "artifact"):
        yield block.segments
    elif block.type == "section":
        yield block.title or []
        for child in block.blocks:
            yield from _refs_of(child)
    elif block.type == "table":
        yield block.title or []
        yield from block.headers
        for row in block.rows:
            yield from row
    elif block.type == "list":
        stack = list(block.items)
        while stack:
            item = stack.pop()
            yield item.segments
            stack.extend(item.items)


def _holds_ranked(block: BlockPlan, ranks: _Ranks) -> bool:
    return block.type == "section" and any(ranks.of(child) is not None for child in _sections(block.blocks))


def _conflicts(blocks: list[BlockPlan], ranks: _Ranks, places: _Places, parent: int | None) -> bool:
    """크기 순서와 어긋난 단계가 있는가.

    작은 제목 아래 큰 제목, 또는 큰 제목 뒤에 형제로 놓였지만 그 제목 안에 들어갈 작은 제목이 있으면 어긋난 것이다.
    """
    seen: list[tuple[int, int | None]] = []
    for block in blocks:
        if block.type != "section":
            continue
        rank = ranks.of(block)
        if rank is not None:
            if parent is not None and rank <= parent:
                return True
            if any(earlier < rank and places.inside(head, block) for earlier, head in seen):
                return True
            seen.append((rank, places.lead(block)))
            if _conflicts(block.blocks, ranks, places, rank):
                return True
        elif not block.title and _conflicts(block.blocks, ranks, places, parent):
            return True
    return False


@dataclass
class _Node:
    rank: int
    title: list[str] | None
    lead: int | None
    blocks: list[BlockPlan | _Node] = field(default_factory=list)


def _rebuild(blocks: list[BlockPlan], ranks: _Ranks, places: _Places) -> list[BlockPlan]:
    """읽는 순서를 지키며 크기 단계로 다시 묶는다.

    제목 뒤의 내용은 같은 크기 이상의 제목이 나올 때까지 가장 깊은 열린 제목에 속한다. 다만 모델이 형제로 둔 블록은 읽는
    순서로 그 제목 뒤에 있고 같은 단에 있을 때만 그 제목 안으로 넣는다. 모델이 어떤 섹션 안에 둔 블록은 그 섹션 밖으로
    빼지 않는다(더 큰 제목만 위로 올린다). 크기를 모르는 제목의 섹션(본문 크기 제목)은 통째로 한 덩어리로 옮기고,
    제목 없는 섹션은 크기 제목을 품을 때만 푼다.
    """
    root: list[BlockPlan | _Node] = []
    stack: list[_Node] = []

    def place(item: BlockPlan | _Node) -> None:
        (stack[-1].blocks if stack else root).append(item)

    def settle(block: BlockPlan, floor: int) -> None:
        while len(stack) > floor and not places.inside(stack[-1].lead, block):
            stack.pop()

    def visit(entries: list[BlockPlan], floor: int) -> None:
        for block in entries:
            floor = min(floor, len(stack))
            if block.type == "section":
                rank = ranks.of(block)
                if rank is not None:
                    while stack and stack[-1].rank >= rank:
                        stack.pop()
                    floor = min(floor, len(stack))
                    settle(block, floor)
                    node = _Node(rank, block.title, places.lead(block))
                    place(node)
                    stack.append(node)
                    visit(block.blocks, len(stack))
                    continue
                if not block.title and _holds_ranked(block, ranks):
                    visit(block.blocks, floor)
                    continue
            settle(block, floor)
            place(block)

    visit(blocks, 0)
    return [_plan(item) for item in root]


def _plan(item: BlockPlan | _Node) -> BlockPlan:
    from hiremate_ai.agents.analysis.application.normalize import SectionPlan

    if isinstance(item, _Node):
        return SectionPlan(type="section", title=item.title, blocks=[_plan(child) for child in item.blocks])
    return item
