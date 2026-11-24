"""검증된 PDF 격자를 원문 번호 계획에 반영한다. 모델이 적은 문자열을 표에 쓰지 않는다."""

from collections import Counter
from collections.abc import Iterator
from typing import TYPE_CHECKING

from hiremate_ai.agents.analysis.application.normalize import (
    ArtifactPlan,
    BlockPlan,
    DocumentPlan,
    ItemPlan,
    ListPlan,
    ParagraphPlan,
    Refs,
    SectionPlan,
    Segment,
    TablePlan,
    _plan_refs,
    _ref_ids,
)

if TYPE_CHECKING:
    from hiremate_ai.agents.analysis.application.layout_grid import GeometryTable


class _References:
    def __init__(self, segments: list[Segment]) -> None:
        self.source = {s.id: s for s in segments}
        self.count = len(segments)

    def ids(self, node: DocumentPlan | BlockPlan | ItemPlan) -> Iterator[int]:
        for ref in _plan_refs(node):
            yield from _ref_ids(ref, self.count)

    def strip(self, refs: Refs | None, target: set[int]) -> Refs:
        kept: Refs = []
        for ref in refs or []:
            ids = _ref_ids(ref, self.count)
            if target.isdisjoint(ids):
                kept.append(ref)
            else:
                kept.extend(str(n) for n in ids if n not in target)
        return kept

    def strip_item(self, item: ItemPlan, target: set[int]) -> ItemPlan | None:
        refs = self.strip(item.segments, target)
        children = [child for it in item.items if (child := self.strip_item(it, target)) is not None]
        return ItemPlan(segments=refs, items=children) if refs or children else None

    def strip_block(self, block: BlockPlan, target: set[int]) -> BlockPlan | None:
        if isinstance(block, SectionPlan):
            refs = self.strip(block.title, target)
            children = [child for b in block.blocks if (child := self.strip_block(b, target)) is not None]
            return block.model_copy(update={"title": refs or None, "blocks": children}) if refs or children else None
        if isinstance(block, (ParagraphPlan, ArtifactPlan)):
            refs = self.strip(block.segments, target)
            return block.model_copy(update={"segments": refs}) if refs else None
        if isinstance(block, ListPlan):
            items = [item for i in block.items if (item := self.strip_item(i, target)) is not None]
            return block.model_copy(update={"items": items}) if items else None
        if isinstance(block, TablePlan):
            refs = self.strip(block.title, target)
            headers = [self.strip(cell, target) for cell in block.headers]
            rows = [[self.strip(cell, target) for cell in row] for row in block.rows]
            if refs or any(headers) or any(any(row) for row in rows):
                return block.model_copy(update={"title": refs or None, "headers": headers, "rows": rows})
            return None
        return block

    def replace(self, blocks: list[BlockPlan], target: set[int], table: TablePlan) -> list[BlockPlan] | None:
        touched = [index for index, b in enumerate(blocks) if not target.isdisjoint(self.ids(b))]
        if not touched:
            return None
        first = touched[0]
        anchor = blocks[first]
        if (
            len(touched) == 1
            and isinstance(anchor, SectionPlan)
            and all(target.isdisjoint(_ref_ids(ref, self.count)) for ref in anchor.title or [])
        ):
            children = self.replace(anchor.blocks, target, table)
            return (
                None
                if children is None
                else [
                    *blocks[:first],
                    anchor.model_copy(update={"blocks": children}),
                    *blocks[first + 1 :],
                ]
            )
        result: list[BlockPlan] = [*blocks[:first], table]
        selected = set(touched)
        captions: list[int] = []
        for i, block in enumerate(blocks[first:], start=first):
            kept = self.strip_block(block, target) if i in selected else block
            if kept is not None:
                if i in selected and isinstance(kept, TablePlan) and not _has_cells(kept):
                    captions.append(len(result))
                result.append(kept)
        # 칸을 모두 격자 표에 넘긴 옛 표에는 캡션만 남는다. 빈 표를 남기지 말고 캡션을 새 표의 제목으로 옮긴다.
        if len(captions) == 1:
            old = result.pop(captions[0])
            assert isinstance(old, TablePlan)
            result[first] = table.model_copy(update={"title": old.title})
        return result

    def insert(self, blocks: list[BlockPlan], left: int, right: int, table: TablePlan) -> list[BlockPlan] | None:
        groups = [set(self.ids(block)) for block in blocks]
        before = [index for index, refs in enumerate(groups) if left in refs]
        after = [index for index, refs in enumerate(groups) if right in refs]
        if len(before) != 1 or len(after) != 1:
            return None
        a, b = before[0], after[0]
        node = blocks[a]
        if a == b and isinstance(node, SectionPlan):
            children = self.insert(node.blocks, left, right, table)
            return (
                None
                if children is None
                else [
                    *blocks[:a],
                    node.model_copy(update={"blocks": children}),
                    *blocks[a + 1 :],
                ]
            )
        if a >= b or any(groups[a + 1 : b]) or max(groups[a]) > left or min(groups[b]) < right:
            return None
        return [*blocks[:b], table, *blocks[b:]]

    def insert_unreferenced(
        self, plan: DocumentPlan, candidate: "GeometryTable", target: set[int], table: TablePlan
    ) -> list[BlockPlan] | None:
        if candidate.mode != "rectangular" or any(self.source[n].page != candidate.page for n in target):
            return None
        used = {n for n in self.ids(plan) if self.source[n].page == candidate.page}
        first, last = min(target), max(target)
        before = max((n for n in used if n < first), default=None)
        after = min((n for n in used if n > last), default=None)
        ys = [self.source[n].y for n in target]
        if before is None or after is None or not self.source[before].y < min(ys) <= max(ys) < self.source[after].y:
            return None
        return self.insert(plan.blocks, before, after, table)


def _refs(row: list[list[int]]) -> list[Refs]:
    return [[str(n) for n in cell] for cell in row]


def _has_cells(table: TablePlan) -> bool:
    return any(table.headers) or any(any(row) for row in table.rows)


def apply_geometry_tables(
    plan: DocumentPlan, segments: list[Segment], candidates: list["GeometryTable"]
) -> DocumentPlan:
    """기존 표가 틀렸으면 교체하고, 빠졌으면 검증 가능한 같은 쪽 경계에만 넣는다."""
    refs = _References(segments)
    for candidate in candidates:
        cells = [n for row in (candidate.headers, *candidate.rows) for cell in row for n in cell]
        target = set(cells) | set(candidate.marks)
        if not cells or len(target) != len(cells) + len(candidate.marks) or not refs.source.keys() >= target:
            continue
        if any(not target.isdisjoint(_ref_ids(ref, refs.count)) for ref in plan.title or []):
            continue
        table = TablePlan(type="table", headers=_refs(candidate.headers), rows=[_refs(row) for row in candidate.rows])
        updated = refs.replace(plan.blocks, target, table)
        if updated is None and not candidate.marks:
            updated = refs.insert_unreferenced(plan, candidate, target, table)
        if updated is None:
            continue
        if candidate.marks:
            updated.append(ArtifactPlan(type="artifact", role="decoration", segments=list(map(str, candidate.marks))))
        revised = plan.model_copy(update={"blocks": updated})
        after = list(refs.ids(revised))
        if Counter(n for n in after if n in target) != Counter(target):
            continue
        if [n for n in after if n not in target] != [n for n in refs.ids(plan) if n not in target]:
            continue
        plan = revised
    return plan
