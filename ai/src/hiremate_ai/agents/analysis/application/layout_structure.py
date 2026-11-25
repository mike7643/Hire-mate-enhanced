"""PDF 파서의 위치, 글자 모양으로 근거가 확실한 문서 구조만 복구한다."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from itertools import pairwise
from statistics import median
from typing import TYPE_CHECKING

from hiremate_ai.agents.analysis.application.layout_outline import refine_outline
from hiremate_ai.contract import (
    DocumentList,
    DocumentListItem,
    DocumentSection,
    DocumentType,
    DocumentVisualNotice,
    NormalizedDocument,
)

if TYPE_CHECKING:
    from hiremate_ai.agents.analysis.application.normalize import (
        BlockPlan,
        DocumentPlan,
        ItemPlan,
        ParagraphPlan,
        Segment,
    )
    from hiremate_ai.contract import (
        DocumentBlock,
        DocumentNormalizationRequest,
        PdfBox,
        PdfGridLine,
        PdfLayout,
        PdfPage,
    )

_NUMBERED = re.compile(r"([1-9][0-9]?)[.)]\s+[^.!?]{2,22}")
_BRACKETED = re.compile(r"\[([^\[\]]{2,65})\]")
_EXCLUDED_CAPTION = re.compile(r"증명\s*사진|얼굴\s*사진|프로필\s*사진|인물\s*사진|배경\s*장식|^표지\b")
_FIGURE_LABEL = re.compile(
    r"^(?:(?:그림|도표|도식|차트|그래프|화면|figure|fig\.?|chart)\s*\d+\b"
    r"|(?:막대|선|꺾은선|산점도|흐름도|구성도|시퀀스\s*다이어그램)\s*[:：])",
    re.IGNORECASE,
)
_TREND = re.compile(r"증가|감소|상승|하락|최저|최고|낮아|높아|높지만|떨어|줄어|늘어|감속|격차")
_MEASURE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|주\s*차|개월|배|건|명|초)")
# 목록 머리표 그림: 크기(pt), 머리표 오른쪽 끝과 글 사이 최대 간격, 항목 안 줄 간격 한도(글자 크기 배수).
_MARK_MIN = 1.0
_MARK_MAX = 8.0
_MARK_GAP = 12.0
_ITEM_LINE_GAP = 2.0
# 제목 앞에 붙이는 기호와 그런 제목의 최대 글자 수. 줄표, 별표, 참고 표시(※)는 목록과 주석에 흔해 넣지 않는다.
_HEADING_MARKS = frozenset("■□◆◇●○▶►▷★☆◎▣◈◼◻•")
# 소제목을 감싸는 괄호. 둥근 괄호는 본문에서 흔해 넣지 않는다.
_BRACKET_PAIRS = {"〈": "〉", "<": ">", "《": "》", "「": "」", "『": "』", "【": "】", "[": "]"}
# 소제목 왼쪽 같은 줄에 두는 칸 이름표(성장과정, 지원동기)의 최대 글자 수.
_LABEL_CHARS = 12
_BRACKET_LINE = re.compile(r"\s*([〈<《「『【\[])([^〈〉<>《》「」『』【】\[\]\n]{1,60})([〉>》」』】\]])\s*")
_MARK_HEADING_CHARS = 30
# 같은 줄로 보는 세로 차이와, 같은 단으로 보는 들여쓰기 한도(pt).
_SAME_LINE_GAP = 1.0
_HEADING_INDENT = 24.0
# 같은 기호 줄이 글자 크기의 이 배수 안에서 위아래로 붙어 있으면 목록 항목으로 본다(줄 간격 1.2~2배).
_LIST_GAP = 2.5
# 쪽 끝 줄이 단 너비의 이 비율을 넘고 문장 끝 부호 없이 끝나면 문단이 다음 쪽으로 이어진다고 본다.
_OPEN_LINE = 0.75
_SENTENCE_END = frozenset(".!?。")
# 섹션 제목이 이보다 길고(공백 뺀 글자 수) 문장 끝 부호로 끝나면 제목이 아니라 문장으로 본다.
_PROSE_TITLE_CHARS = 40
_SENTENCE_BREAK = re.compile(r"[.!?。]\s+\S")
# 시각 자료 안내는 모델 문장 대신 종류별로 고정한 문장만 쓴다.
_NOTICE_TEXT = {
    "image": "원본 PDF {pages}쪽에 이미지가 있습니다. 내용은 원본에서 확인해 주세요.",
    "chart": "원본 PDF {pages}쪽에 차트가 있습니다. 수치는 원본에서 확인해 주세요.",
    "diagram": "원본 PDF {pages}쪽에 도식이 있습니다. 구성은 원본에서 확인해 주세요.",
    "unknown": "원본 PDF {pages}쪽의 시각 자료를 확인해 주세요.",
}


def visual_notice_message(visual_type: str, pages: list[int]) -> str:
    return _NOTICE_TEXT.get(visual_type, _NOTICE_TEXT["unknown"]).format(pages=", ".join(map(str, pages)))


def revise_structure(
    plan: DocumentPlan,
    segments: list[Segment],
    request: DocumentNormalizationRequest,
    *,
    released: list[list[int]] | None = None,
) -> DocumentPlan:
    """원문 번호 계획을 복사해, 구조를 확인할 수 있는 부분만 고친다.

    released는 모델이 잘못 숨겨 계획에서 뺀 조각 묶음이다. 읽는 순서 자리에 다시 넣는다.
    """
    source = {s.id: s for s in segments}
    by_page = _segments_by_page(segments)
    groups = _candidate_groups(request.pages, by_page)
    original = plan
    cleaned, stray = _release_stray_cells(plan.blocks, source)
    if cleaned != plan.blocks:
        plan = plan.model_copy(update={"blocks": cleaned})
    released = sorted([*(released or []), *stray])
    if released:
        plan = plan.model_copy(update={"blocks": _place_released(plan.blocks, released, groups, source)})
    used = set(_ids(_plan_refs(plan.title, plan.blocks), source))
    lifted = _lift_same_style(plan.blocks, source)
    joined = _join_continued_titles(lifted, groups, source, by_page)
    realigned = _realign_titles(joined, groups, source, used)
    if realigned != plan.blocks:
        plan = plan.model_copy(update={"blocks": realigned})
    blocks = plan.blocks
    if request.document_type == DocumentType.COVER_LETTER:
        blocks = _repair_forms(plan, request.pages, by_page, source)
        blocks = _recover_questions(blocks, plan.title, segments, source)
    elif request.document_type == DocumentType.PORTFOLIO:
        blocks = _repair_captions(plan, request.pages, by_page, source)
    # 머리표 그림은 문서 종류와 상관없는 증거라 모든 서류에 쓴다.
    items = _bullet_items(request.pages, by_page)
    if items:
        blocks = _apply_bullets(blocks, items, source)
    title, blocks = _document_title(plan.title, blocks, segments, groups, source)
    blocks = _title_untitled_sections(blocks, groups, source)
    blocks = _demote_prose_titles(blocks, source)
    brackets = {group[0] for group in _bracket_headings(by_page)}
    used = set(_ids(_plan_refs(title, blocks), source))
    blocks = _join_page_continuations(blocks, set(groups) - brackets, brackets, used, source, by_page)
    if brackets:
        blocks = _bracket_sections(blocks, brackets, source)
    blocks = refine_outline(blocks, source, request.pages)
    if title == original.title and blocks == original.blocks:
        return original
    return original.model_copy(update={"title": title, "blocks": blocks})


def _release_stray_cells(
    blocks: list[BlockPlan], source: dict[int, Segment]
) -> tuple[list[BlockPlan], list[list[int]]]:
    """표 칸에서, 표 첫 조각과 그 칸 사이에 다른 섹션 제목이 끼어 있는 조각을 뺀다.

    표는 다른 섹션 제목을 건너뛰어 이어지지 않는다. 그런 칸은 모델이 다른 쪽이나 다른 섹션의 줄을 넣은 것이다. 뺀
    조각을 계획의 다른 블록이 쓰면 그 블록에 맡기고, 아무도 쓰지 않으면 읽는 자리로 되돌리도록 돌려준다. 칸이 모두 빠진
    행은 지운다.
    """
    titles = [n for refs in _section_titles(blocks) for n in sorted(_ids([refs], source))[:1]]
    removed: set[int] = set()

    def clean(entries: list[BlockPlan], ancestors: frozenset[int]) -> list[BlockPlan]:
        result: list[BlockPlan] = []
        for block in entries:
            if block.type == "section":
                own = frozenset(_ids([block.title or []], source))
                children = clean(block.blocks, ancestors | own)
                result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
                continue
            if block.type != "table":
                result.append(block)
                continue
            first = min(_ids(_block_refs(block), source), default=None)
            foreign = [t for t in titles if first is not None and t > first and t not in ancestors]
            if not foreign:
                result.append(block)
                continue
            cut = min(foreign)
            rows: list[list[list[str]]] = []
            for row in block.rows:
                cells: list[list[str]] = []
                for cell in row:
                    ids = [n for ref in cell for n in _numbers(ref, source)]
                    if all(n < cut for n in ids):
                        cells.append(cell)
                        continue
                    cells.append([str(n) for n in ids if n < cut])
                    removed.update(n for n in ids if n >= cut)
                if any(cells):
                    rows.append(cells)
            result.append(block if rows == block.rows else block.model_copy(update={"rows": rows}))
        return result

    cleaned = clean(blocks, frozenset())
    if not removed:
        return blocks, []
    owned = set(_ids(_plan_refs(None, cleaned), source))
    runs: list[list[int]] = []
    for number in sorted(removed - owned):
        if runs and runs[-1][-1] == number - 1 and source[number - 1].page == source[number].page:
            runs[-1].append(number)
        else:
            runs.append([number])
    return cleaned, runs


_Style = tuple[float, str | None, float, str | None]


def _title_style(title: list[str] | None, source: dict[int, Segment]) -> _Style | None:
    """제목 첫 줄의 글자 크기, 색, 왼쪽 끝, 감싼 괄호. 괄호 소제목은 모양이 같아도 번호 제목과 다른 단계다."""
    ids = sorted(_ids([title or []], source))
    if not ids or source[ids[0]].size is None:
        return None
    head = source[ids[0]]
    return head.size or 0.0, head.color, head.x, bracket_kind(head.text)


def _same_style(left: _Style | None, right: _Style | None) -> bool:
    return (
        left is not None
        and right is not None
        and left[:2] == right[:2]
        and abs(left[2] - right[2]) <= 2
        and left[3] == right[3]
    )


def bracket_kind(text: str) -> str | None:
    """줄 전체가 짝 맞는 괄호 하나로 감싸여 있고 안에 글자가 있으면 그 여는 괄호를 돌려준다."""
    match = _BRACKET_LINE.fullmatch(text)
    if match is None or _BRACKET_PAIRS[match[1]] != match[3] or not any(char.isalpha() for char in match[2]):
        return None
    return match[1]


def _lift_same_style(blocks: list[BlockPlan], source: dict[int, Segment]) -> list[BlockPlan]:
    """섹션 바로 아래에 그 섹션 제목과 글자 크기, 색, 왼쪽 끝이 같은 제목의 섹션이 있으면 형제로 올린다.

    같은 모양의 제목은 같은 단계다. 올린 섹션은 원래 부모에서 그 뒤에 있던 블록을 이어 받는다. 읽는 순서상 그 블록은
    올린 제목 아래에 있기 때문이다.
    """
    result: list[BlockPlan] = []
    for block in blocks:
        if block.type != "section":
            result.append(block)
            continue
        children = _lift_same_style(block.blocks, source)
        current = block if children == block.blocks else block.model_copy(update={"blocks": children})
        style = _title_style(current.title, source)
        while True:
            found = next(
                (
                    (i, child)
                    for i, child in enumerate(current.blocks)
                    if child.type == "section" and _same_style(_title_style(child.title, source), style)
                ),
                None,
            )
            if found is None:
                break
            index, child = found
            result.append(current.model_copy(update={"blocks": current.blocks[:index]}))
            current = child.model_copy(update={"blocks": [*child.blocks, *current.blocks[index + 1 :]]})
        result.append(current)
    return result


def _join_continued_titles(
    blocks: list[BlockPlan], groups: dict[int, list[int]], source: dict[int, Segment], by_page: dict[int, list[Segment]]
) -> list[BlockPlan]:
    """모델이 본문이 이어지는 줄을 섹션 제목으로 썼으면, 그 섹션을 앞 내용 뒤에 붙이고 끊긴 문단을 다시 잇는다.

    이어지는 줄의 증거: 바로 앞 번호 줄과 글자 크기, 색, 왼쪽 끝이 같고, 앞 줄이 오른쪽 끝까지 차 있으며, 줄 간격이
    본문과 같거나(같은 쪽) 앞 줄이 쪽의 마지막 줄이고 이 줄이 다음 쪽 첫 줄이다. 문서의 다른 제목이 글자 모양으로
    구분될 때만 쓴다. 그래야 본문과 모양이 같은 줄을 제목이 아니라고 볼 수 있다.
    """
    heads = {n for group in groups.values() for n in group}
    titles = [n for refs in _section_titles(blocks) for n in sorted(_ids([refs], source))[:1]]
    if not any(n in heads for n in titles):
        return blocks
    edges: dict[tuple[int, int], float] = {}
    for page, lines in by_page.items():
        for line in lines:
            key = (page, round(line.x))
            edges[key] = max(edges.get(key, 0.0), line.end_x)

    def continues(number: int) -> bool:
        line, previous = source[number], source.get(number - 1)
        if previous is None or number in heads or line.size is None:
            return False
        if (previous.size, previous.color) != (line.size, line.color) or abs(previous.x - line.x) > 2:
            return False
        if previous.end_x < edges[(previous.page, round(previous.x))] - 3 * line.size:
            return False
        if previous.page == line.page:
            return 0 < line.y - previous.y <= 1.7 * line.size
        return (
            line.page == previous.page + 1
            and line.y == min(s.y for s in by_page[line.page])
            and previous.y == max(s.y for s in by_page[previous.page])
        )

    return _join_continued(blocks, continues, source)


def _section_titles(blocks: list[BlockPlan]) -> Iterator[list[str]]:
    for block in blocks:
        if block.type == "section":
            if block.title:
                yield block.title
            yield from _section_titles(block.blocks)


def _join_continued(
    blocks: list[BlockPlan], continues: Callable[[int], bool], source: dict[int, Segment]
) -> list[BlockPlan]:
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan

    result: list[BlockPlan] = []
    for block in blocks:
        if block.type != "section":
            result.append(block)
            continue
        children = _join_continued(block.blocks, continues, source)
        title = sorted(_ids([block.title or []], source))
        if not title or not continues(title[0]):
            result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
            continue
        first = children[0] if children else None
        if first is not None and first.type == "paragraph" and set(title) <= set(_ids([first.segments], source)):
            body = children
        else:
            body = [ParagraphPlan(type="paragraph", segments=block.title or []), *children]
        if result and result[-1].type == "section":
            holder = result[-1]
            result[-1] = holder.model_copy(update={"blocks": _append_continued(holder.blocks, body, title[0], source)})
        else:
            result = _append_continued(result, body, title[0], source)
    return result


def _append_continued(
    blocks: list[BlockPlan], body: list[BlockPlan], start: int, source: dict[int, Segment]
) -> list[BlockPlan]:
    """이어지는 내용을 붙인다. 앞 문단이 바로 앞 번호에서 끝나면 첫 문단을 그 문단에 잇는다."""
    last, lead = (blocks[-1] if blocks else None), body[0]
    if (
        last is not None
        and last.type == "paragraph"
        and lead.type == "paragraph"
        and not last.keep_lines
        and max(_ids([last.segments], source), default=0) == start - 1
    ):
        joined = last.model_copy(update={"segments": [*last.segments, *lead.segments]})
        return [*blocks[:-1], joined, *body[1:]]
    return [*blocks, *body]


def _realign_titles(
    blocks: list[BlockPlan], groups: dict[int, list[int]], source: dict[int, Segment], used: set[int]
) -> list[BlockPlan]:
    """이어진 형제 섹션에서 제목만 읽는 순서를 어겼으면, 각 내용 블록을 번호가 바로 앞선 제목의 섹션으로 옮긴다.

    조각 번호가 읽는 순서다. 내용 블록끼리는 모델이 둔 순서대로 번호가 커질 때만 고친다. 내용 순서까지 바뀌었으면
    (사이드바를 문서 끝에 둔 것처럼) 모델 판단으로 보고 두지 않는다. 다시 나눌 때 계획 어디에도 없는 제목 후보 줄이
    섹션 사이에 있으면 그 줄도 제목으로 쓴다.
    """
    nested: list[BlockPlan] = []
    for block in blocks:
        if block.type == "section":
            children = _realign_titles(block.blocks, groups, source, used)
            block = block if children == block.blocks else block.model_copy(update={"blocks": children})
        nested.append(block)
    result: list[BlockPlan] = []
    start = 0
    while start < len(nested):
        end = start
        while end < len(nested) and nested[end].type == "section":
            end += 1
        if end - start > 1:
            result.extend(_realign_run(nested[start:end], groups, source, used))
            start = end
        else:
            result.append(nested[start])
            start += 1
    return result


def _realign_run(
    run: list[BlockPlan], groups: dict[int, list[int]], source: dict[int, Segment], used: set[int]
) -> list[BlockPlan]:
    from hiremate_ai.agents.analysis.application.normalize import SectionPlan

    titles = {n for section in run if section.type == "section" for n in _ids([section.title or []], source)}
    order: list[int] = []
    # 내용 블록 묶음: (첫 번호, 끝 번호, 블록들). 번호 없는 블록(시각 자료 안내)은 바로 앞 묶음에 붙여 함께 옮긴다.
    units: list[tuple[int, int, list[BlockPlan]]] = []
    splits: dict[int, list[str]] = {}
    # 어긋남을 찾으려고 제목과 묶음이 계획에서 몇 번째 섹션에 있었는지 적는다.
    heads: list[tuple[int, int]] = []
    homes: list[int] = []
    for position, section in enumerate(run):
        if section.type != "section":
            return run
        own = sorted(_ids([section.title or []], source))
        if own:
            order.append(own[0])
            splits.setdefault(own[0], section.title or [])
            heads.append((position, own[0]))
        for child in section.blocks:
            ids = [n for n in _ids(_block_refs(child), source) if n not in titles]
            if ids:
                units.append((min(ids), max(ids), [child]))
                homes.append(position)
                order.append(min(ids))
            elif units:
                units[-1][2].append(child)
            else:
                return run
    if not units or all(a < b for a, b in pairwise(order)):
        return run
    if any(left[1] >= right[0] for left, right in pairwise(units)):
        return run
    for (first, last, _members), home in zip(units, homes, strict=True):
        lines = [source[n] for n in range(first, last + 1) if n not in titles]
        for position, head in heads:
            ahead = (position == home and head > first) or (position > home and head < first)
            label = source[head]
            # 어긋난 제목이 내용과 다른 단에 있으면 왼쪽 칸 이름표 양식이다. 이름표는 답 옆 가운데에 있어
            # 번호가 답 첫 줄 뒤에 올 수 있으므로 번호만으로 옮기지 않는다.
            if ahead and (label.end_x <= min(s.x for s in lines) or label.x >= max(s.end_x for s in lines)):
                return run
    low, high = units[0][0], units[-1][1]
    for first, group in groups.items():
        if low < first < high and not used.intersection(group):
            splits.setdefault(first, [str(n) for n in group])
    lead: list[BlockPlan] = []
    buckets: dict[int, list[BlockPlan]] = {first: [] for first in splits}
    for first, _last, members in units:
        owner = max((split for split in splits if split < first), default=None)
        (lead if owner is None else buckets[owner]).extend(members)
    rebuilt: list[BlockPlan] = [
        SectionPlan(type="section", title=splits[first], blocks=buckets[first]) for first in sorted(splits)
    ]
    if not lead:
        return rebuilt
    if run[0].type == "section" and not run[0].title:
        return [SectionPlan(type="section", title=None, blocks=lead), *rebuilt]
    return [*lead, *rebuilt]


def _place_released(
    blocks: list[BlockPlan], released: list[list[int]], groups: dict[int, list[int]], source: dict[int, Segment]
) -> list[BlockPlan]:
    """되살릴 조각 묶음을 바로 앞 번호 조각이 있는 블록 뒤에 넣는다. 앞 조각이 섹션 제목이면 그 섹션 맨 앞이다.

    번호가 이어지는 묶음끼리는 한 덩어리로 보고, 그 안에서 제목 후보로 시작하는 묶음마다 제목 없는 섹션을 연다.
    제목은 뒤의 _title_untitled_sections가, 목록은 _apply_bullets가 글자 모양과 머리표로 정한다.
    """
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan, SectionPlan

    placed = {n for group in released for n in group}
    owned = {n for n in _ids(_plan_refs(None, blocks), source)}
    runs: list[list[list[int]]] = []
    for group in sorted(released):
        last = runs[-1][-1][-1] if runs else None
        if last is not None and group[0] == last + 1 and source[group[0]].page == source[last].page:
            runs[-1].append(group)
        else:
            runs.append([group])
    by_anchor: dict[int | None, list[BlockPlan]] = {}
    for run in runs:
        made: list[BlockPlan] = []
        for group in run:
            paragraph = ParagraphPlan(type="paragraph", segments=[str(n) for n in group])
            heading = groups.get(group[0])
            if heading is not None and group[: len(heading)] == heading:
                made.append(SectionPlan(type="section", title=None, blocks=[paragraph]))
            elif made and made[-1].type == "section":
                section = made[-1]
                made[-1] = section.model_copy(update={"blocks": [*section.blocks, paragraph]})
            else:
                made.append(paragraph)
        anchor = next((n for n in range(run[0][0] - 1, 0, -1) if n in owned and n not in placed), None)
        by_anchor.setdefault(anchor, []).extend(made)
    result = _insert_after(blocks, by_anchor, source)
    return [*by_anchor.pop(None, []), *result]


def _insert_after(
    blocks: list[BlockPlan], by_anchor: dict[int | None, list[BlockPlan]], source: dict[int, Segment]
) -> list[BlockPlan]:
    result: list[BlockPlan] = []
    for block in blocks:
        if block.type == "section":
            front = [made for n in _ids([block.title or []], source) for made in by_anchor.pop(n, [])]
            children = _insert_after(block.blocks, by_anchor, source)
            changed = front or children != block.blocks
            result.append(block.model_copy(update={"blocks": [*front, *children]}) if changed else block)
            continue
        result.append(block)
        for n in sorted(_ids(_block_refs(block), source)):
            result.extend(by_anchor.pop(n, []))
    return result


def _document_title(
    title: list[str] | None,
    blocks: list[BlockPlan],
    segments: list[Segment],
    groups: dict[int, list[int]],
    source: dict[int, Segment],
) -> tuple[list[str] | None, list[BlockPlan]]:
    """모델이 문서 제목을 비웠을 때, 첫 쪽에서 가장 큰 글자로 쓴 제목 후보 한 줄을 문서 제목으로 올린다.

    그 줄이 문단 하나를 통째로 이루고, 맨 앞이나 제목 없는 맨 앞 섹션의 첫 블록일 때만 옮긴다.
    """
    first_page = [s for s in segments if s.page == segments[0].page and s.size] if segments else []
    if title or not first_page:
        return title, blocks
    largest = max(s.size or 0 for s in first_page)
    tops = [s.id for s in first_page if s.size == largest]
    group = groups.get(tops[0])
    if group is None or group != tops:
        return title, blocks

    def whole(block: BlockPlan) -> bool:
        return block.type == "paragraph" and not block.keep_lines and list(_ids([block.segments], source)) == group

    refs = [str(n) for n in group]
    for index, block in enumerate(blocks):
        if whole(block):
            return refs, [*blocks[:index], *blocks[index + 1 :]]
        if block.type == "section" and not block.title and block.blocks and whole(block.blocks[0]):
            rest = block.blocks[1:]
            kept = [block.model_copy(update={"blocks": rest})] if rest else []
            return refs, [*blocks[:index], *kept, *blocks[index + 1 :]]
    return title, blocks


def _candidate_groups(pages: list[PdfPage], by_page: dict[int, list[Segment]]) -> dict[int, list[int]]:
    """코드 보정에 쓰는 제목 후보. 모델에 보내는 후보에 기호 제목 줄을 더한다(모델 요청은 바꾸지 않는다)."""
    from hiremate_ai.agents.analysis.application.normalize import _heading_candidates

    groups = {group[0]: group for group in _heading_candidates(by_page, bulleted_lines(pages, by_page))}
    for group in [*_mark_headings(by_page), *_bracket_headings(by_page)]:
        if not any(group[0] in existing for existing in groups.values()):
            groups[group[0]] = group
    return groups


def _bracket_headings(by_page: dict[int, list[Segment]]) -> list[list[int]]:
    """같은 괄호로 감싼 짧은 한 줄이 같은 들여쓰기로 두 번 이상 본문 앞에 나오면 소제목 줄로 본다.

    〈소제목〉, [소제목]처럼 본문과 글자 모양이 같아 글자 모양으로는 찾지 못하는 소제목이다. 아래(쪽 맨 아래 줄이면
    다음 쪽 첫머리)에 같은 단의 줄이 없으면(본문이 이어지지 않으면) 소제목으로 보지 않는다.
    """
    found: list[tuple[str, Segment]] = []
    numbers = sorted(by_page)
    ordered = {n: sorted((s for s in by_page[n] if s.x < s.end_x), key=lambda s: (s.y, s.x)) for n in numbers}
    for position, number in enumerate(numbers):
        lines = ordered[number]
        following = ordered[numbers[position + 1]] if position + 1 < len(numbers) else []
        for index, line in enumerate(lines):
            kind = bracket_kind(line.text)
            if kind is None:
                continue
            below = _column_neighbour(iter(lines[index + 1 :]), line)
            if below is None:
                # 쪽 맨 아래 소제목은 본문이 다음 쪽 같은 단에서 시작한다.
                below = next((s for s in following if abs(s.x - line.x) <= _HEADING_INDENT), None)
            if below is None:
                continue
            found.append((kind, line))
    return [
        [line.id]
        for kind, line in found
        if sum(1 for other_kind, other in found if other_kind == kind and abs(other.x - line.x) <= 2) >= 2
    ]


def _bracket_sections(blocks: list[BlockPlan], heads: set[int], source: dict[int, Segment]) -> list[BlockPlan]:
    """소제목 줄로 시작하는 문단을 그 소제목의 섹션으로 세운다. 섹션은 다음 소제목이나 섹션 앞까지 이어진다.

    소제목 줄과 본문이 한 문단에 붙어 있으면 소제목만 떼어 제목으로 쓴다. 모델이 이미 소제목으로 만든 섹션(안에
    섹션이 없는 것)은 같은 방법으로 다시 세워 형제로 둔다. 문단에서 떼어 낸 소제목 뒤에 내용이 없으면 문단으로
    남긴다. 모델이 만든 섹션은 비어 있어도 섹션으로 둔다. 모델이 빠뜨린 본문 조각은 조립 때 제목 뒤에 되살아난다.
    """
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan, SectionPlan

    result: list[BlockPlan] = []
    open_title: list[str] | None = None
    body: list[BlockPlan] = []
    kept = False

    def close() -> None:
        nonlocal open_title, body, kept
        if open_title is not None:
            if body or kept:
                result.append(SectionPlan(type="section", title=open_title, blocks=body))
            else:
                result.append(ParagraphPlan(type="paragraph", segments=open_title))
        open_title, body, kept = None, [], False

    def place(block: BlockPlan) -> None:
        (body if open_title is not None else result).append(block)

    def lead(refs: list[str] | None) -> int | None:
        ids = list(_ids([refs or []], source))
        return ids[0] if ids else None

    def title_end(ids: list[int]) -> int:
        """문단 앞에서 제목으로 뗄 조각 수. 소제목 줄, 또는 같은 줄 왼쪽 칸 이름표와 그 옆 소제목 줄이다."""
        if ids[0] in heads:
            return 1
        if len(ids) > 1 and ids[1] in heads and ids[1] == ids[0] + 1:
            label, head = source[ids[0]], source[ids[1]]
            if (
                label.page == head.page
                and abs(label.y - head.y) <= _SAME_LINE_GAP
                and label.end_x <= head.x
                and 0 < len(label.text.strip()) <= _LABEL_CHARS
            ):
                return 2
        return 0

    queue: list[BlockPlan] = list(blocks)
    while queue:
        block = queue.pop(0)
        if block.type == "section":
            inner = [b for b in block.blocks if b.type == "section"]
            if lead(block.title) in heads and not inner and block.title is not None:
                close()
                open_title, body, kept = list(block.title), [], True
                # 모델이 만든 소제목 섹션 안의 문단도 소제목으로 시작할 수 있으므로 같은 줄에서 다시 읽는다.
                queue[:0] = list(block.blocks)
                continue
            children = _bracket_sections(block.blocks, heads, source)
            close()
            result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
            continue
        if block.type == "paragraph" and not block.keep_lines:
            ids = list(_ids([block.segments], source))
            taken = set(_ids([open_title or []], source))
            if ids and ids[0] in taken:
                # 모델이 섹션 제목 조각을 첫 문단에도 넣었다. 제목에 쓴 조각만 빼고 그 섹션 본문으로 둔다.
                rest = [n for n in ids if n not in taken]
                if rest:
                    body.append(ParagraphPlan(type="paragraph", segments=_compact(rest)))
                continue
            end = title_end(ids) if ids else 0
            if end:
                close()
                open_title = [str(n) for n in ids[:end]]
                if len(ids) > end:
                    body.append(ParagraphPlan(type="paragraph", segments=_compact(ids[end:])))
                continue
        place(block)
    close()
    return result


def _demote_prose_titles(blocks: list[BlockPlan], source: dict[int, Segment]) -> list[BlockPlan]:
    """문장으로 쓴 긴 섹션 제목(공백 빼고 40자 넘고, 문장 끝 부호로 끝나거나 중간에 문장이 끝나는)은 본문이다.

    모델이 쪽 묶음 첫 줄이나 안내 문장을 제목으로 쓴 경우다. 글자나 숫자가 하나도 없는 제목(코드 주석 \"/**\" 같은)도
    본문으로 돌린다. 제목 앞머리에 짧은 이름 조각(\"[경력]\" 같은)이 따로 있으면
    그 조각만 제목으로 남기고 나머지를 첫 문단으로 내린다. 이름 조각이 없으면 섹션을 풀어 제목 줄을 문단으로 두고
    내용은 그 뒤에 그대로 잇는다.
    """
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan

    result: list[BlockPlan] = []
    for block in blocks:
        if block.type != "section":
            result.append(block)
            continue
        children = _demote_prose_titles(block.blocks, source)
        ids = list(_ids([block.title or []], source))
        text = " ".join(source[n].text.strip() for n in ids)
        compact = "".join(text.split())
        symbolic = bool(compact) and not any(char.isalnum() for char in compact)
        prose = len(compact) > _PROSE_TITLE_CHARS and (
            compact[-1:] in _SENTENCE_END or _SENTENCE_BREAK.search(text) is not None
        )
        if not (symbolic or prose):
            result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
            continue
        head = ids[0]
        name = "".join(source[head].text.split())
        if len(ids) > 1 and 0 < len(name) <= _LABEL_CHARS and name[-1:] not in _SENTENCE_END:
            lead = ParagraphPlan(type="paragraph", segments=_compact(ids[1:]))
            result.append(block.model_copy(update={"title": [str(head)], "blocks": [lead, *children]}))
            continue
        result.extend([ParagraphPlan(type="paragraph", segments=_compact(ids)), *children])
    return result


def _join_page_continuations(
    blocks: list[BlockPlan],
    headings: set[int],
    heads: set[int],
    used: set[int],
    source: dict[int, Segment],
    by_page: dict[int, list[Segment]],
) -> list[BlockPlan]:
    """모델이 쪽 경계에서 제목 없이 연 섹션의 앞부분이 앞 섹션 글의 이어짐이면 앞 섹션으로 옮긴다.

    쪽마다, 또는 쪽 묶음마다 따로 답한 모델은 쪽이 바뀌는 자리에서 제목 없는 섹션을 새로 열곤 한다. 앞 섹션 마지막
    줄이 단 오른쪽 끝까지 차 있고(문단이 끝나지 않았고) 제목 없는 섹션이 다음 쪽의 바로 다음 조각에서 시작하면, 그
    앞부분은 앞 섹션 글이다. 옮기는 범위는 처음 나오는 섹션이나 제목 후보 줄 앞까지다. 괄호 소제목 줄은 함께 옮겨
    소제목 규칙이 그 자리에서 세우게 한다. 앞 섹션 안에서는 맨 끝 섹션을 따라 내려가되 괄호 소제목 섹션 앞에서
    멈춘다(그 섹션은 소제목 규칙이 형제로 다시 세운다). 잇는 증거는 _continues에 있다.
    """
    result: list[BlockPlan] = []
    for block in blocks:
        if block.type == "section":
            children = _join_page_continuations(block.blocks, headings, heads, used, source, by_page)
            if children != block.blocks:
                block = block.model_copy(update={"blocks": children})
        previous = result[-1] if result else None
        if block.type == "section" and not block.title and previous is not None and previous.type == "section":
            leading: list[BlockPlan] = []
            for child in block.blocks:
                first = next(_ids(_block_refs(child), source), None)
                if child.type not in ("paragraph", "list") or first is None or first in headings:
                    break
                leading.append(child)
            if leading and _continues(previous, leading[0], used, source, by_page):
                result[-1] = _append_deep(previous, leading, heads, source)
                rest = block.blocks[len(leading) :]
                if not rest:
                    continue
                block = block.model_copy(update={"blocks": rest})
        result.append(block)
    return result


def _continues(
    previous: BlockPlan,
    block: BlockPlan,
    used: set[int],
    source: dict[int, Segment],
    by_page: dict[int, list[Segment]],
) -> bool:
    """block이 다음 쪽에서 previous의 마지막 줄을 잇는가.

    두 줄의 글자 크기, 색, 왼쪽 끝이 같고, 앞 줄이 그 단의 오른쪽 끝(글자 세 개 안)까지 차 있거나 단 너비의 3/4을
    넘고 문장 끝 부호 없이 끝난다. 사이 조각은 계획이 쓰지 않은 것(쪽 번호 같은)만 있다.
    """
    last_id = max(_ids(_block_refs(previous), source), default=None)
    first_id = min(_ids(_block_refs(block), source), default=None)
    if last_id is None or first_id is None or first_id <= last_id:
        return False
    if any(n in used for n in range(last_id + 1, first_id)):
        return False
    last, first = source[last_id], source[first_id]
    if first.page <= last.page or last.size is None:
        return False
    if (last.size, last.color) != (first.size, first.color) or abs(last.x - first.x) > 2:
        return False
    edge = max(s.end_x for s in by_page.get(last.page, []) if round(s.x) == round(last.x))
    line = sorted((s for s in by_page[last.page] if abs(s.y - last.y) <= _SAME_LINE_GAP), key=lambda s: s.x)
    line_end = line[-1].end_x
    if line_end >= edge - 3 * last.size:
        return True
    # 낱말 단위 줄바꿈은 줄 끝을 글자 서너 개만큼 남긴다. 줄이 단 너비의 3/4을 넘고 문장 끝 부호로 끝나지 않으면 잇는다.
    return line_end >= last.x + _OPEN_LINE * (edge - last.x) and line[-1].text.rstrip()[-1:] not in _SENTENCE_END


def _append_deep(section: BlockPlan, extra: list[BlockPlan], heads: set[int], source: dict[int, Segment]) -> BlockPlan:
    """섹션 맨 끝 섹션을 따라 내려가 extra를 붙인다. 괄호 소제목 섹션 앞에서는 멈춘다."""
    assert section.type == "section"
    last = section.blocks[-1] if section.blocks else None
    if last is not None and last.type == "section":
        lead = next(_ids([last.title or []], source), None)
        reopened = lead in heads and not any(child.type == "section" for child in last.blocks)
        if not reopened:
            deeper = _append_deep(last, extra, heads, source)
            return section.model_copy(update={"blocks": [*section.blocks[:-1], deeper]})
    return section.model_copy(update={"blocks": [*section.blocks, *extra]})


def _compact(ids: list[int]) -> list[str]:
    runs: list[list[int]] = []
    for number in ids:
        if runs and runs[-1][-1] == number - 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return [str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs]


def _mark_headings(by_page: dict[int, list[Segment]]) -> list[list[int]]:
    """같은 기호로 시작하는 짧은 한 줄이 같은 들여쓰기로 두 번 이상 본문 앞에 나오면 제목 줄로 본다.

    "■ 성장과정"처럼 본문보다 글자가 작아 글자 모양으로는 찾지 못하는 제목이다. 같은 단에서 줄 간격 안에 붙은 위나
    아래 줄도 같은 기호로 시작하면 목록 항목이라 뺀다. 바로 아래 줄이 없거나 제목보다 왼쪽에서
    시작하면(아래에 딸린 내용이 아니면) 제목으로 보지 않는다. 양식 이력서처럼 제목 아래 내용이 표 칸으로 들여 쓰여
    있어도 그 표가 제목에 딸린 내용이다.
    """
    found: list[tuple[str, Segment]] = []
    for segments in by_page.values():
        lines = sorted((s for s in segments if s.x < s.end_x), key=lambda s: (s.y, s.x))
        for index, line in enumerate(lines):
            text = line.text.strip()
            mark, body = text[:1], text[1:].strip()
            if mark not in _HEADING_MARKS or not 0 < len(body) <= _MARK_HEADING_CHARS:
                continue
            if not any(char.isalpha() for char in body):
                continue
            above = _column_neighbour(reversed(lines[:index]), line)
            below = _column_neighbour(iter(lines[index + 1 :]), line)
            after = _next_line(iter(lines[index + 1 :]), line)
            if after is None or after.x < line.x - 2:
                continue
            # 목록 항목은 같은 기호 줄이 줄 간격 안에 붙어 있다. 사이에 다른 내용(표 칸)이 있으면 목록이 아니다.
            gap = _LIST_GAP * (line.size or 10.0)
            if any(
                n is not None and n.text.strip()[:1] == mark and abs(n.x - line.x) <= 2 and abs(n.y - line.y) <= gap
                for n in (above, below)
            ):
                continue
            found.append((mark, line))
    return [
        [line.id]
        for mark, line in found
        if sum(1 for other_mark, other in found if other_mark == mark and abs(other.x - line.x) <= 2) >= 2
    ]


def _next_line(lines: Iterator[Segment], line: Segment) -> Segment | None:
    """읽는 순서로 줄이 바뀐 가장 가까운 줄(단과 상관없이)."""
    return next((other for other in lines if abs(other.y - line.y) >= _SAME_LINE_GAP), None)


def _column_neighbour(lines: Iterator[Segment], line: Segment) -> Segment | None:
    """같은 단(들여쓰기 한도 안)에서 줄이 바뀐 가장 가까운 줄."""
    for other in lines:
        if abs(other.y - line.y) >= _SAME_LINE_GAP and abs(other.x - line.x) <= _HEADING_INDENT:
            return other
    return None


def _title_untitled_sections(
    blocks: list[BlockPlan], groups: dict[int, list[int]], source: dict[int, Segment]
) -> list[BlockPlan]:
    """모델이 제목 없이 나눈 섹션이 글자 모양으로 찾은 제목 후보 줄로 시작하면 그 줄을 섹션 제목으로 올린다.

    같은 줄 오른쪽에 붙은 조각(기간 같은)은 제목에 함께 넣는다. 모델이 단 제목, 섹션 밖의 문단, 줄을 지키라는
    문단은 건드리지 않는다. 제목을 빼고 남는 내용이 없으면 그대로 둔다.
    """
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan

    result: list[BlockPlan] = []
    for block in blocks:
        if block.type != "section":
            result.append(block)
            continue
        children = _title_untitled_sections(block.blocks, groups, source)
        first = children[0] if children else None
        if not block.title and first is not None and first.type == "paragraph" and not first.keep_lines:
            ids = list(_ids([first.segments], source))
            group = groups.get(ids[0]) if ids else None
            if group is not None and ids[: len(group)] == group:
                end = len(group)
                last = source[group[-1]]
                while end < len(ids) and _same_line_right(source[ids[end]], last):
                    last = source[ids[end]]
                    end += 1
                rest = [ParagraphPlan(type="paragraph", segments=[str(n) for n in ids[end:]])] if end < len(ids) else []
                body = [*rest, *children[1:]]
                if body:
                    result.append(block.model_copy(update={"title": [str(n) for n in ids[:end]], "blocks": body}))
                    continue
        result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
    return result


def _same_line_right(segment: Segment, previous: Segment) -> bool:
    return segment.page == previous.page and abs(segment.y - previous.y) < 1.5 and segment.x >= previous.end_x


def _marks(layout: PdfLayout) -> list[PdfBox]:
    """머리표로 볼 수 있는 작은 그림. 원이나 삼각형은 mark_boxes, 네모는 작은 정사각형 채움이다."""
    found = list(layout.mark_boxes)
    for rect in layout.fill_rects:
        width, height = rect.x2 - rect.x1, rect.y2 - rect.y1
        if (
            min(width, height) >= _MARK_MIN
            and max(width, height) <= _MARK_MAX
            and max(width, height) <= 2 * min(width, height)
        ):
            found.append(rect)
    return found


def _led_lines(layout: PdfLayout, lines: list[Segment]) -> dict[int, tuple[Segment, PdfBox]]:
    """머리표 그림이 바로 왼쪽에 있는 줄과 그 머리표."""
    led: dict[int, tuple[Segment, PdfBox]] = {}
    for mark in _marks(layout):
        middle = (mark.y1 + mark.y2) / 2
        candidates = [
            s
            for s in lines
            if mark.x2 <= s.x + 0.5
            and s.x - mark.x2 <= _MARK_GAP
            and s.y - (s.size or 10) <= middle <= s.y + 1
            # 앞 글자의 일부처럼 다른 조각과 겹치는 그림은 머리표가 아니다.
            and not any(
                o is not s and abs(o.y - s.y) < 1 and o.x - 0.5 < mark.x2 and mark.x1 < o.end_x + 0.5 for o in lines
            )
        ]
        if candidates:
            head = min(candidates, key=lambda s: s.x - mark.x2)
            led.setdefault(head.id, (head, mark))
    return led


def bulleted_lines(pages: list[PdfPage], by_page: dict[int, list[Segment]]) -> set[int]:
    """머리표 그림으로 시작하는 줄의 조각 번호."""
    found: set[int] = set()
    for page in pages:
        lines = [s for s in by_page.get(page.number, []) if s.x < s.end_x]
        if page.layout is not None and lines:
            found.update(_led_lines(page.layout, lines))
    return found


def _bullet_items(pages: list[PdfPage], by_page: dict[int, list[Segment]]) -> list[list[int]]:
    """머리표 그림이 줄 바로 왼쪽에 있는 줄마다 목록 항목 하나를 만든다. 항목은 같은 들여쓰기로 이어지는 줄까지다."""
    items: list[list[int]] = []
    for page in pages:
        lines = [s for s in by_page.get(page.number, []) if s.x < s.end_x]
        if page.layout is None or not lines:
            continue
        led = _led_lines(page.layout, lines)
        for head, mark in sorted(led.values(), key=lambda pair: (pair[0].y, pair[0].x)):
            item = [head]
            limit = _ITEM_LINE_GAP * (head.size or 10)
            for line in sorted((s for s in lines if abs(s.x - head.x) <= 2 and s.y > head.y + 1), key=lambda s: s.y):
                last = item[-1]
                blocked = any(last.y + 1 < o.y < line.y - 1 and o.x < head.x - 2 and o.end_x > mark.x1 for o in lines)
                if line.id in led or line.y - last.y > limit or blocked:
                    break
                item.append(line)
            items.append([s.id for s in item])
    return items


def _apply_bullets(blocks: list[BlockPlan], items: list[list[int]], source: dict[int, Segment]) -> list[BlockPlan]:
    """모델 블록을 머리표 항목에 맞춘다. 항목 경계가 블록 안에서 뒤섞였으면 그 블록은 그대로 둔다."""
    from hiremate_ai.agents.analysis.application.normalize import ListPlan

    member = {number: index for index, item in enumerate(items) for number in item}
    result: list[BlockPlan] = []
    for block in blocks:
        if block.type == "section":
            children = _apply_bullets(block.blocks, items, source)
            result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
        elif block.type == "paragraph" and not block.keep_lines:
            result.extend(_split_bulleted(block, items, member, source))
        elif block.type == "list" and not any(item.items for item in block.items):
            ids = list(_ids((item.segments for item in block.items), source))
            found = _item_run(ids, items, member)
            exact = found is not None and found[0] == 0 and found[2] == len(ids)
            result.append(_bullet_list([items[i] for i in found[1]]) if found is not None and exact else block)
        else:
            result.append(block)
    # 한 항목씩 따로 떨어진 문단과 목록을 이어 붙인다.
    merged: list[BlockPlan] = []
    for block in result:
        if _exact_items(block, items, source) and merged and _exact_items(merged[-1], items, source):
            previous = merged.pop()
            assert previous.type in ("list", "paragraph") and block.type in ("list", "paragraph")
            merged.append(ListPlan(type="list", items=[*_as_items(previous, source), *_as_items(block, source)]))
        elif block.type == "paragraph" and _exact_items(block, items, source):
            merged.append(ListPlan(type="list", items=_as_items(block, source)))
        else:
            merged.append(block)
    return merged


def _item_run(ids: list[int], items: list[list[int]], member: dict[int, int]) -> tuple[int, list[int], int] | None:
    """ids 안에서 머리표 항목들이 빠짐없이 차례로 붙어 있는 구간(시작, 항목 번호들, 끝)을 찾는다."""
    start = next((i for i, number in enumerate(ids) if number in member), None)
    if start is None:
        return None
    position, run = start, []
    while position < len(ids) and ids[position] in member:
        index = member[ids[position]]
        item = items[index]
        if ids[position : position + len(item)] != item:
            return None
        run.append(index)
        position += len(item)
    if any(number in member for number in ids[position:]):
        return None
    return start, run, position


def _split_bulleted(
    block: ParagraphPlan, items: list[list[int]], member: dict[int, int], source: dict[int, Segment]
) -> list[BlockPlan]:
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan

    ids = list(_ids([block.segments], source))
    found = _item_run(ids, items, member)
    if found is None or (found[0] == 0 and found[2] == len(ids) and len(found[1]) == 1):
        return [block]
    start, run, end = found
    lead = [ParagraphPlan(type="paragraph", segments=[str(n) for n in ids[:start]])] if start else []
    tail = [ParagraphPlan(type="paragraph", segments=[str(n) for n in ids[end:]])] if end < len(ids) else []
    return [*lead, _bullet_list([items[i] for i in run]), *tail]


def _bullet_list(groups: list[list[int]]) -> BlockPlan:
    from hiremate_ai.agents.analysis.application.normalize import ItemPlan, ListPlan

    return ListPlan(type="list", items=[ItemPlan(segments=[str(n) for n in group]) for group in groups])


def _exact_items(block: BlockPlan, items: list[list[int]], source: dict[int, Segment]) -> bool:
    """블록이 머리표 항목과 정확히 같은 묶음인가. 문단은 항목 하나, 목록은 항목마다 하나여야 한다."""
    if block.type == "paragraph":
        return not block.keep_lines and list(_ids([block.segments], source)) in items
    if block.type == "list":
        return bool(block.items) and all(
            not item.items and list(_ids([item.segments], source)) in items for item in block.items
        )
    return False


def _as_items(block: BlockPlan, source: dict[int, Segment]) -> list[ItemPlan]:
    from hiremate_ai.agents.analysis.application.normalize import ItemPlan

    if block.type == "list":
        groups = [item.segments for item in block.items]
    elif block.type == "paragraph":
        groups = [block.segments]
    else:
        return []
    return [ItemPlan(segments=[str(n) for n in _ids([group], source)]) for group in groups]


def _repair_forms(
    plan: DocumentPlan, pages: list[PdfPage], by_page: dict[int, list[Segment]], source: dict[int, Segment]
) -> list[BlockPlan]:
    # 문서 제목이나 목록, 잡음과 충돌하는 위치는 모델 구조를 그대로 둔다.
    protected = set(_ids([plan.title or [], *_protected_refs(plan.blocks)], source))
    blocks = plan.blocks
    found = [forms for page in pages if len(forms := _forms(page, by_page.get(page.number, []))) >= 3]
    if not found:
        spanning = _spanning_forms(pages, by_page)
        found = [spanning] if len(spanning) >= 2 else []
    for forms in found:
        for label, body in forms:
            target = {s.id for s in (*label, *body)}
            if not target & protected:
                blocks = _splice(blocks, _form_section(label, body), target, source)
    return blocks


def _spanning_forms(pages: list[PdfPage], by_page: dict[int, list[Segment]]) -> list[_Form]:
    """칸 표 양식: 세로선이 왼쪽 좁은 항목 칸과 본문 칸을 나누고, 가로선이 항목을 나눈다. 본문 칸은 쪽을 넘을 수 있다.

    쪽마다 표 틀(왼쪽 테두리, 항목 칸 경계, 오른쪽 테두리 세로선)을 찾아 가로선으로 칸을 자른다. 위쪽 가로선 없이 쪽
    맨 위에서 시작한 첫 칸에 항목 이름이 없으면 앞 쪽 칸의 본문이 이어진 것이다. 항목 이름은 짧고(합쳐 30자 이하)
    본문은 세 줄 넘는 글이어야 한다. 칸을 가로지르는 조각이 있는 쪽은 쓰지 않는다.
    """
    forms: list[tuple[list[Segment], list[Segment]]] = []
    for page in pages:
        frame = _label_frame(page)
        if frame is None:
            continue
        left, divider, right, top, bottom, cuts = frame
        lines = sorted((s for s in by_page.get(page.number, []) if s.x < s.end_x), key=lambda s: (s.y, s.x))
        inside = [s for s in lines if top < s.y < bottom and left - 1 <= s.x and s.end_x <= right + 1]
        if any(s.x < divider - 1 and s.end_x > divider + 1 for s in inside):
            continue
        bounds = [top, *(y for y in cuts if top + 2 < y < bottom - 2), bottom]
        for index, (upper, lower) in enumerate(pairwise(bounds)):
            cell = [s for s in inside if upper < s.y < lower]
            label = [s for s in cell if s.end_x <= divider + 1]
            body = [s for s in cell if s.x >= divider - 1]
            if label:
                forms.append((label, body))
            elif index == 0 and not any(abs(y - top) <= 2 for y in cuts) and forms and body:
                forms[-1][1].extend(body)
    return [
        (label, body)
        for label, body in forms
        if sum(len(s.text.strip()) for s in label) <= 30
        and any(char.isalpha() for s in label for char in s.text)
        and len(body) >= 3
        and median(len(s.text) for s in body) >= 25
    ]


def _label_frame(page: PdfPage) -> tuple[float, float, float, float, float, list[float]] | None:
    """쪽의 칸 표 틀: (왼쪽 테두리, 항목 칸 경계, 오른쪽 테두리, 위, 아래, 가로선 y 목록). 없으면 None."""
    layout = page.layout
    if layout is None:
        return None
    for evidence, thickness in ((layout.fill_rects, 1.6), (page.grid_lines, 0.0)):
        across, down = _rules(evidence, thickness)
        columns: dict[int, list[float]] = {}
        for start, end, x in down:
            key = next((k for k in columns if abs(k - x) <= 1.5), round(x))
            extent = columns.setdefault(key, [start, end])
            extent[0], extent[1] = min(extent[0], start), max(extent[1], end)
        tall = sorted((x, a, b) for x, (a, b) in columns.items() if b - a >= layout.height * 0.2)
        if len(tall) < 3:
            continue
        (left, top, bottom), (right, _, _) = tall[0], tall[-1]
        dividers = [x for x, a, b in tall[1:-1] if 0.07 < (x - left) / (right - left) < 0.18]
        if right - left < layout.width * 0.7 or len(dividers) != 1:
            continue
        cuts = sorted(y for a, b, y in across if a <= left + 2 and b >= right - 2)
        return left, dividers[0], right, top, bottom, cuts
    return None


def ensure_visual_notices(
    document: NormalizedDocument, pages: list[PdfPage], segments: list[Segment]
) -> NormalizedDocument:
    """근거가 있고 아직 안내하지 않은 쪽만 원본 확인 안내를 덧붙인다."""
    entries = list(_walk(document.blocks))
    covered = {page for block in entries if isinstance(block, DocumentVisualNotice) for page in block.pages}
    by_page = _segments_by_page(segments)
    missing = {
        page.number
        for page in pages
        if page.number not in covered
        and page.number <= document.document.page_count
        and page.layout is not None
        and page.image_count
        and _supports_visual_notice(page.layout, by_page.get(page.number, []))
    }
    if not missing:
        return document
    occupied = {entry.id for entry in entries}
    notices = []
    for page_number in sorted(missing):
        notice_id = f"visual-notice-page-{page_number}"
        while notice_id in occupied:
            notice_id += "-extra"
        occupied.add(notice_id)
        notices.append(
            DocumentVisualNotice(
                id=notice_id,
                type="visual_notice",
                pages=[page_number],
                visual_type="unknown",
                detection="suspected",
                message=visual_notice_message("unknown", [page_number]),
            )
        )
    return NormalizedDocument(
        schema_version=document.schema_version, document=document.document, blocks=[*document.blocks, *notices]
    )


def _segments_by_page(segments: list[Segment]) -> dict[int, list[Segment]]:
    result: dict[int, list[Segment]] = {}
    for segment in segments:
        result.setdefault(segment.page, []).append(segment)
    return result


def _numbers(ref: str, source: dict[int, Segment]) -> range:
    first_text, _, last_text = ref.partition("-")
    first, last = int(first_text), int(last_text or first_text)
    # 유효하지 않은 참조는 수정하지 않고 원래 조립기의 검증에 맡긴다.
    return range(first, last + 1) if first in source and last in source else range(0)


def _ids(refs: Iterable[list[str]], source: dict[int, Segment]) -> Iterator[int]:
    for group in refs:
        for ref in group:
            yield from _numbers(ref, source)


def _item_refs(items: list[ItemPlan]) -> Iterator[list[str]]:
    for item in items:
        yield item.segments
        yield from _item_refs(item.items)


def _block_refs(block: BlockPlan) -> Iterator[list[str]]:
    if block.type == "paragraph" or block.type == "artifact":
        yield block.segments
    elif block.type == "section":
        yield block.title or []
        for child in block.blocks:
            yield from _block_refs(child)
    elif block.type == "table":
        yield block.title or []
        yield from block.headers
        for row in block.rows:
            yield from row
    elif block.type == "list":
        yield from _item_refs(block.items)


def _plan_refs(title: list[str] | None, blocks: list[BlockPlan]) -> Iterator[list[str]]:
    yield title or []
    for block in blocks:
        yield from _block_refs(block)


def _protected_refs(blocks: list[BlockPlan]) -> Iterator[list[str]]:
    for block in blocks:
        if block.type in ("list", "artifact"):
            yield from _block_refs(block)
        elif block.type == "section":
            yield from _protected_refs(block.blocks)


type _Rule = tuple[float, float, float]  # (start, end, axis)
type _Form = tuple[list[Segment], list[Segment]]


def _rules(evidence: Iterable[PdfBox | PdfGridLine], thickness: float) -> tuple[list[_Rule], list[_Rule]]:
    across: list[_Rule] = []
    down: list[_Rule] = []
    for line in evidence:
        left, right = sorted((line.x1, line.x2))
        top, bottom = sorted((line.y1, line.y2))
        if bottom - top <= thickness and right - left > 18:
            across.append((left, right, (top + bottom) / 2))
        elif right - left <= thickness and bottom - top > 14:
            down.append((top, bottom, (left + right) / 2))
    merged: list[_Rule] = []
    first = 0
    for left, right, y in sorted(across, key=lambda rule: (rule[2], rule[0])):
        while first < len(merged) and y - merged[first][2] > 0.6:
            first += 1
        for index in range(first, len(merged)):
            a, b, axis = merged[index]
            if left <= b + 1.1 and right >= a - 1.1:
                merged[index] = (min(a, left), max(b, right), axis)
                break
        else:
            merged.append((left, right, y))
    # 뒤늦게 발견한 조각이 앞의 두 수평선을 잇는 경우도 합친다.
    across = []
    for left, right, y in sorted(merged, key=lambda rule: (round(rule[2]), rule[0])):
        if across and abs(across[-1][2] - y) <= 0.6 and left <= across[-1][1] + 1.1:
            a, b, axis = across[-1]
            across[-1] = (a, max(b, right), axis)
        else:
            across.append((left, right, y))
    return sorted(across, key=lambda rule: (rule[2], rule[0])), down


def _forms(page: PdfPage, lines: list[Segment]) -> list[_Form]:
    layout = page.layout
    if layout is None:
        return []
    ordered = sorted(lines, key=lambda s: (s.y, s.x))
    # 채운 규칙선의 끝점이 클리핑 경계에 닿으면 grid_lines에 없을 수 있다. 서로 다른 단서를 섞지는 않는다.
    filled = _form_bands(layout, ordered, *_rules(layout.fill_rects, 1.6))
    return filled if len(filled) >= 3 else _form_bands(layout, ordered, *_rules(page.grid_lines, 0))


def _form_bands(layout: PdfLayout, ordered: list[Segment], across: list[_Rule], down: list[_Rule]) -> list[_Form]:
    forms: list[_Form] = []
    for (left, right, top), (a, b, bottom) in pairwise(across):
        if (
            abs(left - a) > 1.2
            or abs(right - b) > 1.2
            or not 0 <= left < right <= layout.width
            or not 0 <= top < bottom <= layout.height
            or right - left < layout.width * 0.7
            or not 95 < bottom - top < 160
        ):
            continue
        dividers = [x for t, z, x in down if t <= top + 2 and z >= bottom - 2 and left + 15 < x < right - 15]
        if len(dividers) != 1 or not 0.07 < (dividers[0] - left) / (right - left) < 0.18:
            continue
        divider = dividers[0]
        label: list[Segment] = []
        body: list[Segment] = []
        for s in ordered:
            if not (top < s.y < bottom and left < (s.x + s.end_x) / 2 < right):
                continue
            in_label = left - 1 <= s.x < s.end_x <= divider + 1
            in_body = divider - 1 <= s.x < s.end_x <= right + 1
            if in_label == in_body:
                break
            (label if in_label else body).append(s)
        else:
            if (
                label
                and len(body) >= 3
                and sum(len(s.text) for s in label) <= 30
                and median(len(s.text) for s in body) >= 25
            ):
                forms.append((label, body))
    return forms


def _refs(segments: Iterable[Segment]) -> list[str]:
    return [str(s.id) for s in segments]


def _paragraph(segments: list[Segment]) -> ParagraphPlan:
    from hiremate_ai.agents.analysis.application.normalize import ParagraphPlan

    return ParagraphPlan(type="paragraph", segments=_refs(segments))


def _form_section(label: list[Segment], body: list[Segment]) -> BlockPlan:
    from hiremate_ai.agents.analysis.application.normalize import SectionPlan

    pitch = median(b.y - a.y for a, b in pairwise(body))
    paragraphs: list[list[Segment]] = [[]]
    for segment in body:
        if paragraphs[-1] and segment.y - paragraphs[-1][-1].y > pitch * 1.55:
            paragraphs.append([])
        paragraphs[-1].append(segment)
    return SectionPlan(type="section", title=_refs(label), blocks=[_paragraph(lines) for lines in paragraphs])


def _split_refs(refs: list[str], target: set[int], source: dict[int, Segment]) -> tuple[list[str], list[str]]:
    before: list[str] = []
    after: list[str] = []
    output = before
    for ref in refs:
        numbers = _numbers(ref, source)
        if target.isdisjoint(numbers):
            output.append(ref)
            continue
        for number in numbers:
            if number in target:
                output = after
            else:
                output.append(str(number))
    return before, after


def _strip_form_refs(block: BlockPlan, target: set[int], source: dict[int, Segment]) -> BlockPlan | None:
    if target.isdisjoint(_ids(_block_refs(block), source)):
        return block

    def strip(refs: list[str]) -> list[str]:
        before, after = _split_refs(refs, target, source)
        return before + after

    if block.type == "paragraph":
        refs = strip(block.segments)
        return block.model_copy(update={"segments": refs}) if refs else None
    if block.type == "section":
        title = strip(block.title or [])
        children = [child for item in block.blocks if (child := _strip_form_refs(item, target, source)) is not None]
        return block.model_copy(update={"title": title or None, "blocks": children}) if title or children else None
    if block.type == "table":
        title = strip(block.title or [])
        headers = [strip(cell) for cell in block.headers]
        rows = [[strip(cell) for cell in row] for row in block.rows]
        if title or any(headers) or any(cell for row in rows for cell in row):
            return block.model_copy(update={"title": title or None, "headers": headers, "rows": rows})
        return None
    return block


def _splice(
    blocks: list[BlockPlan], section: BlockPlan, target: set[int], source: dict[int, Segment]
) -> list[BlockPlan]:
    refs_by_block = [set(_ids(_block_refs(block), source)) for block in blocks]
    touched = [index for index, refs in enumerate(refs_by_block) if not refs.isdisjoint(target)]
    if len(touched) == 1:
        index = touched[0]
        anchor = blocks[index]
        if anchor.type == "section":
            heading = list(_ids([anchor.title or []], source))
            if heading and max(heading) < min(target):
                children = _splice(anchor.blocks, section, target, source)
                return [*blocks[:index], anchor.model_copy(update={"blocks": children}), *blocks[index + 1 :]]

    result: list[BlockPlan] = []
    placed = False
    last = max(target)
    for block, refs in zip(blocks, refs_by_block, strict=True):
        overlap = bool(refs & target)
        if not placed and (overlap or (refs and min(refs) > last)):
            placed = True
            if overlap and block.type == "paragraph":
                before, after = _split_refs(block.segments, target, source)
                if before:
                    result.append(block.model_copy(update={"segments": before}))
                result.append(section)
                if after:
                    result.append(block.model_copy(update={"segments": after}))
                continue
            result.append(section)
        remaining = _strip_form_refs(block, target, source) if overlap else block
        if remaining is not None:
            result.append(remaining)
    if not placed:
        result.append(section)
    return result


def _question_heading(head: Segment, body: Segment) -> bool:
    return (
        _NUMBERED.fullmatch(head.text.strip()) is not None
        and any(char.isalpha() for char in head.text)
        and head.page == body.page
        and abs(head.x - body.x) < 4
        and head.x < head.end_x
        and body.x < body.end_x
        and head.size is not None
        and body.size is not None
        and body.size > 0
        and head.size >= body.size + 0.35
        and body.y - head.y > 12
    )


def _question_section(head: Segment, body: list[Segment]) -> BlockPlan:
    from hiremate_ai.agents.analysis.application.normalize import SectionPlan

    sections: list[BlockPlan] = []
    target = sections
    pending: list[Segment] = []
    for index, line in enumerate(body):
        bracket = _BRACKETED.fullmatch(line.text.strip())
        following = body[index + 1] if index + 1 < len(body) else None
        # 대괄호 모양만으로 승격하지 않는다. 복구할 질문 안에서 같은 단의 본문이 뒤따라야 한다.
        if (
            bracket
            and any(char.isalpha() for char in bracket[1])
            and following is not None
            and abs(line.x - head.x) < 4
            and abs(line.x - following.x) < 4
            and following.y - line.y > 1
            and not _BRACKETED.fullmatch(following.text.strip())
        ):
            if pending:
                target.append(_paragraph(pending))
                pending = []
            subsection = SectionPlan(type="section", title=_refs([line]), blocks=[])
            sections.append(subsection)
            target = subsection.blocks
        else:
            pending.append(line)
    if pending:
        target.append(_paragraph(pending))
    return SectionPlan(type="section", title=_refs([head]), blocks=sections)


def _name_untitled_questions(
    blocks: list[BlockPlan], headings: set[int], source: dict[int, Segment], claimed: set[int]
) -> list[BlockPlan]:
    """제목 없는 섹션이 문항 제목 줄로 시작하면 그 줄을 제목으로 떼고, 다음 문항 앞까지의 형제 블록을 안에 넣는다.

    문항 제목은 번호가 하나씩 이어지고 글자 모양과 왼쪽 끝이 같은 앞뒤 문항이 있어야 한다. 문항 범위(이 문항 첫 줄부터
    다음 문항 제목 앞까지, 마지막 문항은 가진 조각 끝까지)에서 섹션과 넣을 형제가 갖지 않은 조각은 계획 어디에도
    없어야 한다. 그런 조각은 조립 때 바로 앞 조각 뒤에 되살아나므로 같은 문항 안에 남는다. 마지막 문항은 뒤따르는
    형제 섹션만 넣는다(뒤의 맺음말 문단은 문항 밖에 둔다).
    """
    ordered = sorted(headings)
    result: list[BlockPlan] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        index += 1
        if block.type != "section" or block.title or not block.blocks:
            result.append(block)
            continue
        first = block.blocks[0]
        start = next(_ids([first.segments], source), None) if first.type == "paragraph" else None
        if first.type != "paragraph" or first.keep_lines or start not in headings:
            result.append(block)
            continue
        assert start is not None
        position = ordered.index(start)
        neighbour = ordered[position + 1] if position + 1 < len(ordered) else None
        series = neighbour if neighbour is not None else (ordered[position - 1] if position else None)
        if series is None or not _same_series(source[start], source[series], after=neighbour is not None):
            result.append(block)
            continue
        taken: list[BlockPlan] = []
        while index + len(taken) < len(blocks):
            candidate = blocks[index + len(taken)]
            ids = list(_ids(_block_refs(candidate), source))
            inside = bool(ids) and min(ids) > start and (neighbour is None or max(ids) < neighbour)
            if not inside or (neighbour is None and candidate.type != "section"):
                break
            taken.append(candidate)
        owned = set(_ids(_block_refs(block), source)).union(*(set(_ids(_block_refs(b), source)) for b in taken))
        scope = set(range(start, neighbour if neighbour is not None else max(owned) + 1))
        if not owned <= scope or (scope - owned) & claimed:
            result.append(block)
            continue
        before, after = _split_refs(first.segments, {start}, source)
        if before:
            result.append(block)
            continue
        body = [first.model_copy(update={"segments": after})] if after else []
        # 문항 안으로 넣는 제목 없는 섹션은 쪽 경계에서 모델이 연 묶음일 뿐이라 풀어서 넣는다.
        inner = [child for b in taken for child in (b.blocks if b.type == "section" and not b.title else [b])]
        result.append(block.model_copy(update={"title": [str(start)], "blocks": [*body, *block.blocks[1:], *inner]}))
        index += len(taken)
    return result


def _same_series(head: Segment, other: Segment, *, after: bool) -> bool:
    """두 문항 제목이 번호가 하나 차이로 이어지고 글자 모양과 왼쪽 끝이 같은가. after면 other가 다음 문항이다."""
    a, b = (_NUMBERED.fullmatch(s.text.strip()) for s in (head, other))
    if a is None or b is None:
        return False
    step = int(b[1]) - int(a[1]) if after else int(a[1]) - int(b[1])
    return step == 1 and abs(head.x - other.x) < 4 and (head.size, head.color) == (other.size, other.color)


def _recover_questions(
    blocks: list[BlockPlan], title: list[str] | None, segments: list[Segment], source: dict[int, Segment]
) -> list[BlockPlan]:
    numbered = {index for index, (head, body) in enumerate(pairwise(segments)) if _question_heading(head, body)}
    if len(numbered) < 2:
        return blocks
    claimed = set(_ids(_plan_refs(title, blocks), source))
    blocks = _name_untitled_questions(blocks, {segments[i].id for i in numbered}, source, claimed)
    used = set(_ids(_plan_refs(title, blocks), source))
    for index in sorted(numbered):
        head = segments[index]
        if head.id in used:
            continue
        body = []
        for next_index in range(index + 1, len(segments)):
            line = segments[next_index]
            if line.id in used or line.page != head.page or next_index in numbered:
                break
            body.append(line)
        if body:
            blocks = _splice(blocks, _question_section(head, body), {head.id, *(s.id for s in body)}, source)
    return blocks


def _caption_slots(lines: list[Segment], boxes: list[PdfBox]) -> dict[int, int]:
    slots = {}
    for line in lines:
        matches = [
            index
            for index, box in enumerate(boxes)
            if 0 < line.y - box.y2 < 36 and box.x1 - 4 <= line.x < line.end_x <= box.x2 + 4
        ]
        if len(matches) == 1:
            slots[line.id] = matches[0]
    return slots


def _caption_groups(
    block: BlockPlan, slots: dict[int, int], source: dict[int, Segment], used: Counter[int]
) -> dict[int, list[Segment]]:
    groups: dict[int, list[Segment]] = {}
    for refs in _block_refs(block):
        for ref in refs:
            numbers = _numbers(ref, source)
            if not numbers:
                return {}
            for number in numbers:
                if number not in slots or used[number] != 1:
                    return {}
                groups.setdefault(slots[number], []).append(source[number])
    return groups


def _caption_neighbor(line: Segment, preceding: Segment) -> bool:
    return (
        0 < line.y - preceding.y <= 16
        and line.size is not None
        and preceding.size is not None
        and min(line.size, preceding.size) > 0
        and abs(line.size - preceding.size) < 0.5
    )


def _split_captions(
    blocks: list[BlockPlan],
    slots: dict[int, int],
    source: dict[int, Segment],
    used: Counter[int],
    absent: list[Segment],
) -> list[BlockPlan]:
    result: list[BlockPlan] = []
    for block in blocks:
        if block.type == "section":
            children = _split_captions(block.blocks, slots, source, used, absent)
            result.append(block if children == block.blocks else block.model_copy(update={"blocks": children}))
            continue
        groups = _caption_groups(block, slots, source, used) if block.type in ("paragraph", "table") else {}
        if len(groups) < 2:
            result.append(block)
            continue
        for line in absent:
            group = groups.get(slots[line.id])
            if group and not used[line.id] and any(_caption_neighbor(line, preceding) for preceding in group):
                group.append(line)
                used[line.id] = 1
        result.extend(_paragraph(sorted(groups[slot], key=lambda s: (s.y, s.x))) for slot in sorted(groups))
    return result


def _repair_captions(
    plan: DocumentPlan, pages: list[PdfPage], by_page: dict[int, list[Segment]], source: dict[int, Segment]
) -> list[BlockPlan]:
    blocks = plan.blocks
    used: Counter[int] | None = None
    for page in pages:
        layout = page.layout
        if layout is None or len(layout.image_boxes) < 2:
            continue
        boxes = sorted((box for box in layout.image_boxes if box.x2 - box.x1 > layout.width * 0.13), key=lambda b: b.x1)
        if len(boxes) < 2:
            continue
        lines = by_page.get(page.number, [])
        slots = _caption_slots(lines, boxes)
        if len(set(slots.values())) < 2:
            continue
        if used is None:
            used = Counter(_ids(_plan_refs(plan.title, blocks), source))
        absent = sorted((s for s in lines if s.id in slots and not used[s.id]), key=lambda s: (s.y, s.x))
        blocks = _split_captions(blocks, slots, source, used, absent)
    return blocks


def _walk(entries: Iterable[DocumentBlock | DocumentListItem]) -> Iterator[DocumentBlock | DocumentListItem]:
    for entry in entries:
        yield entry
        if isinstance(entry, DocumentSection):
            yield from _walk(entry.blocks)
        elif isinstance(entry, (DocumentList, DocumentListItem)):
            yield from _walk(entry.items)


def _notice_caption(line: Segment, body_size: float | None) -> bool:
    clean = " ".join(line.text.split())
    if not 5 <= len(clean) <= 120 or _EXCLUDED_CAPTION.search(clean):
        return False
    if _FIGURE_LABEL.search(clean):
        return True
    # 그림 번호가 없으면 위 본문보다 작은 글자로 쓴 수치와 추세가 함께 있어야 한다.
    return bool(
        line.size is not None
        and body_size is not None
        and 0 < line.size <= body_size * 0.9
        and _MEASURE.search(clean)
        and _TREND.search(clean)
    )


def _supports_visual_notice(layout: PdfLayout, lines: list[Segment]) -> bool:
    for box in layout.image_boxes:
        left, top = max(box.x1, 0), max(box.y1, 0)
        right, bottom = min(box.x2, layout.width), min(box.y2, layout.height)
        width, height = right - left, bottom - top
        if (
            width < layout.width * 0.18
            or height < layout.height * 0.12
            or width < height * 1.12
            or width * height > layout.width * layout.height * 0.65
        ):
            continue
        body_sizes = [line.size for line in lines if line.size and line.y < top and len(line.text.strip()) >= 60]
        body_size = median(body_sizes) if body_sizes else None
        for line in lines:
            if (
                3 <= line.y - bottom <= min(30, layout.height * 0.055)
                and left - 4 <= line.x < line.end_x <= right + 4
                and abs((line.x + line.end_x - left - right) / 2) <= width * 0.25
                and _notice_caption(line, body_size)
            ):
                return True
    return False
