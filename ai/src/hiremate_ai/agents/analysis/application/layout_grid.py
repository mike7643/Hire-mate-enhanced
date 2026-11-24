"""Extract table cell references from PDF geometry, without rewriting source text."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hiremate_ai.agents.analysis.application.normalize import Segment
    from hiremate_ai.contract import PdfFillRect, PdfLayout, PdfPage

# Coordinates are PDF points. Snap near-identical rules, but not across cell-size gaps.
_AXIS_TOL = 0.65
_JOIN_TOL = 1.5
_TEXT_TOL = 1.0
_MIN_CELL = 6.0
_MAX_RULES = 1024
_MAX_CELLS = 256
_DOT_TOL = 2.0
_MIN_DOTS = 20
_OPEN_TOL = 1.2
_RGB = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class GeometryTable:
    """One-based source references; marks are not members of any cell."""

    page: int
    mode: str
    headers: list[list[int]]
    rows: list[list[list[int]]]
    marks: list[int]
    continuation_page: int | None = None


@dataclass(frozen=True)
class _Rule:
    axis: float
    start: float
    end: float

    def covers(self, start: float, end: float) -> bool:
        return self.start <= start + _JOIN_TOL and self.end >= end - _JOIN_TOL

    def crosses(self, start: float, end: float) -> bool:
        return self.start < end - _JOIN_TOL and self.end > start + _JOIN_TOL

    def touches(self, axes: list[float]) -> bool:
        index = bisect_left(axes, self.start - _JOIN_TOL)
        return index < len(axes) and axes[index] <= self.end + _JOIN_TOL


@dataclass(frozen=True)
class _Frame:
    xs: list[float]
    ys: list[float]


@dataclass(frozen=True)
class _PageGeometry:
    number: int
    layout: PdfLayout
    horizontal: list[_Rule]
    vertical: list[_Rule]
    vertical_sources: set[_Rule]
    segments: list[Segment]
    baselines: list[float]

    def between(self, top: float, bottom: float) -> list[Segment]:
        return self.segments[bisect_right(self.baselines, top) : bisect_left(self.baselines, bottom)]


def _axis_groups(rules: set[_Rule]) -> list[list[_Rule]]:
    groups: list[list[_Rule]] = []
    for rule in sorted(rules, key=lambda r: (r.axis, r.start, r.end)):
        if not groups or rule.axis - groups[-1][0].axis > _AXIS_TOL:
            groups.append([])
        groups[-1].append(rule)
    return groups


def _merge_rules(rules: set[_Rule]) -> list[_Rule]:
    merged: list[_Rule] = []
    for group in _axis_groups(rules):
        axis = median(r.axis for r in group)
        intervals = sorted((r.start, r.end) for r in group)
        start, end = intervals[0]
        for left, right in intervals[1:]:
            if left > end + _JOIN_TOL:
                merged.append(_Rule(axis, start, end))
                start, end = left, right
            else:
                end = max(end, right)
        merged.append(_Rule(axis, start, end))
    return merged


def _page_geometry(page: PdfPage, segments: list[Segment]) -> _PageGeometry | None:
    if page.layout is None or not segments:
        return None
    horizontal: set[_Rule] = set()
    vertical: set[_Rule] = set()
    for line in page.grid_lines:
        left, right = sorted((line.x1, line.x2))
        top, bottom = sorted((line.y1, line.y2))
        if top == bottom and right - left >= _MIN_CELL:
            horizontal.add(_Rule(top, left, right))
        elif left == right and bottom - top >= _MIN_CELL:
            vertical.add(_Rule(left, top, bottom))
    for fill in page.layout.fill_rects:
        width, height = fill.x2 - fill.x1, fill.y2 - fill.y1
        if width >= _MIN_CELL and height <= 1.6:
            horizontal.add(_Rule((fill.y1 + fill.y2) / 2, fill.x1, fill.x2))
        elif height >= _MIN_CELL and width <= 1.6:
            vertical.add(_Rule((fill.x1 + fill.x2) / 2, fill.y1, fill.y2))
    if len(horizontal) + len(vertical) > _MAX_RULES:
        return None
    ordered = sorted(
        (s for s in segments if all(map(isfinite, (s.x, s.y, s.end_x))) and s.x <= s.end_x),
        key=lambda s: (s.y, s.x, s.id),
    )
    return _PageGeometry(
        page.number,
        page.layout,
        _merge_rules(horizontal),
        _merge_rules(vertical),
        vertical,
        ordered,
        [s.y for s in ordered],
    )


def _width_bands(horizontal: list[_Rule]) -> list[list[_Rule]]:
    bands: list[list[_Rule]] = []
    for rule in horizontal:
        for band in bands:
            if abs(band[0].start - rule.start) <= _JOIN_TOL and abs(band[0].end - rule.end) <= _JOIN_TOL:
                band.append(rule)
                break
        else:
            bands.append([rule])
    return bands


def _wide_caps(geometry: _PageGeometry, left: float, right: float) -> list[_Rule]:
    return [r for r in geometry.horizontal if r.start < left - _JOIN_TOL and abs(r.end - right) <= _OPEN_TOL]


def _has_sides(geometry: _PageGeometry, left: float, right: float, top: float, bottom: float) -> bool:
    return all(
        any(abs(r.axis - x) <= _JOIN_TOL and r.covers(top, bottom) for r in geometry.vertical) for x in (left, right)
    )


def _shared_cap_rows(
    geometry: _PageGeometry, left: float, right: float, ys: list[float], caps: list[_Rule]
) -> list[float]:
    # A merged left margin can share the top and bottom rules of a closed
    # subgrid. Neither cap can be ignored when checking for unfilled cells.
    above = [r for r in caps if _MIN_CELL <= ys[0] - r.axis <= 100]
    below = [r for r in caps if _MIN_CELL <= r.axis - ys[-1] <= 100]
    if above and below:
        top, bottom = above[-1], below[0]
        if _same_extent(top, bottom) and _has_sides(geometry, left, right, top.axis, bottom.axis):
            return [top.axis, *ys, bottom.axis]
    return ys


def _ruled_runs(band: list[_Rule], dividers: list[_Rule], caps: list[float]) -> list[list[float]]:
    """Split equal-width rules where no column divider bridges the gap."""
    runs: list[list[float]] = []
    run: list[float] = []
    for lo, hi in zip(band, band[1:], strict=False):
        cap = bisect_right(caps, lo.axis + _AXIS_TOL)
        interrupted = cap < len(caps) and caps[cap] < hi.axis - _AXIS_TOL
        if not interrupted and any(r.covers(lo.axis, hi.axis) for r in dividers):
            if not run:
                run.append(lo.axis)
            run.append(hi.axis)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _solid_frames(geometry: _PageGeometry, *, split_at_caps: bool = False) -> list[tuple[_Frame, bool]]:
    result: list[tuple[_Frame, bool]] = []
    for band in _width_bands(geometry.horizontal):
        left, right = band[0].start, band[0].end
        dividers = [r for r in geometry.vertical if left + _JOIN_TOL < r.axis < right - _JOIN_TOL]
        if not dividers:
            continue
        caps = _wide_caps(geometry, left, right)
        for ys in _ruled_runs(band, dividers, [r.axis for r in caps] if split_at_caps else []):
            if not split_at_caps and caps:
                ys = _shared_cap_rows(geometry, left, right, ys, caps)
            top, bottom = ys[0], ys[-1]
            if any(
                r not in band and top + _AXIS_TOL < r.axis < bottom - _AXIS_TOL and r.crosses(left, right)
                for r in geometry.horizontal
            ):
                continue  # A partial row rule implies a split or merged cell.
            # PDFBox can expose thin rectangles inside glyphs as individual
            # rules. A floating stroke is not an attached partial divider.
            cuts = [r for r in dividers if r.crosses(top, bottom) and r.touches(ys)]
            if not cuts or any(not r.covers(top, bottom) for r in cuts):
                continue
            xs = [left, *(r.axis for r in cuts), right]
            if (len(xs) - 1) * (len(ys) - 1) > _MAX_CELLS or any(
                b - a < _MIN_CELL for axes in (xs, ys) for a, b in zip(axes, axes[1:], strict=False)
            ):
                continue
            closed = _has_sides(geometry, left, right, top, bottom)
            result.append((_Frame(xs, ys), closed))
    return sorted(result, key=lambda candidate: (candidate[0].ys[0], candidate[0].xs[0]))


def _cells(geometry: _PageGeometry, frame: _Frame) -> list[list[list[Segment]]] | None:
    xs, ys = frame.xs, frame.ys
    grid: list[list[list[Segment]]] = [[[] for _ in xs[1:]] for _ in ys[1:]]
    for span in geometry.between(ys[0], ys[-1]):
        center = (span.x + span.end_x) / 2
        if not xs[0] < center < xs[-1]:
            continue
        j = bisect_right(xs, center) - 1
        i = bisect_left(ys, span.y) - 1
        if span.x < xs[j] - _TEXT_TOL or span.end_x > xs[j + 1] + _TEXT_TOL or span.y == ys[i + 1]:
            return None
        grid[i][j].append(span)
    return grid


def _ids(row: list[list[Segment]]) -> list[list[int]]:
    return [[s.id for s in cell] for cell in row]


def _colored_fills(geometry: _PageGeometry) -> list[PdfFillRect]:
    result: list[PdfFillRect] = []
    for fill in geometry.layout.fill_rects:
        if fill.x2 - fill.x1 >= 10 and fill.y2 - fill.y1 >= 10:
            rgb = bytes.fromhex(fill.color[1:])
            if max(rgb) - min(rgb) >= 31:
                result.append(fill)
    return result


def _dot_rules(geometry: _PageGeometry) -> list[_Rule]:
    dots = {
        _Rule(fill.y1, fill.x1, fill.x2)
        for fill in geometry.layout.fill_rects
        if fill.x2 - fill.x1 < 3 and fill.y2 - fill.y1 < 2
    }
    lines: list[_Rule] = []
    for group in _axis_groups(dots):
        if len(group) < _MIN_DOTS:
            continue
        intervals = sorted((r.start, r.end) for r in group)
        start, end = intervals[0]
        count = 1
        for a, b in intervals[1:]:
            if a > end + 4.5:
                if count >= _MIN_DOTS:
                    lines.append(_Rule(group[0].axis, start, end))
                start, end, count = a, b, 1
            else:
                count += a > end - _AXIS_TOL
                end = max(end, b)
        if count >= _MIN_DOTS:
            lines.append(_Rule(group[0].axis, start, end))
    return lines


def _same_extent(a: _Rule, b: _Rule, tol: float = _JOIN_TOL) -> bool:
    return abs(a.start - b.start) <= tol and abs(a.end - b.end) <= tol


def _pair_row(geometry: _PageGeometry, fills: list[PdfFillRect], top: _Rule, bottom: _Rule) -> list[list[list[int]]]:
    spans = [s for s in geometry.between(top.axis, bottom.axis) if top.start < (s.x + s.end_x) / 2 < top.end]
    if not spans or len(spans) % 2 or spans[-1].y - spans[0].y > 1.5:
        return []
    spans.sort(key=lambda s: (s.x, s.id))
    row_fills = [r for r in fills if abs(r.y1 - top.axis) <= _DOT_TOL and abs(r.y2 - bottom.axis) <= _DOT_TOL]
    pairs: list[list[list[int]]] = []
    for label, value in zip(spans[::2], spans[1::2], strict=True):
        matched = [r for r in row_fills if r.x1 - _TEXT_TOL <= label.x <= label.end_x <= r.x2 + _TEXT_TOL]
        if len(matched) != 1 or any(r.x1 < (value.x + value.end_x) / 2 < r.x2 for r in row_fills):
            return []
        edge = matched[0].x2
        if (
            value.x < edge - _TEXT_TOL
            or value.end_x > top.end + _TEXT_TOL
            or not any(abs(r.axis - edge) <= _JOIN_TOL and r.covers(top.axis, bottom.axis) for r in geometry.vertical)
        ):
            return []
        pairs.append([[label.id], [value.id]])
    return pairs


def _pairs(geometry: _PageGeometry, fills: list[PdfFillRect], dots: list[_Rule]) -> list[GeometryTable]:
    if not fills:
        return []
    tables: list[GeometryTable] = []
    bottom_rules = sorted(geometry.horizontal + dots, key=lambda r: r.axis)
    for rule in geometry.horizontal:
        if not any(abs(r.y1 - rule.axis) <= _DOT_TOL for r in fills):
            continue
        top = rule
        rows: list[list[list[int]]] = []
        for bottom in bottom_rules:
            if bottom.axis <= top.axis + _DOT_TOL or not _same_extent(rule, bottom, _DOT_TOL):
                continue
            if not 10 <= bottom.axis - top.axis <= 70:
                break
            band = _pair_row(geometry, fills, top, bottom)
            if not band:
                break
            rows.extend(band)
            top = bottom
            if len(rows) * 2 > _MAX_CELLS:
                break
        if 2 <= len(rows) <= _MAX_CELLS // 2:
            tables.append(GeometryTable(geometry.number, "pairs", [], rows, []))
    return tables


def _aligned_fills(fills: list[PdfFillRect]) -> list[list[PdfFillRect]]:
    rows: list[list[PdfFillRect]] = []
    for fill in sorted(fills, key=lambda r: (r.y1, r.x1)):
        if not rows or fill.y1 - rows[-1][0].y1 > _DOT_TOL:
            rows.append([])
        rows[-1].append(fill)
    bands: list[list[PdfFillRect]] = []
    for row in rows:
        for fill in sorted(row, key=lambda r: r.x1):
            if (
                not bands
                or abs(fill.y1 - bands[-1][0].y1) > _DOT_TOL
                or abs(fill.y2 - bands[-1][0].y2) > _DOT_TOL
                or abs(fill.x1 - bands[-1][-1].x2) > _DOT_TOL
            ):
                bands.append([])
            bands[-1].append(fill)
    return [band for band in bands if len(band) >= 2]


def _placeholder(span: Segment) -> bool:
    text = span.text.strip()
    return bool(text) and all(c in "~～/／" for c in text)


def _dotted(geometry: _PageGeometry, fills: list[PdfFillRect], dots: list[_Rule]) -> list[GeometryTable]:
    tables: list[GeometryTable] = []
    bands = _aligned_fills(fills)
    for band in bands:
        top, bottom = band[0].y1, band[0].y2
        xs = [band[0].x1, *(fill.x2 for fill in band)]
        next_header = min((b[0].y1 for b in bands if b[0].y1 > top + _DOT_TOL), default=geometry.layout.height)
        boundaries = sorted(
            {
                r.axis
                for r in dots
                if bottom - _DOT_TOL <= r.axis < next_header
                and abs(r.start - xs[0]) <= _DOT_TOL
                and abs(r.end - xs[-1]) <= _DOT_TOL
            }
        )
        if len(boundaries) < 2 or abs(boundaries[0] - bottom) > _DOT_TOL:
            continue
        ys = [top, *boundaries]
        if (len(xs) - 1) * (len(ys) - 1) > _MAX_CELLS or any(
            not _MIN_CELL <= hi - lo <= 100 for lo, hi in zip(ys, ys[1:], strict=False)
        ):
            continue
        grid = _cells(geometry, _Frame(xs, ys))
        if grid is None or not all(grid[0]) or any(_placeholder(s) for c in grid[0] for s in c):
            continue
        rows: list[list[list[int]]] = []
        marks: list[int] = []
        placeholder_rows = 0
        for row in grid[1:]:
            spans = [s for cell in row for s in cell]
            if all(_placeholder(s) for s in spans):
                marks.extend(s.id for s in spans)
                placeholder_rows += bool(spans)
            else:
                rows.append(_ids(row))
        if rows:
            tables.append(GeometryTable(geometry.number, "dotted", _ids(grid[0]), rows, marks))
        elif placeholder_rows >= 3:
            # Several explicitly marked empty rows still support the styled
            # header. Unused, mostly blank form templates are left alone.
            tables.append(GeometryTable(geometry.number, "rectangular", _ids(grid[0]), [], sorted(marks)))
    return tables


def _color_contrast(a: str | None, b: str | None) -> bool:
    if a is None or b is None or not _RGB.fullmatch(a) or not _RGB.fullmatch(b):
        return False
    return max(abs(x - y) for x, y in zip(bytes.fromhex(a[1:]), bytes.fromhex(b[1:]), strict=True)) >= 10


def _background(fills: list[PdfFillRect], left: float, top: float, right: float, bottom: float) -> str:
    for fill in reversed(fills):
        if (
            min(fill.x2, right) - max(fill.x1, left) >= (right - left) * 0.8
            and min(fill.y2, bottom) - max(fill.y1, top) >= (bottom - top) * 0.7
        ):
            return fill.color.lower()
    return "#ffffff"


def _has_header(geometry: _PageGeometry, frame: _Frame, rows: list[list[list[Segment]]]) -> bool:
    if len(rows) < 2:
        return False
    fills = [fill for fill in geometry.layout.fill_rects if fill.y2 - fill.y1 >= _MIN_CELL]
    columns = list(zip(frame.xs, frame.xs[1:], strict=False))
    top, below = frame.ys[:2]
    colors = {_background(fills, left, top, right, below) for left, right in columns}
    if len(colors) == 1:
        (color,) = colors
        if all(
            _color_contrast(color, _background(fills, left, a, right, b))
            for a, b in zip(frame.ys[1:], frame.ys[2:], strict=False)
            for left, right in columns
        ):
            return True
    # A text color is evidence only when it is independently present in every
    # header cell and contrasts with all body text, not just a single link.
    text_colors = {span.color.lower() if span.color else None for cell in rows[0] for span in cell}
    if len(text_colors) == 1:
        (text_color,) = text_colors
        return all(_color_contrast(text_color, span.color) for row in rows[1:] for cell in row for span in cell)
    return False


def _narrative_boxes(frame: _Frame, rows: list[list[list[Segment]]]) -> bool:
    if len(frame.xs) != 3:
        return False
    left, divider, right = frame.xs
    # A narrow prompt followed by a ruled box filled with prose is a section,
    # not a series of records. Leave its fragments to normal assembly.
    return (
        (divider - left) / (right - left) < 0.18
        and all(b - a >= 95 for a, b in zip(frame.ys, frame.ys[1:], strict=False))
        and all(
            sum(len(span.text.strip()) for span in label) <= 30
            and len(prose) >= 3
            and median(len(span.text.strip()) for span in prose) >= 25
            for label, prose in rows
        )
    )


def _rectangular(geometry: _PageGeometry, frames: list[tuple[_Frame, bool]]) -> list[GeometryTable]:
    tables: list[GeometryTable] = []
    for frame, closed in frames:
        grid = _cells(geometry, frame)
        if grid and all(all(row) for row in grid) and not _narrative_boxes(frame, grid):
            header = _has_header(geometry, frame, grid)
            if closed or header:
                ids = [_ids(row) for row in grid]
                tables.append(GeometryTable(geometry.number, "rectangular", ids[0] if header else [], ids[header:], []))
    return tables


def _seamed_rows(geometry: _PageGeometry) -> list[GeometryTable]:
    """Recognize label/value rows with native painted seams instead of vertical rules."""
    rules = {
        _Rule((r.y1 + r.y2) / 2, r.x1, r.x2)
        for r in geometry.layout.fill_rects
        if r.y2 - r.y1 <= 1.6 and r.x2 - r.x1 >= _MIN_CELL
    }
    seams: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    for stripe in _axis_groups(rules):
        stripe.sort(key=lambda r: r.start)
        for index, (first, second) in enumerate(zip(stripe, stripe[1:], strict=False)):
            if abs(first.end - second.start) > _AXIS_TOL:
                continue
            # Only two fragments per rule; do not guess which joins in a
            # wider fragmented line represent real column boundaries.
            if index and first.start - stripe[index - 1].end <= _JOIN_TOL:
                continue
            if index + 2 < len(stripe) and stripe[index + 2].start - second.end <= _JOIN_TOL:
                continue
            key = (round(first.start, 1), round((first.end + second.start) / 2, 1), round(second.end, 1))
            seams[key].append(first.axis)
    tables: list[GeometryTable] = []
    for (left, seam, right), rules_y in seams.items():
        if not 5 <= len(rules_y) <= _MAX_CELLS // 2:
            continue
        steps = [b - a for a, b in zip(rules_y, rules_y[1:], strict=False)]
        if not 14 <= min(steps) <= max(steps) <= 45 or max(steps) - min(steps) > 1.5:
            continue
        top = rules_y[0] - median(steps)
        if top < 0 or any(abs(r.axis - seam) <= _JOIN_TOL and r.crosses(top, rules_y[-1]) for r in geometry.vertical):
            continue
        grid = _cells(geometry, _Frame([left, seam, right], [top, *rules_y]))
        if not grid or any(
            len(row[0]) != 1 or len(row[1]) != 1 or abs(row[0][0].y - row[1][0].y) > 1.5 for row in grid
        ):
            continue
        x_offsets = [[row[j][0].x for row in grid] for j in range(2)]
        baselines = [axis - row[0][0].y for row, axis in zip(grid, rules_y, strict=True)]
        if all(max(offsets) - min(offsets) <= 2 for offsets in [*x_offsets, baselines]):
            tables.append(GeometryTable(geometry.number, "rectangular", [], [_ids(row) for row in grid], []))
    return tables


def _adjacent(geometry: _PageGeometry, frames: list[tuple[_Frame, bool]]) -> list[GeometryTable]:
    open_sides: list[tuple[_Frame, list[list[list[int]]]]] = []
    for frame, closed in frames:
        left, right = frame.xs[0], frame.xs[-1]
        top, bottom = frame.ys[0], frame.ys[-1]
        if closed or not 85 <= right - left < geometry.layout.width * 0.47 or len(frame.xs) > 9:
            continue
        if any(not 14 <= hi - lo <= 100 for lo, hi in zip(frame.ys, frame.ys[1:], strict=False)):
            continue
        if any(
            r.crosses(top, bottom) and any(abs(r.axis - edge) <= _JOIN_TOL for edge in (left, right))
            for r in geometry.vertical
        ):
            continue
        if any(
            abs(fill.y1 - top) < 2
            and left <= fill.x1 < right
            and 12 < fill.y2 - fill.y1 < 42
            and fill.x2 - fill.x1 > 18
            for fill in geometry.layout.fill_rects
        ):
            continue
        grid = _cells(geometry, frame)
        if grid and all(all(row) for row in grid):
            open_sides.append((frame, [_ids(row) for row in grid]))
    return [
        GeometryTable(geometry.number, "adjacent-open-grid", [], rows, [])
        for frame, rows in open_sides
        if any(
            abs(frame.ys[0] - other.ys[0]) <= _OPEN_TOL
            and 9 < max(frame.xs[0] - other.xs[-1], other.xs[0] - frame.xs[-1]) < 90
            for other, _ in open_sides
        )
    ]


def _open_top_rows(geometry: _PageGeometry) -> list[tuple[list[float], list[list[Segment]]]]:
    candidates: list[tuple[list[float], list[list[Segment]]]] = []
    for band in _width_bands(geometry.horizontal):
        bottom = band[0]
        left, right = bottom.start, bottom.end
        if right - left < 85:
            continue
        # Keep raw divider ends: merging dividers across the first horizontal
        # edge would erase the evidence of where the open top row begins.
        starts = [
            r.start
            for r in geometry.vertical_sources
            if left + 8 < r.axis < right - 8 and 14 <= bottom.axis - r.start <= 75 and abs(r.end - bottom.axis) <= 2
        ]
        if not starts:
            continue
        top = median(starts)
        if top > geometry.layout.height * 0.08:
            continue
        xs = [left]
        xs.extend(r.axis for r in geometry.vertical if left + 8 < r.axis < right - 8 and r.covers(top, bottom.axis))
        xs.append(right)
        if not 5 <= len(xs) <= 9 or any(b - a < _MIN_CELL for a, b in zip(xs, xs[1:], strict=False)):
            continue
        grid = _cells(geometry, _Frame(xs, [top, bottom.axis]))
        if grid and not grid[0][0] and not grid[0][1] and all(grid[0][2:]):
            candidates.append((xs, grid[0]))
    return candidates


def _continued(before: _PageGeometry, after: _PageGeometry) -> list[GeometryTable]:
    if after.number != before.number + 1:
        return []
    next_rows = _open_top_rows(after)
    if not next_rows:
        return []
    late_frames = [
        frame
        for frame, _ in _solid_frames(before, split_at_caps=True)
        if frame.ys[-1] >= before.layout.height * 0.88 and 4 <= len(frame.xs) <= 8 and len(frame.ys) >= 4
    ]
    tables: list[GeometryTable] = []
    for frame in late_frames:
        xs, top = frame.xs, frame.ys[0]
        matches = [
            row[1:]
            for cuts, row in next_rows
            if len(cuts) == len(xs) + 1 and all(abs(a - b) <= _OPEN_TOL for a, b in zip(xs, cuts[1:], strict=True))
        ]
        if len(matches) != 1:
            continue
        grid = _cells(before, frame)
        if not grid or not all(not row[0] and all(row[1:]) for row in grid):
            continue
        caps = [r.axis for r in _wide_caps(before, xs[0], xs[-1]) if 14 <= top - r.axis <= 55]
        if not caps:
            continue
        # The wider cap marks a separate leading band; the left-side label is
        # outside the body lattice and must stay a separate source fragment.
        header = _cells(before, _Frame(xs, [max(caps), top]))
        if header and all(header[0]):
            tables.append(
                GeometryTable(
                    before.number,
                    "continued-open-grid",
                    _ids(header[0]),
                    [_ids(row) for row in [*grid, matches[0]]],
                    [],
                    continuation_page=after.number,
                )
            )
    return tables


def detect_tables(pages: list[PdfPage], segments: list[Segment]) -> list[GeometryTable]:
    """Return disjoint table candidates for pages with PDF layout metadata."""
    by_page: dict[int, list[Segment]] = defaultdict(list)
    for span in segments:
        by_page[span.page].append(span)
    used: set[int] = set()
    tables: list[GeometryTable] = []
    adjacent: list[GeometryTable] = []
    continued: list[GeometryTable] = []
    previous: _PageGeometry | None = None

    def add(candidates: list[GeometryTable]) -> None:
        for table in candidates:
            refs = {ref for row in [table.headers, *table.rows] for cell in row for ref in cell} | set(table.marks)
            if refs and not refs & used:
                tables.append(table)
                used.update(refs)

    for page in pages:
        geometry = _page_geometry(page, by_page[page.number])
        if geometry is None:
            continue
        fills, dots = _colored_fills(geometry), _dot_rules(geometry)
        frames = _solid_frames(geometry)
        add(_pairs(geometry, fills, dots))
        add(_dotted(geometry, fills, dots))
        add(_rectangular(geometry, frames))
        add(_seamed_rows(geometry))
        adjacent.extend(_adjacent(geometry, frames))
        if previous is not None:
            continued.extend(_continued(previous, geometry))
        previous = geometry
    add(adjacent)
    add(continued)
    return tables
