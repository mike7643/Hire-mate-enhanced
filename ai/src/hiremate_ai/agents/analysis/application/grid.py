"""그린 선이 모든 셀을 닫는 직사각형 격자만 찾는다."""

from dataclasses import dataclass

from hiremate_ai.contract import PdfGridLine

# 채운 사각형의 중심선을 쓰므로 교점에서 약간 떨어진 선도 이어질 수 있다.
_EPS = 1.5
_MIN_CELL = 6
_MAX_CELLS = 256


@dataclass(frozen=True)
class _Rule:
    axis: float
    start: float
    end: float


def _merge(rules: list[_Rule]) -> list[_Rule]:
    axes: list[list[_Rule]] = []
    for rule in sorted(rules, key=lambda r: r.axis):
        if not axes or rule.axis - axes[-1][0].axis > _EPS:
            axes.append([])
        axes[-1].append(rule)
    merged: list[_Rule] = []
    for group in axes:
        axis = sum(rule.axis for rule in group) / len(group)
        intervals = sorted((rule.start, rule.end) for rule in group)
        lo, hi = intervals[0]
        for start, end in intervals[1:]:
            if start > hi + _EPS:
                merged.append(_Rule(axis, lo, hi))
                lo, hi = start, end
            else:
                hi = max(hi, end)
        merged.append(_Rule(axis, lo, hi))
    return merged


def _same_width(first: _Rule, other: _Rule) -> bool:
    return abs(first.start - other.start) <= _EPS and abs(first.end - other.end) <= _EPS


def closed_grids(lines: list[PdfGridLine]) -> list[tuple[list[float], list[float]]]:
    horizontal: list[_Rule] = []
    vertical: list[_Rule] = []
    for line in lines:
        if line.y1 == line.y2 and abs(line.x2 - line.x1) >= _MIN_CELL:
            horizontal.append(_Rule(line.y1, min(line.x1, line.x2), max(line.x1, line.x2)))
        elif line.x1 == line.x2 and abs(line.y2 - line.y1) >= _MIN_CELL:
            vertical.append(_Rule(line.x1, min(line.y1, line.y2), max(line.y1, line.y2)))
    across, down = _merge(horizontal), _merge(vertical)
    bands: list[list[_Rule]] = []
    for rule in across:
        group = next((group for group in bands if _same_width(group[0], rule)), None)
        if group is None:
            bands.append([rule])
        else:
            group.append(rule)
    result: list[tuple[list[float], list[float]]] = []
    for band in bands:
        if len(band) < 4:
            continue
        left, right = band[0].start, band[0].end
        top, bottom = band[0].axis, band[-1].axis
        rows = [r.axis for r in band]
        horizontal_inside = [
            r for r in across if top < r.axis < bottom and r.start < right - _EPS and r.end > left + _EPS
        ]
        if any(r not in band for r in horizontal_inside):
            continue
        cuts = [
            v for v in down if left - _EPS <= v.axis <= right + _EPS and v.start < bottom - _EPS and v.end > top + _EPS
        ]
        if len(cuts) < 3 or any(v.start > top + _EPS or v.end < bottom - _EPS for v in cuts):
            continue
        columns = [v.axis for v in cuts]
        if abs(columns[0] - left) > _EPS or abs(columns[-1] - right) > _EPS:
            continue
        if (len(columns) - 1) * (len(rows) - 1) > _MAX_CELLS:
            continue
        if any(b - a < _MIN_CELL for axes in (columns, rows) for a, b in zip(axes, axes[1:], strict=False)):
            continue
        result.append((columns, rows))
    return result
