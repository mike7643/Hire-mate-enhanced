"""정제 문서를 검색할 수 있는 단위로 나눈다."""

import re
from dataclasses import dataclass

from hiremate_ai.agents.guardrails import personal_fact
from hiremate_ai.contract import (
    DocumentBlock,
    DocumentList,
    DocumentListItem,
    DocumentParagraph,
    DocumentSection,
    DocumentTable,
    NormalizedDocument,
)

MAX_CHARS = 1200
_CONTACT = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:\+?\d[\d\s()-]{7,}\d)")
_BOUNDARY = re.compile(r"(?<=[.!?。])\s+|\n+")
# 주소는 지역으로 지원자를 찾는 데 쓰일 수 있어 색인하지 않는다 (규칙 6). 업무 글의 지역명까지 빼지 않도록 항목 이름이
# 주소인 칸, 우편번호 표기, '주소:'로 시작하는 글만 본다.
_ADDRESS_LABEL = re.compile(r"\s*(?:현\s*)?(?:주\s*소|주소지|거주지|자택\s*주소)\s*[:：]?\s*")
_ADDRESS = re.compile(r"\(\s*우\s*[:.]?\s*\d{5}\s*\)|^\s*(?:현\s*)?(?:주\s*소|주소지|거주지)\s*[:：]")
# 연락처 section은 전화, 이메일과 함께 사는 곳이 적혀 있어 통째로 색인하지 않는다.
_CONTACT_SECTION = re.compile(
    r"(?i)\s*(?:contact(?:\s+info)?|연락처|인적\s*사항|개인\s*정보|personal\s+info(?:rmation)?)\s*"
)


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    evidence: str
    block_id: str
    pages: tuple[int, ...]


def document_chunks(document: NormalizedDocument, *, blind: bool, applicant_name: str) -> list[DocumentChunk]:
    """문맥은 임베딩에 넣고 근거와 페이지는 별도로 보존한다."""
    result: list[DocumentChunk] = []

    def safe(text: str) -> bool:
        return bool(text.strip()) and not (
            (applicant_name and applicant_name in text)
            or _CONTACT.search(text)
            or _ADDRESS.search(text)
            or personal_fact(text, blind=blind)
        )

    def add(text: str, path: list[str], block_id: str, pages: list[int]) -> None:
        if not safe(text):
            return
        prefix = " > ".join(path)
        max_body = max(100, MAX_CHARS - len(prefix) - 1)
        for part in _split(text, max_body):
            value = f"{prefix}\n{part}" if prefix else part
            result.append(DocumentChunk(value, part, block_id, tuple(pages)))

    def items(values: list[DocumentListItem], path: list[str], parents: list[str]) -> None:
        for item in values:
            if not safe(item.text):
                continue
            add(" > ".join([*parents, item.text]), path, item.id, item.pages)
            items(item.items, path, [*parents, item.text])

    def visit(block: DocumentBlock, path: list[str]) -> None:
        if isinstance(block, DocumentSection):
            if block.title and _CONTACT_SECTION.fullmatch(block.title):
                return
            # 가리거나 검색할 수 없는 제목도 자식의 검색 문맥에 넣지 않는다.
            next_path = [*path, block.title] if block.title and safe(block.title) else path
            for child in block.blocks:
                visit(child, next_path)
        elif isinstance(block, DocumentParagraph):
            add(block.text, path, block.id, block.pages)
        elif isinstance(block, DocumentList):
            items(block.items, path, [])
        elif isinstance(block, DocumentTable):
            # '주소 | 값' 형태가 머리글이면 다음 칸은 열 제목인지 주소 값인지 구분할 수 없다.
            if any(
                _ADDRESS_LABEL.fullmatch(header) and following
                for header, following in zip(block.headers, block.headers[1:], strict=False)
            ):
                return
            table_path = [*path, block.title] if block.title and safe(block.title) else path
            for row in block.rows:
                cells = _without_address(block.headers, row)
                if not all(safe(cell) for _, cell in cells):
                    continue
                if not all(not header or safe(header) for header in block.headers):
                    continue
                text = " | ".join(f"{header}: {cell}" if header else cell for header, cell in cells)
                add(text, table_path, block.id, block.pages)

    title = document.document.title
    path = [title] if title and safe(title) else []
    for block in document.blocks:
        visit(block, path)
    return result


def _without_address(headers: list[str], row: list[str]) -> list[tuple[str, str]]:
    """표 행에서 주소 칸을 뺀 (열 제목, 값).

    열 제목이 주소면 그 칸을, 본문에 '주소' 칸이 있으면 그 칸과 다음 값 칸을 뺀다.
    """
    cells: list[tuple[str, str]] = []
    skip = False
    for index, cell in enumerate(row):
        header = headers[index] if index < len(headers) else ""
        if skip:
            skip = False
            continue
        if _ADDRESS_LABEL.fullmatch(cell):
            skip = True
            continue
        if header and _ADDRESS_LABEL.fullmatch(header):
            continue
        if cell:
            cells.append((header, cell))
    return cells


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            boundaries = list(_BOUNDARY.finditer(text, start, end))
            if boundaries:
                end = boundaries[-1].end()
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return parts
