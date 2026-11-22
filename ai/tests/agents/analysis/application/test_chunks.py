from uuid import uuid4

from hiremate_ai.agents.analysis.application.chunks import document_chunks
from hiremate_ai.contract import NormalizedDocument


def test_chunks_preserve_context_pages_and_table_rows_without_visual_notices() -> None:
    document = NormalizedDocument.model_validate(
        {
            "schemaVersion": "1.0",
            "document": {"id": str(uuid4()), "type": "portfolio", "title": "포트폴리오", "pageCount": 2},
            "blocks": [
                {
                    "id": "s1",
                    "type": "section",
                    "title": "결제 프로젝트",
                    "pages": [1, 2],
                    "blocks": [
                        {"id": "p1", "type": "paragraph", "pages": [1], "text": "승인 서버를 개선했다."},
                        {
                            "id": "l1",
                            "type": "list",
                            "pages": [1],
                            "items": [
                                {
                                    "id": "i1",
                                    "text": "결제 처리",
                                    "pages": [1],
                                    "items": [{"id": "i2", "text": "오류율을 낮췄다.", "pages": [1], "items": []}],
                                }
                            ],
                        },
                        {
                            "id": "t1",
                            "type": "table",
                            "pages": [2],
                            "title": "처리 속도",
                            "headers": ["항목", "결과"],
                            "rows": [["응답 시간", "20% 단축"], ["처리량", "30% 향상"]],
                            "reconstruction": "inferred",
                        },
                        {
                            "id": "v1",
                            "type": "visual_notice",
                            "pages": [2],
                            "visualType": "chart",
                            "detection": "suspected",
                            "message": "원본 PDF를 확인하세요.",
                        },
                    ],
                }
            ],
        }
    )
    chunks = document_chunks(document, blind=False, applicant_name="홍길동")
    texts = [chunk.text for chunk in chunks]
    assert any("포트폴리오 > 결제 프로젝트" in text and "승인 서버를 개선했다." in text for text in texts)
    assert any("결제 처리" in text and "오류율을 낮췄다." in text for text in texts)
    assert any("항목: 응답 시간" in text and "결과: 20% 단축" in text for text in texts)
    assert any(chunk.block_id == "t1" and chunk.pages == (2,) for chunk in chunks)
    assert not any("원본 PDF를 확인하세요" in text for text in texts)


def test_does_not_index_personal_data_even_as_context() -> None:
    document = NormalizedDocument.model_validate(
        {
            "schemaVersion": "1.0",
            "document": {"id": str(uuid4()), "type": "resume", "title": "홍길동 이력서", "pageCount": 1},
            "blocks": [
                {
                    "id": "s",
                    "type": "section",
                    "title": "학력",
                    "pages": [1],
                    "blocks": [{"id": "p", "type": "paragraph", "pages": [1], "text": "한빛대학교 졸업"}],
                },
                {"id": "p2", "type": "paragraph", "pages": [1], "text": "이메일 test@example.com"},
                {"id": "p3", "type": "paragraph", "pages": [1], "text": "결제 시스템을 개발했다."},
            ],
        }
    )
    chunks = document_chunks(document, blind=True, applicant_name="홍길동")
    assert len(chunks) == 1
    assert "결제 시스템을 개발했다." in chunks[0].text
    assert "홍길동" not in chunks[0].text
    assert all("test@example.com" not in chunk.text and "한빛대학교" not in chunk.text for chunk in chunks)


def test_artifacts_are_kept_in_the_document_but_not_indexed() -> None:
    document = NormalizedDocument.model_validate(
        {
            "schemaVersion": "1.0",
            "document": {"id": str(uuid4()), "type": "resume", "title": None, "pageCount": 1},
            "blocks": [
                {"id": "p1", "type": "paragraph", "pages": [1], "text": "소성·수세·코팅 공정을 개발했다."},
                {"id": "a1", "type": "artifact", "pages": [1], "role": "page_number", "text": "- 1 -"},
            ],
        }
    )
    chunks = document_chunks(document, blind=False, applicant_name="김민지")
    assert [chunk.text for chunk in chunks] == ["소성·수세·코팅 공정을 개발했다."]


def test_addresses_and_contact_sections_are_not_indexed_but_work_places_are() -> None:
    document = NormalizedDocument.model_validate(
        {
            "schemaVersion": "1.0",
            "document": {"id": str(uuid4()), "type": "resume", "title": None, "pageCount": 1},
            "blocks": [
                {
                    "id": "t1",
                    "type": "table",
                    "pages": [1],
                    "title": None,
                    "headers": [],
                    "rows": [
                        ["지원분야", "생산관리"],
                        ["주소", "(우: 40653) 인천 부평구 라온길 178, 1101호"],
                        ["경력", "11년 6개월", "주소", "인천 부평구"],
                    ],
                    "reconstruction": "inferred",
                },
                {
                    "id": "t2",
                    "type": "table",
                    "pages": [1],
                    "title": None,
                    "headers": ["회사", "거주지"],
                    "rows": [["새솔정밀", "경남 창원시"]],
                    "reconstruction": "inferred",
                },
                {
                    "id": "s1",
                    "type": "section",
                    "title": "CONTACT",
                    "pages": [1],
                    "blocks": [
                        {
                            "id": "l1",
                            "type": "list",
                            "pages": [1],
                            "items": [{"id": "i1", "text": "경남 창원시", "pages": [1], "items": []}],
                        }
                    ],
                },
                {"id": "p1", "type": "paragraph", "pages": [1], "text": "주소: 서울 마포구 늘봄로 1"},
                {"id": "p2", "type": "paragraph", "pages": [1], "text": "인천 공장에서 사출 생산 실적을 관리했다."},
            ],
        }
    )
    chunks = [chunk.text for chunk in document_chunks(document, blind=False, applicant_name="윤주연")]
    assert chunks == [
        "지원분야 | 생산관리",
        "경력 | 11년 6개월",
        "회사: 새솔정밀",
        "인천 공장에서 사출 생산 실적을 관리했다.",
    ]
