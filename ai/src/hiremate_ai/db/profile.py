"""지원자 프로필 (FR-08, 13, 14).

분석 Agent(지원서 · 평가 · 면접)가 낸 결과를 지원자별로 쌓아 두고, Root Agent가 조건으로 지원자를 찾을 때 쓴다.
프로필은 그래프(Neo4j)에 저장한다. 저장과 조회는 db.profile_store가 한다.

그래프 모양
    (ApplicationProfile)-[:HAS_CLAIM]->(Claim)-[:SUPPORTED_BY]->(Evidence)-[:FROM_SOURCE]->(Source)
    (Claim)-[:FROM_SOURCE]->(Source)                근거 문장이 없는 항목
    (ApplicationProfile)-[:HAS_EXPERIENCE]->(Experience)<-[:ABOUT_EXPERIENCE]-(Claim)
    (Claim)-[:USES_SKILL]->(Skill)                  항목에 나온 기술 · 도구
    (Claim)-[:IN_DOMAIN]->(Domain)                  항목의 업무 분야

- ApplicationProfile: 지원서 하나. 경력 연수, 마지막 회사처럼 조건 검색에 쓰는 값
- Claim: 프로필 항목 한 줄. 분류, 내용, 확인 상태, 만든 Agent, 전형, 수행 주체(actor), 한 일의 종류(action)
- Evidence: 원문에서 찾은 근거 문장. Source: 그 문장이 나온 자료 (이력서, 평가 결과, 면접 답변 등)
- Experience: 그 지원서의 회사 · 프로젝트 하나. 같은 경험에서 나온 항목들이 함께 가리킨다. 지원자끼리 합치지 않는다
- Skill, Domain: 같은 기업 안에서 항목들이 함께 가리키는 기술 · 업무 분야

이름이 같은지는 대소문자와 공백만 무시하고 본다. 표기가 다른 같은 것(카프카와 Kafka)은 합치지 않는다.
노드의 이름은 처음 저장된 표기다. 기술 이름과 회사 · 프로젝트 이름은 원문에 적힌 것만 들어오고(분석 Agent가 확인),
업무 분야는 모델이 근거 문장으로 분류한 값이다.

이 모듈의 CandidateProfile, ProfileItem은 그래프에서 읽은 값이다.

규칙
- 분석 Agent가 낸 값을 그대로 저장한다. 이 모듈은 점수를 다시 매기거나 문장을 고치지 않는다.
- 이름, 연락처는 backend에 있으므로 저장하지 않고 지원서 ID로만 연결한다.
- 나이, 성별, 출신 지역, 혼인 여부, 신체 조건은 저장하지 않는다 (아키텍처 규칙 6).
- 조회는 항상 기업 · 채용 범위로 거른다 (아키텍처 규칙 4). Skill, Domain 노드도 기업마다 따로 둔다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class ProfileStatus(StrEnum):
    """프로필 분석 상태."""

    DONE = "DONE"  # 분석을 마쳤다
    CHECK = "CHECK"  # 문서 본문이 비어 있는 등 근거가 부족해 담당자 확인이 필요하다
    FAILED = "FAILED"  # 분석이 실패했다. 마감 전에 다시 돌린다


class ItemCategory(StrEnum):
    """프로필 항목 분류. 새 분류가 필요하면 팀과 정한 뒤 여기에 더한다."""

    EXPERIENCE = "EXPERIENCE"  # 직무 경험 (예: 카드 결제 승인 시스템 개발 4년)
    DOMAIN = "DOMAIN"  # 업무 도메인 (예: 금융, 이커머스). "금융 경험자" 검색에 쓴다
    SKILL = "SKILL"  # 기술 · 역량 (예: Java, Spring, Kafka)
    QUALIFICATION = "QUALIFICATION"  # 공고 자격 조건 대조 결과 (예: 경력 3년 이상 충족)
    COMPANY_FIT = "COMPANY_FIT"  # 회사 인재상 · 문화와 관련된 경험
    STRENGTH = "STRENGTH"  # 평가 · 면접에서 드러난 강점
    CHECK = "CHECK"  # 근거가 부족해 확인이 필요한 점


class Verification(StrEnum):
    """프로필 항목의 확인 상태. 분류(무슨 내용인가)와 따로 '얼마나 확실한가'를 적는다 (아키텍처 3장, NFR-03).

    근거 문장이 없는 항목은 언제나 UNCONFIRMED다 ("근거가 없으면 '미확인'"). 값은 db.profile_store가 정한다.
    """

    CONFIRMED = "CONFIRMED"  # 평가 · 면접에서 확인됨
    EVIDENCED = "EVIDENCED"  # 원문(제출 문서 · 평가 자료 · 면접 답변)에 근거 문장이 있음
    UNCONFIRMED = "UNCONFIRMED"  # 근거가 없거나 아직 확인되지 않음 ('미확인')
    CONFLICT = "CONFLICT"  # 자료끼리 어긋남 ('상충'. 예: 서류와 면접 답변이 다름)


class ItemAgent(StrEnum):
    """항목을 만든 분석 Agent. 같은 Agent가 다시 분석하면 그 Agent의 항목만 바꾼다."""

    APPLICATION = "application"  # 지원서 분석
    ASSESSMENT = "assessment"  # 평가 분석
    INTERVIEW = "interview"  # 면접 분석


@dataclass(frozen=True)
class ProfileItem:
    """프로필 항목 한 줄 (Claim). 내용과 근거 문장, 출처, 확인 상태를 함께 둔다 (아키텍처 규칙 7)."""

    id: UUID
    category: str  # ItemCategory 값
    agent: str  # ItemAgent 값
    content: str  # 정리한 내용 (예: "금융 (카드 결제 · 정산)")
    evidence: str | None  # 원문에서 찾은 근거 문장. 없으면 None
    verification: str  # Verification 값 (확인 상태)
    source: str  # 출처 (예: 이력서, 자기소개서, 평가 결과, 면접 답변)
    stage_id: UUID | None  # 평가 · 면접 항목이면 그 전형 ID. 지원서 항목은 None
    created_at: datetime
    experience: str | None = None  # 이 항목이 속한 회사 · 프로젝트 이름. 없으면 None
    actor: str | None = None  # 수행 주체 (SELF, TEAM, UNCLEAR). 모르면 None
    action: str | None = None  # 한 일의 종류 (DEVELOP, OPERATE 등). 없으면 None
    skills: tuple[str, ...] = ()  # 항목에 나온 기술 · 도구 이름. 이름 순서
    domains: tuple[str, ...] = ()  # 항목의 업무 분야. 이름 순서


@dataclass(frozen=True)
class CandidateProfile:
    """지원자 한 명의 프로필 (ApplicationProfile). 지원서(application_id)마다 하나다."""

    # backend의 지원서 · 기업 · 채용 ID
    application_id: UUID
    company_id: UUID
    recruitment_id: UUID
    status: str  # ProfileStatus 값
    # 조건 검색에 쓰는 값. 분석 전이거나 찾지 못하면 None
    career_years: int | None
    last_company: str | None
    summary: str
    created_at: datetime
    updated_at: datetime
    # 저장된 순서
    items: list[ProfileItem] = field(default_factory=list)
