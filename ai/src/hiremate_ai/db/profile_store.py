"""지원자 프로필 저장과 조회 (FR-08). 프로필 그래프(db.profile)를 읽고 쓰는 Cypher는 이 모듈에만 둔다.

분석 Agent(지원서 · 평가 · 면접)가 분석을 마치면 save_profile을 불러 결과를 프로필에 넣는다.

규칙
- 받은 값을 그대로 저장한다. 점수를 다시 매기거나 문장을 고치지 않는다 (다른 역할의 결과를 재해석하지 않음).
- 같은 Agent가 같은 전형을 다시 분석하면 그 Agent · 전형의 항목만 새 결과로 바꾸고 다른 항목은 그대로 둔다.
- 제출 문서를 함께 주면 근거 문장이 원문에 있는지 확인하고, 없으면 그 항목을 버린다 (아키텍처 규칙 7).
- 항목마다 확인 상태를 적는다. 넘기지 않으면 근거 문장이 있으면 근거 있음, 없으면 미확인이다.
  근거 문장이 없는 항목은 무엇을 넘겼든 미확인이다 ("근거가 없으면 '미확인'").
- 저장 한 번이 그래프 트랜잭션 하나다. PostgreSQL과 함께 쓰는 쪽은 프로필을 먼저 저장하고 나서 커밋한다.
  커밋이 실패해도 같은 분석을 다시 저장하면 같은 항목으로 바뀌므로 프로필이 어긋난 채 남지 않는다.
- 조회는 기업 · 채용이 맞는 프로필만 돌려준다 (아키텍처 규칙 4).

사용 예 (지원서 분석 Agent)
    profile = await save_profile(
        graph,
        company_id=company_id,
        recruitment_id=applicant.info.recruitment_id,
        application_id=applicant.info.application_id,
        agent=ItemAgent.APPLICATION,
        fields=ProfileFields(status=ProfileStatus.DONE, career_years=4, ...),
        items=[NewItem(category=ItemCategory.DOMAIN, content="금융 (카드 결제)", evidence="...", source="이력서")],
        documents=applicant.documents,
    )
"""

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from neo4j import AsyncDriver, AsyncManagedTransaction
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from hiremate_ai.agents.guardrails import evidence_source
from hiremate_ai.contract import ApplicantDocument
from hiremate_ai.db.graph import ensure_schema
from hiremate_ai.db.profile import CandidateProfile, ItemAgent, ItemCategory, ProfileItem, ProfileStatus, Verification

logger = logging.getLogger(__name__)

# 프로필 노드를 만들거나 찾는다. SET으로 노드를 잠가 같은 지원서의 저장이 차례로 실행되게 한다
_MERGE_PROFILE = """
MERGE (p:ApplicationProfile {applicationId: $applicationId})
ON CREATE SET p.companyId = $companyId, p.recruitmentId = $recruitmentId, p.status = $status, p.summary = '',
              p.createdAt = $now
SET p.updatedAt = $now
RETURN p.companyId AS companyId, p.recruitmentId AS recruitmentId
"""

_SET_FIELDS = """
MATCH (p:ApplicationProfile {applicationId: $applicationId})
SET p.status = $status, p.careerYears = $careerYears, p.lastCompany = $lastCompany, p.summary = $summary
"""

_SET_TIMES = """
MATCH (p:ApplicationProfile {applicationId: $applicationId})
SET p.createdAt = $createdAt, p.updatedAt = $updatedAt
"""

# 근거 문장은 항목과 함께 지운다
_DELETE_CLAIMS = """
MATCH (:ApplicationProfile {applicationId: $applicationId})-[:HAS_CLAIM]->(c:Claim {agent: $agent})
WHERE c.stageId = $stageId OR ($stageId IS NULL AND c.stageId IS NULL)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)
DETACH DELETE c, e
"""

# 값을 바꾸지 않고 프로필 노드를 잠근다. 옮기는 동안 같은 지원서의 저장이 끼어들지 못한다
_LOCK_PROFILE = """
MATCH (p:ApplicationProfile {applicationId: $applicationId})
SET p.updatedAt = p.updatedAt
RETURN p.companyId AS companyId, p.recruitmentId AS recruitmentId
"""

# 프로필에 항목이 있는 (Agent, 전형)
_CLAIM_GROUPS = """
MATCH (:ApplicationProfile {applicationId: $applicationId})-[:HAS_CLAIM]->(c:Claim)
RETURN DISTINCT c.agent AS agent, c.stageId AS stageId
"""

_CREATE_CLAIMS = """
MATCH (p:ApplicationProfile {applicationId: $applicationId})
UNWIND $items AS item
CREATE (p)-[:HAS_CLAIM]->(c:Claim {
    id: item.id, category: item.category, agent: item.agent, content: item.content,
    verification: item.verification, seq: item.seq, createdAt: item.createdAt
})
SET c.stageId = item.stageId
MERGE (s:Source {key: item.sourceKey})
ON CREATE SET s.applicationId = $applicationId, s.companyId = p.companyId, s.recruitmentId = p.recruitmentId,
              s.kind = item.source, s.stageId = item.stageId
FOREACH (_ IN CASE WHEN item.evidence IS NULL THEN [] ELSE [1] END |
    CREATE (c)-[:SUPPORTED_BY]->(:Evidence {id: item.evidenceId, quote: item.evidence})-[:FROM_SOURCE]->(s))
FOREACH (_ IN CASE WHEN item.evidence IS NULL THEN [1] ELSE [] END |
    CREATE (c)-[:FROM_SOURCE]->(s))
FOREACH (_ IN CASE WHEN item.category = 'SKILL' THEN [1] ELSE [] END |
    MERGE (k:Skill {key: item.conceptKey})
    ON CREATE SET k.companyId = p.companyId, k.name = item.content
    CREATE (c)-[:USES_SKILL]->(k))
FOREACH (_ IN CASE WHEN item.category = 'DOMAIN' THEN [1] ELSE [] END |
    MERGE (d:Domain {key: item.conceptKey})
    ON CREATE SET d.companyId = p.companyId, d.name = item.content
    CREATE (c)-[:IN_DOMAIN]->(d))
"""

# 항목이 모두 빠진 자료 노드는 남기지 않는다
_DELETE_UNUSED_SOURCES = """
MATCH (s:Source {applicationId: $applicationId})
WHERE NOT (s)<-[:FROM_SOURCE]-()
DELETE s
"""

_LOAD = """
MATCH (p:ApplicationProfile)
WHERE p.applicationId IN $applicationIds AND p.companyId = $companyId AND p.recruitmentId = $recruitmentId
OPTIONAL MATCH (p)-[:HAS_CLAIM]->(c:Claim)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Evidence)-[:FROM_SOURCE]->(quoted:Source)
OPTIONAL MATCH (c)-[:FROM_SOURCE]->(stated:Source)
WITH p, c, e, coalesce(quoted, stated) AS s
ORDER BY c.createdAt, c.seq
RETURN p AS profile, collect(CASE WHEN c IS NULL THEN null ELSE {
    id: c.id, category: c.category, agent: c.agent, content: c.content, verification: c.verification,
    stageId: c.stageId, createdAt: c.createdAt, evidence: e.quote, source: s.kind
} END) AS items
"""

_OWNER = """
MATCH (p:ApplicationProfile {applicationId: $applicationId})
RETURN p.companyId AS companyId, p.recruitmentId AS recruitmentId
"""


class _Input(BaseModel):
    """필드는 snake_case, JSON 이름은 camelCase다 (contract.py와 같은 규칙)."""

    model_config = ConfigDict(
        alias_generator=to_camel, validate_by_name=True, validate_by_alias=True, serialize_by_alias=True, frozen=True
    )


class ProfileFields(_Input):
    """지원서 분석이 정하는 지원자 요약 값. 평가 · 면접 분석은 넘기지 않는다."""

    status: ProfileStatus
    career_years: int | None
    last_company: str | None
    summary: str = ""


class NewItem(_Input):
    """프로필에 넣을 항목 한 줄."""

    category: ItemCategory
    content: str
    evidence: str | None = None  # 원문 근거 문장. 근거가 없는 항목(확인 필요 등)은 None
    source: str  # 출처 (이력서, 자기소개서, 평가 결과, 면접 답변 등)
    # 확인 상태. 평가 · 면접 분석이 확인됨 · 상충을 알 때 넘긴다. 비우면 근거 문장으로 정한다
    verification: Verification | None = None


class ProfileScopeError(ValueError):
    """이미 있는 프로필과 기업 · 채용이 다를 때. 다른 기업의 프로필을 덮어쓰지 않게 막는다 (아키텍처 규칙 4)."""


async def save_profile(
    graph: AsyncDriver,
    *,
    company_id: UUID,
    recruitment_id: UUID,
    application_id: UUID,
    agent: ItemAgent,
    items: Sequence[NewItem],
    fields: ProfileFields | None = None,
    stage_id: UUID | None = None,
    documents: Sequence[ApplicantDocument] = (),
) -> CandidateProfile:
    """분석 결과를 지원자 프로필에 저장하고 저장된 프로필을 돌려준다.

    agent: 결과를 낸 분석 Agent. 이 Agent가 stage_id 전형에서 쓴 이전 항목만 바뀐다.
    fields: 지원서 분석이면 넘긴다. 없으면 요약 값은 그대로 둔다.
    stage_id: 평가 · 면접 분석이면 그 전형 ID. 지원서 분석은 None.
    documents: 근거 문장을 확인할 제출 문서. 비우면 확인하지 않는다 (평가 결과 · 면접 답변처럼 문서가 아닌 자료).
    """
    checked = [item for item in (_checked(item, documents) for item in items) if item is not None]
    if len(checked) < len(items):
        logger.info("근거 문장을 원문에서 찾지 못한 항목 %d개를 버렸다 (%s)", len(items) - len(checked), application_id)

    now = datetime.now(UTC)
    rows = [
        _claim_row(
            application_id,
            company_id,
            seq,
            id=uuid4(),
            category=item.category,
            agent=agent,
            content=item.content,
            evidence=item.evidence,
            verification=_verification(item),
            source=item.source,
            stage_id=stage_id,
            created_at=now,
        )
        for seq, item in enumerate(checked)
    ]

    async def write(tx: AsyncManagedTransaction) -> None:
        # 1) 지원서 분석 전에 다른 결과가 먼저 오면 요약 값 없이 '확인 필요'로 만들어 둔다
        await _merge_profile(tx, application_id, company_id, recruitment_id, now)
        if fields is not None:
            await tx.run(
                _SET_FIELDS,
                applicationId=str(application_id),
                status=fields.status.value,
                careerYears=fields.career_years,
                lastCompany=fields.last_company,
                summary=fields.summary,
            )
        # 2) 같은 Agent · 같은 전형의 이전 항목은 지우고 새 항목을 더한다
        await tx.run(
            _DELETE_CLAIMS,
            applicationId=str(application_id),
            agent=agent.value,
            stageId=str(stage_id) if stage_id is not None else None,
        )
        if rows:
            await tx.run(_CREATE_CLAIMS, applicationId=str(application_id), items=rows)
        await tx.run(_DELETE_UNUSED_SOURCES, applicationId=str(application_id))

    await ensure_schema(graph)
    async with graph.session() as session:
        await session.execute_write(write)
    saved = await load_profiles(
        graph, company_id=company_id, recruitment_id=recruitment_id, application_ids=[application_id]
    )
    return saved[0]


async def import_profile(graph: AsyncDriver, profile: CandidateProfile) -> int:
    """다른 저장소에서 읽은 프로필을 옮겨 넣고 옮긴 항목 수를 돌려준다 (db.profile_legacy).

    항목 ID와 시각을 지킨다. 그래프에 프로필이 없으면 요약 값까지 그대로 만든다.
    이미 있으면 그래프가 더 새 것이므로 요약 값은 두고, 그래프에 항목이 없는 Agent · 전형의 항목만 채운다.
    그래서 다시 옮겨도 결과가 같다. 기업 · 채용이 다른 프로필이 있으면 ProfileScopeError다.
    """
    application_id = str(profile.application_id)

    async def write(tx: AsyncManagedTransaction) -> int:
        owner = await (await tx.run(_LOCK_PROFILE, applicationId=application_id)).single()
        existed = owner is not None
        if owner is not None:
            if (owner["companyId"], owner["recruitmentId"]) != (str(profile.company_id), str(profile.recruitment_id)):
                raise ProfileScopeError(f"지원서 {profile.application_id}의 프로필은 다른 기업 · 채용에 속한다")
        else:
            await _merge_profile(
                tx, profile.application_id, profile.company_id, profile.recruitment_id, profile.created_at
            )
            await tx.run(
                _SET_FIELDS,
                applicationId=application_id,
                status=profile.status,
                careerYears=profile.career_years,
                lastCompany=profile.last_company,
                summary=profile.summary,
            )
        found = await tx.run(_CLAIM_GROUPS, applicationId=application_id)
        groups = {(record["agent"], record["stageId"]) async for record in found}
        rows = [
            _claim_row(
                profile.application_id,
                profile.company_id,
                seq,
                id=item.id,
                category=item.category,
                agent=item.agent,
                content=item.content,
                evidence=item.evidence,
                verification=item.verification,
                source=item.source,
                stage_id=item.stage_id,
                created_at=item.created_at,
            )
            for seq, item in enumerate(profile.items)
            if (str(item.agent), str(item.stage_id) if item.stage_id is not None else None) not in groups
        ]
        if rows:
            await tx.run(_CREATE_CLAIMS, applicationId=application_id, items=rows)
        if not existed:
            # _merge_profile이 적은 수정 시각을 원래 값으로 되돌린다
            await tx.run(
                _SET_TIMES, applicationId=application_id, createdAt=profile.created_at, updatedAt=profile.updated_at
            )
        return len(rows)

    await ensure_schema(graph)
    async with graph.session() as session:
        return await session.execute_write(write)


async def load_profiles(
    graph: AsyncDriver, *, company_id: UUID, recruitment_id: UUID, application_ids: Sequence[UUID]
) -> list[CandidateProfile]:
    """기업 · 채용이 맞는 프로필을 항목과 함께 읽는다. 받은 지원서 순서로 돌려주고 프로필이 없는 지원서는 뺀다."""
    order = list(dict.fromkeys(application_ids))
    if not order:
        return []
    records, _, _ = await graph.execute_query(
        _LOAD,
        applicationIds=[str(application_id) for application_id in order],
        companyId=str(company_id),
        recruitmentId=str(recruitment_id),
    )
    found = {profile.application_id: profile for profile in (_profile(record) for record in records)}
    return [found[application_id] for application_id in order if application_id in found]


async def load_profile(
    graph: AsyncDriver, *, company_id: UUID, recruitment_id: UUID, application_id: UUID
) -> CandidateProfile | None:
    """프로필 하나. 없거나 기업 · 채용이 다르면 None."""
    found = await load_profiles(
        graph, company_id=company_id, recruitment_id=recruitment_id, application_ids=[application_id]
    )
    return found[0] if found else None


async def profile_owner(graph: AsyncDriver, application_id: UUID) -> tuple[UUID, UUID] | None:
    """프로필이 속한 (기업, 채용). 프로필이 없으면 None."""
    records, _, _ = await graph.execute_query(_OWNER, applicationId=str(application_id))
    if not records:
        return None
    return UUID(records[0]["companyId"]), UUID(records[0]["recruitmentId"])


async def _merge_profile(
    tx: AsyncManagedTransaction, application_id: UUID, company_id: UUID, recruitment_id: UUID, now: datetime
) -> None:
    result = await tx.run(
        _MERGE_PROFILE,
        applicationId=str(application_id),
        companyId=str(company_id),
        recruitmentId=str(recruitment_id),
        status=ProfileStatus.CHECK.value,
        now=now,
    )
    owner = await result.single(strict=True)
    if (owner["companyId"], owner["recruitmentId"]) != (str(company_id), str(recruitment_id)):
        # 예외가 나면 트랜잭션이 되돌려져 남의 프로필은 그대로다
        raise ProfileScopeError(f"지원서 {application_id}의 프로필은 다른 기업 · 채용에 속한다")


def _claim_row(
    application_id: UUID,
    company_id: UUID,
    seq: int,
    *,
    id: UUID,
    category: str,
    agent: str,
    content: str,
    evidence: str | None,
    verification: str,
    source: str,
    stage_id: UUID | None,
    created_at: datetime,
) -> dict[str, Any]:
    """항목 한 줄을 Cypher 매개변수로 바꾼다.

    자료 노드는 지원서, 전형, 출처마다 하나고 기술과 도메인 노드는 기업마다 하나다.
    """
    stage = str(stage_id) if stage_id is not None else None
    return {
        "id": str(id),
        "category": str(category),
        "agent": str(agent),
        "content": content,
        "evidence": evidence,
        "evidenceId": str(uuid4()),
        "verification": str(verification),
        "source": source,
        "sourceKey": f"{application_id}|{stage or ''}|{source}",
        "conceptKey": f"{company_id}|{' '.join(content.casefold().split())}",
        "stageId": stage,
        "seq": seq,
        "createdAt": created_at,
    }


def _profile(record: Mapping[str, Any]) -> CandidateProfile:
    node = record["profile"]
    return CandidateProfile(
        application_id=UUID(node["applicationId"]),
        company_id=UUID(node["companyId"]),
        recruitment_id=UUID(node["recruitmentId"]),
        status=node["status"],
        career_years=node.get("careerYears"),
        last_company=node.get("lastCompany"),
        summary=node.get("summary") or "",
        created_at=node["createdAt"].to_native(),
        updated_at=node["updatedAt"].to_native(),
        items=[
            ProfileItem(
                id=UUID(item["id"]),
                category=item["category"],
                agent=item["agent"],
                content=item["content"],
                evidence=item["evidence"],
                verification=item["verification"],
                source=item["source"],
                stage_id=UUID(item["stageId"]) if item["stageId"] else None,
                created_at=item["createdAt"].to_native(),
            )
            for item in record["items"]
        ],
    )


def _checked(item: NewItem, documents: Sequence[ApplicantDocument]) -> NewItem | None:
    """근거 문장이 원문에 있으면 실제 문서 이름으로 출처를 맞춰 돌려주고, 없으면 None."""
    if item.evidence is None or not documents:
        return item
    found = evidence_source(item.evidence, documents)
    if found is None:
        return None
    return item.model_copy(update={"source": found.label})


def _verification(item: NewItem) -> Verification:
    """항목의 확인 상태를 정한다."""
    # 1) 근거 문장이 없으면 무엇을 넘겼든 미확인 (근거 없이 '확인됨'을 적지 못하게)
    if item.evidence is None:
        return Verification.UNCONFIRMED
    # 2) 분석 Agent가 넘긴 값이 있으면 그대로, 없으면 근거 있음
    return item.verification or Verification.EVIDENCED
