"""PostgreSQL 프로필 테이블을 그래프로 옮기는 함수 테스트. Docker로 띄운 PostgreSQL과 Neo4j를 쓴다."""

from uuid import uuid4

from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from hiremate_ai.db.profile import ItemAgent, ItemCategory, ProfileStatus, Verification
from hiremate_ai.db.profile_legacy import CopyResult, LegacyCandidateProfile, LegacyProfileItem, copy_legacy_profiles
from hiremate_ai.db.profile_store import NewItem, load_profile, save_profile


def legacy_profile() -> LegacyCandidateProfile:
    """가상 지원자 정유민의 서류, 면접 분석이 테이블에 남아 있다고 가정한 프로필."""
    return LegacyCandidateProfile(
        application_id=uuid4(),
        company_id=uuid4(),
        recruitment_id=uuid4(),
        status=ProfileStatus.DONE,
        career_years=4,
        last_company="한빛카드",
        summary="카드 결제 승인 · 정산 4년",
        items=[
            LegacyProfileItem(
                category=ItemCategory.DOMAIN,
                agent=ItemAgent.APPLICATION,
                content="금융 (카드 결제 · 정산)",
                evidence="한빛카드에서 카드 결제 승인 시스템을 개발했습니다.",
                verification=Verification.EVIDENCED,
                source="이력서",
            ),
            LegacyProfileItem(
                category=ItemCategory.STRENGTH,
                agent=ItemAgent.INTERVIEW,
                content="장애 대응 당번으로 승인 서버를 직접 운영",
                evidence="장애 당번으로 승인 서버를 직접 운영했습니다.",
                verification=Verification.CONFIRMED,
                source="1차 면접 답변",
                stage_id=uuid4(),
            ),
        ],
    )


async def test_copies_profiles_as_saved_and_again_without_change(db_session: AsyncSession, graph: AsyncDriver) -> None:
    legacy = legacy_profile()
    db_session.add(legacy)
    await db_session.flush()
    await db_session.refresh(legacy)

    first = await copy_legacy_profiles(db_session, graph)
    again = await copy_legacy_profiles(db_session, graph)

    assert first == CopyResult(profiles=1, items=2, skipped_items=0)
    assert again == CopyResult(profiles=1, items=0, skipped_items=2)
    profile = await load_profile(
        graph,
        company_id=legacy.company_id,
        recruitment_id=legacy.recruitment_id,
        application_id=legacy.application_id,
    )
    assert profile is not None
    assert (profile.status, profile.career_years, profile.last_company, profile.summary) == (
        ProfileStatus.DONE,
        4,
        "한빛카드",
        "카드 결제 승인 · 정산 4년",
    )
    assert (profile.created_at, profile.updated_at) == (legacy.created_at, legacy.updated_at)
    saved = {item.id: item for item in profile.items}
    assert set(saved) == {item.id for item in legacy.items}
    for item in legacy.items:
        copied = saved[item.id]
        assert (copied.category, copied.agent, copied.content, copied.evidence) == (
            item.category,
            item.agent,
            item.content,
            item.evidence,
        )
        assert (copied.verification, copied.source, copied.stage_id, copied.created_at) == (
            item.verification,
            item.source,
            item.stage_id,
            item.created_at,
        )


async def test_does_not_overwrite_analysis_saved_to_graph_after_cutover(
    db_session: AsyncSession, graph: AsyncDriver
) -> None:
    legacy = legacy_profile()
    db_session.add(legacy)
    await db_session.flush()
    await db_session.refresh(legacy)
    # 전환 뒤 지원서를 다시 분석한 결과가 그래프에 먼저 들어와 있다
    await save_profile(
        graph,
        company_id=legacy.company_id,
        recruitment_id=legacy.recruitment_id,
        application_id=legacy.application_id,
        agent=ItemAgent.APPLICATION,
        items=[NewItem(category=ItemCategory.SKILL, content="Kafka", source="이력서")],
    )

    result = await copy_legacy_profiles(db_session, graph)

    assert result == CopyResult(profiles=1, items=1, skipped_items=1)
    profile = await load_profile(
        graph,
        company_id=legacy.company_id,
        recruitment_id=legacy.recruitment_id,
        application_id=legacy.application_id,
    )
    assert profile is not None
    # 지원서 항목은 그래프의 새 결과 그대로고, 그래프에 없던 면접 항목만 채워진다
    assert sorted((item.agent, item.content) for item in profile.items) == [
        (ItemAgent.APPLICATION, "Kafka"),
        (ItemAgent.INTERVIEW, "장애 대응 당번으로 승인 서버를 직접 운영"),
    ]
    assert profile.status == ProfileStatus.CHECK
