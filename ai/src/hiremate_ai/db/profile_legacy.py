"""그래프로 옮기기 전의 지원자 프로필 테이블 (ai.candidate_profiles, ai.profile_items).

지원자 프로필은 그래프(db.profile, db.profile_store)에 저장한다. 이 테이블은 이미 쌓인 프로필을
그래프에 옮길 때만 읽는다. 앱은 여기에 쓰지 않는다. 옮긴 결과를 확인한 뒤 테이블과 이 모듈을 함께 지운다.

옮기기 (ai 폴더에서, 운영은 ai 컨테이너에서): uv run python -m hiremate_ai.db.profile_legacy
DB 주소는 DATABASE_URL, 그래프는 GRAPH_URI, GRAPH_USER, GRAPH_PASSWORD를 쓴다.
테이블은 바꾸지 않는다. 그래프에 이미 있는 프로필은 그래프에 없는 Agent, 전형의 항목만 채우므로
다시 실행해도 결과가 같고, 옮긴 뒤 새로 분석한 결과를 덮어쓰지 않는다.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from neo4j import AsyncDriver
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship

from hiremate_ai.db.base import SCHEMA, Base
from hiremate_ai.db.graph import graph_driver
from hiremate_ai.db.profile import CandidateProfile, ProfileItem
from hiremate_ai.db.profile_store import import_profile
from hiremate_ai.db.session import session_factory


class LegacyCandidateProfile(Base):
    """ai.candidate_profiles 한 줄. 지원서(application_id)마다 하나다."""

    __tablename__ = "candidate_profiles"
    __table_args__ = (Index("ix_candidate_profiles_scope", "company_id", "recruitment_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # backend의 지원서 · 기업 · 채용 ID. 다른 스키마라 외래 키는 걸지 않는다
    application_id: Mapped[UUID] = mapped_column(unique=True)
    company_id: Mapped[UUID]
    recruitment_id: Mapped[UUID]
    status: Mapped[str] = mapped_column(String(10))  # ProfileStatus 값
    # 조건 검색에 쓰는 값. 분석 전이거나 찾지 못하면 None
    career_years: Mapped[int | None]
    last_company: Mapped[str | None] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list["LegacyProfileItem"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan", lazy="selectin", order_by="LegacyProfileItem.created_at"
    )


class LegacyProfileItem(Base):
    """ai.profile_items 한 줄."""

    __tablename__ = "profile_items"
    __table_args__ = (Index("ix_profile_items_profile_id", "profile_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    profile_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.candidate_profiles.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(20))  # ItemCategory 값
    agent: Mapped[str] = mapped_column(String(20))  # ItemAgent 값
    content: Mapped[str] = mapped_column(Text)  # 정리한 내용 (예: "금융 (카드 결제 · 정산)")
    evidence: Mapped[str | None] = mapped_column(Text)  # 원문에서 찾은 근거 문장. 없으면 None
    verification: Mapped[str] = mapped_column(String(12))  # Verification 값 (확인 상태)
    source: Mapped[str] = mapped_column(String(50))  # 출처 (예: 이력서, 자기소개서, 평가 결과, 면접 답변)
    stage_id: Mapped[UUID | None]  # 평가 · 면접 항목이면 그 전형 ID. 지원서 항목은 None
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    profile: Mapped[LegacyCandidateProfile] = relationship(back_populates="items")


@dataclass(frozen=True)
class CopyResult:
    """옮긴 결과. skipped_items는 그래프에 같은 Agent, 전형의 항목이 이미 있어 옮기지 않은 항목 수다."""

    profiles: int
    items: int
    skipped_items: int


async def copy_legacy_profiles(session: AsyncSession, graph: AsyncDriver) -> CopyResult:
    """테이블의 프로필을 모두 그래프로 옮긴다. 테이블은 바꾸지 않는다. 다시 실행해도 결과가 같다."""
    profiles = (await session.scalars(select(LegacyCandidateProfile).order_by(LegacyCandidateProfile.created_at))).all()
    copied = 0
    total = 0
    for legacy in profiles:
        # 테이블의 순서(저장 시각, 같으면 id)를 그래프의 저장 순서로 삼는다
        items = sorted(legacy.items, key=lambda item: (item.created_at, item.id))
        total += len(items)
        copied += await import_profile(
            graph,
            CandidateProfile(
                application_id=legacy.application_id,
                company_id=legacy.company_id,
                recruitment_id=legacy.recruitment_id,
                status=legacy.status,
                career_years=legacy.career_years,
                last_company=legacy.last_company,
                summary=legacy.summary,
                created_at=legacy.created_at,
                updated_at=legacy.updated_at,
                items=[
                    ProfileItem(
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
                    for item in items
                ],
            ),
        )
    return CopyResult(profiles=len(profiles), items=copied, skipped_items=total - copied)


async def _copy() -> CopyResult:
    graph = graph_driver()
    try:
        async with session_factory()() as session:
            return await copy_legacy_profiles(session, graph)
    finally:
        await graph.close()


def main() -> None:
    result = asyncio.run(_copy())
    print(f"프로필 {result.profiles}개, 옮긴 항목 {result.items}개")
    print(f"그래프에 이미 있어 건너뛴 항목 {result.skipped_items}개")


if __name__ == "__main__":
    main()
