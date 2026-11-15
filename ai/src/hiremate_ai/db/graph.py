"""지원자 프로필 그래프(Neo4j) 연결. 드라이버는 처음 쓸 때 만든다.

그래프에는 지원자 프로필(db.profile)만 둔다. 호출 로그, 대화방, 분석 원본, 문서 색인, NCS는
PostgreSQL ai 스키마에 남는다.
제약과 색인은 리비전 파일 없이 ensure_schema가 처음 쓸 때 만든다 (있으면 건너뛴다).
"""

from functools import cache
from typing import Annotated
from weakref import WeakSet

from fastapi import Depends
from neo4j import AsyncDriver, AsyncGraphDatabase

from hiremate_ai.config import settings

# 고유 제약은 같은 값의 노드가 둘 생기는 것을 막고 그 값으로 찾는 색인도 겸한다
_SCHEMA = (
    "CREATE CONSTRAINT application_profile_application_id IF NOT EXISTS "
    "FOR (p:ApplicationProfile) REQUIRE p.applicationId IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT source_key IF NOT EXISTS FOR (s:Source) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT experience_key IF NOT EXISTS FOR (x:Experience) REQUIRE x.key IS UNIQUE",
    "CREATE CONSTRAINT skill_key IF NOT EXISTS FOR (k:Skill) REQUIRE k.key IS UNIQUE",
    "CREATE CONSTRAINT domain_key IF NOT EXISTS FOR (d:Domain) REQUIRE d.key IS UNIQUE",
    "CREATE INDEX application_profile_scope IF NOT EXISTS FOR (p:ApplicationProfile) ON (p.companyId, p.recruitmentId)",
)
_prepared: WeakSet[AsyncDriver] = WeakSet()


@cache
def graph_driver(uri: str | None = None) -> AsyncDriver:
    """uri가 없으면 settings.graph_uri를 쓴다. 주소마다 드라이버를 하나만 만든다."""
    return AsyncGraphDatabase.driver(
        uri or settings.graph_uri,
        auth=(settings.graph_user, settings.graph_password.get_secret_value()),
        # 아직 한 번도 쓰지 않은 속성이나 레이블을 조회할 때 나오는 경고. 빈 그래프에서는 늘 나온다
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


async def ensure_schema(graph: AsyncDriver) -> None:
    """제약과 색인을 만든다. 드라이버마다 한 번만 실행한다."""
    if graph in _prepared:
        return
    for statement in _SCHEMA:
        await graph.execute_query(statement)
    _prepared.add(graph)


def get_graph() -> AsyncDriver:
    """FastAPI 의존성. 앱이 함께 쓰는 드라이버를 돌려준다."""
    return graph_driver()


GraphDep = Annotated[AsyncDriver, Depends(get_graph)]
"""라우터 매개변수 타입. 예: async def read(request: ReadRequest, graph: GraphDep) -> CandidateProfileView"""
