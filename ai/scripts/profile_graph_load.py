"""PostgreSQL에 쌓인 지원자 프로필(ai.candidate_profiles, ai.profile_items)을 프로필 그래프(Neo4j)로 옮긴다.

ai 폴더에서: uv run python scripts/profile_graph_load.py
DB 주소는 ai/.env의 DATABASE_URL, 그래프는 GRAPH_URI, GRAPH_USER, GRAPH_PASSWORD를 쓴다.

테이블은 바꾸지 않는다. 그래프에 이미 있는 프로필은 그래프에 없는 Agent, 전형의 항목만 채우므로
다시 실행해도 결과가 같고, 옮긴 뒤 새로 분석한 결과를 덮어쓰지 않는다.
"""

import asyncio

from hiremate_ai.db.graph import graph_driver
from hiremate_ai.db.profile_legacy import CopyResult, copy_legacy_profiles
from hiremate_ai.db.session import session_factory


async def load() -> CopyResult:
    graph = graph_driver()
    try:
        async with session_factory()() as session:
            return await copy_legacy_profiles(session, graph)
    finally:
        await graph.close()


def main() -> None:
    result = asyncio.run(load())
    print(f"프로필 {result.profiles}개, 옮긴 항목 {result.items}개")
    print(f"그래프에 이미 있어 건너뛴 항목 {result.skipped_items}개")


if __name__ == "__main__":
    main()
