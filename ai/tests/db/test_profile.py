"""지원자 프로필 그래프의 제약 테스트. Docker로 띄운 Neo4j에 실제로 넣어 본다."""

from uuid import uuid4

import pytest
from neo4j import AsyncDriver
from neo4j.exceptions import ConstraintError

from hiremate_ai.db.graph import ensure_schema


async def test_one_profile_per_application(graph: AsyncDriver) -> None:
    await ensure_schema(graph)
    application_id = str(uuid4())
    await graph.execute_query("CREATE (:ApplicationProfile {applicationId: $id})", id=application_id)

    with pytest.raises(ConstraintError):
        await graph.execute_query("CREATE (:ApplicationProfile {applicationId: $id})", id=application_id)


async def test_schema_is_created_once_and_can_run_again(graph: AsyncDriver) -> None:
    await ensure_schema(graph)
    await ensure_schema(graph)

    records, _, _ = await graph.execute_query("SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names")
    assert set(records[0]["names"]) == {
        "application_profile_application_id",
        "claim_id",
        "evidence_id",
        "source_key",
        "experience_key",
        "skill_key",
        "domain_key",
    }
