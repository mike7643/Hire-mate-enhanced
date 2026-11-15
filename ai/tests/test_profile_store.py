"""프로필 저장 함수 테스트. 지원서 분석 Agent가 낸 것처럼 만든 더미 결과(tests/samples/profiles.json)를 저장해 본다.

Docker로 띄운 Neo4j에 실제로 넣고 읽어 본다.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from neo4j import AsyncDriver
from pydantic import BaseModel, TypeAdapter

from hiremate_ai.agents.guardrails import blocked_attribute
from hiremate_ai.contract import ApplicantDetail, CompanyProfile
from hiremate_ai.db.profile import CandidateProfile, ItemAgent, ItemCategory, ProfileItem, ProfileStatus, Verification
from hiremate_ai.db.profile_store import (
    NewItem,
    ProfileFields,
    ProfileScopeError,
    import_profile,
    load_profile,
    load_profiles,
    profile_owner,
    save_profile,
)

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


class SampleProfile(BaseModel):
    """더미 분석 결과 한 명분. 2번 지원서 분석 Agent가 완성되면 실제 결과로 바뀐다."""

    name: str
    fields: ProfileFields
    items: list[NewItem]


SAMPLE_PROFILES = {
    sample.name: sample
    for sample in TypeAdapter(list[SampleProfile]).validate_json((SAMPLES_DIR / "profiles.json").read_bytes())
}


async def save_sample(
    graph: AsyncDriver, company: CompanyProfile, applicant: ApplicantDetail, items: list[NewItem] | None = None
) -> CandidateProfile:
    """가상 지원자 한 명의 더미 분석 결과를 지원서 분석 Agent처럼 저장한다."""
    sample = SAMPLE_PROFILES[applicant.info.name]
    return await save_profile(
        graph,
        company_id=company.company_id,
        recruitment_id=applicant.info.recruitment_id,
        application_id=applicant.info.application_id,
        agent=ItemAgent.APPLICATION,
        fields=sample.fields,
        items=sample.items if items is None else items,
        documents=applicant.documents,
    )


async def reload(graph: AsyncDriver, application_id: UUID) -> CandidateProfile:
    """그래프에서 다시 읽는다."""
    owner = await profile_owner(graph, application_id)
    assert owner is not None
    profile = await load_profile(graph, company_id=owner[0], recruitment_id=owner[1], application_id=application_id)
    assert profile is not None
    return profile


async def count(graph: AsyncDriver, pattern: str) -> int:
    """패턴에 맞는 그래프 조각 수."""
    records, _, _ = await graph.execute_query(f"MATCH {pattern} RETURN count(*) AS total")
    return int(records[0]["total"])


def test_dummy_profiles_cover_every_sample_applicant(sample_applicants: dict[str, ApplicantDetail]) -> None:
    assert set(SAMPLE_PROFILES) == set(sample_applicants)
    # 더미 프로필에도 차별 조건 낱말을 넣지 않는다 (규칙 6)
    assert [
        item.content for sample in SAMPLE_PROFILES.values() for item in sample.items if blocked_attribute(item.content)
    ] == []


async def test_saves_application_analysis(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    await save_sample(graph, sample_company, applicant)

    profile = await reload(graph, applicant.info.application_id)

    assert profile.company_id == sample_company.company_id
    assert profile.status == ProfileStatus.DONE
    assert (profile.career_years, profile.last_company) == (4, "한빛카드")
    assert len(profile.items) == 5
    assert {item.agent for item in profile.items} == {ItemAgent.APPLICATION}
    domain = next(item for item in profile.items if item.category == ItemCategory.DOMAIN)
    assert (domain.content, domain.source, domain.verification) == (
        "금융 (카드 결제 · 정산)",
        "이력서",
        Verification.EVIDENCED,
    )


async def test_saves_all_sample_applicants_as_given(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    for applicant in sample_applicants.values():
        await save_sample(graph, sample_company, applicant)

    applicant = next(iter(sample_applicants.values()))
    saved = {
        profile.application_id: profile
        for profile in await load_profiles(
            graph,
            company_id=sample_company.company_id,
            recruitment_id=applicant.info.recruitment_id,
            application_ids=[applicant.info.application_id for applicant in sample_applicants.values()],
        )
    }

    for applicant in sample_applicants.values():
        sample = SAMPLE_PROFILES[applicant.info.name]
        profile = saved[applicant.info.application_id]
        # 받은 값을 그대로 저장한다. 근거가 모두 원문에 있으므로 버려지는 항목도 없다
        assert profile.career_years == sample.fields.career_years
        assert len(profile.items) == len(sample.items)
    # 문서 본문이 빈 윤태오는 확인 필요로 남는다 (정답표 7번)
    assert saved[sample_applicants["윤태오"].info.application_id].status == ProfileStatus.CHECK


async def test_drops_item_whose_evidence_is_not_in_documents(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    items = [
        NewItem(
            category=ItemCategory.EXPERIENCE,
            content="팀장 경험",
            evidence="10명 규모 팀을 이끌었습니다.",
            source="이력서",
        ),
        NewItem(category=ItemCategory.CHECK, content="대용량 근거 확인 필요", evidence=None, source="이력서"),
    ]

    profile = await save_sample(graph, sample_company, applicant, items)

    # 원문에 없는 근거를 단 항목은 버리고, 근거가 없는 확인 필요 항목은 남긴다
    assert [item.content for item in profile.items] == ["대용량 근거 확인 필요"]


async def test_sets_verification_from_evidence_unless_agent_gives_it(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    sample = SAMPLE_PROFILES["정유민"]
    no_evidence = NewItem(category=ItemCategory.CHECK, content="대용량 근거 확인 필요", source="이력서")
    # 1) 지원서 분석은 확인 상태를 넘기지 않는다 → 근거 문장이 있으면 근거 있음, 없으면 미확인
    await save_sample(graph, sample_company, applicant, [sample.items[0], no_evidence])
    # 2) 면접 분석은 확인됨 · 상충을 넘긴다. 근거 문장이 없으면 무엇을 넘겼든 미확인
    await save_profile(
        graph,
        company_id=sample_company.company_id,
        recruitment_id=applicant.info.recruitment_id,
        application_id=applicant.info.application_id,
        agent=ItemAgent.INTERVIEW,
        stage_id=uuid4(),
        items=[
            NewItem(
                category=ItemCategory.STRENGTH,
                content="장애 대응 당번 경험",
                evidence="장애 당번으로 승인 서버를 직접 운영했습니다.",
                source="1차 면접 답변",
                verification=Verification.CONFIRMED,
            ),
            NewItem(
                category=ItemCategory.CHECK,
                content="정산 배치 개선에서 맡은 역할이 서류와 다름",
                evidence="설계는 팀장님이 하셨고 저는 테스트를 맡았습니다.",
                source="1차 면접 답변",
                verification=Verification.CONFLICT,
            ),
            NewItem(
                category=ItemCategory.STRENGTH,
                content="근거 없이 확인됨을 적은 항목",
                source="1차 면접 답변",
                verification=Verification.CONFIRMED,
            ),
        ],
    )

    profile = await reload(graph, applicant.info.application_id)

    states = {item.content: item.verification for item in profile.items}
    assert states == {
        sample.items[0].content: Verification.EVIDENCED,
        "대용량 근거 확인 필요": Verification.UNCONFIRMED,
        "장애 대응 당번 경험": Verification.CONFIRMED,
        "정산 배치 개선에서 맡은 역할이 서류와 다름": Verification.CONFLICT,
        "근거 없이 확인됨을 적은 항목": Verification.UNCONFIRMED,
    }


async def test_corrects_source_to_document_that_has_the_sentence(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    wrong_source = NewItem(
        category=ItemCategory.COMPANY_FIT,
        content="장애 회고 문서화",
        evidence="장애 회고를 문서로 남겨 같은 문제가 반복되지 않게 했습니다.",
        source="이력서",  # 실제로는 자기소개서에 있는 문장
    )

    profile = await save_sample(graph, sample_company, applicant, [wrong_source])

    assert profile.items[0].source == "자기소개서"


async def test_reanalysis_replaces_only_items_of_same_agent_and_stage(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    await save_sample(graph, sample_company, applicant)
    assessment_stage = uuid4()
    await save_profile(
        graph,
        company_id=sample_company.company_id,
        recruitment_id=applicant.info.recruitment_id,
        application_id=applicant.info.application_id,
        agent=ItemAgent.ASSESSMENT,
        stage_id=assessment_stage,
        items=[NewItem(category=ItemCategory.STRENGTH, content="과제 테스트 코드 충실", source="평가 결과")],
    )

    # 지원서를 다시 분석해 항목이 하나로 줄었다
    sample = SAMPLE_PROFILES["정유민"]
    await save_sample(graph, sample_company, applicant, sample.items[:1])
    profile = await reload(graph, applicant.info.application_id)

    application_items = [item for item in profile.items if item.agent == ItemAgent.APPLICATION]
    assessment_items = [item for item in profile.items if item.agent == ItemAgent.ASSESSMENT]
    assert [item.content for item in application_items] == [sample.items[0].content]
    assert [(item.content, item.stage_id) for item in assessment_items] == [("과제 테스트 코드 충실", assessment_stage)]


async def test_result_before_application_analysis_creates_check_profile(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["김하늘"]

    profile = await save_profile(
        graph,
        company_id=sample_company.company_id,
        recruitment_id=applicant.info.recruitment_id,
        application_id=applicant.info.application_id,
        agent=ItemAgent.INTERVIEW,
        stage_id=uuid4(),
        items=[NewItem(category=ItemCategory.STRENGTH, content="장애 대응 경험을 구체적으로 설명", source="면접 답변")],
    )

    assert profile.status == ProfileStatus.CHECK
    assert len(profile.items) == 1


async def test_rejects_profile_of_other_company(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    await save_sample(graph, sample_company, applicant)

    with pytest.raises(ProfileScopeError):
        await save_profile(
            graph,
            company_id=uuid4(),  # 다른 기업
            recruitment_id=applicant.info.recruitment_id,
            application_id=applicant.info.application_id,
            agent=ItemAgent.APPLICATION,
            items=[],
        )


async def test_rejected_save_leaves_profile_unchanged(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    before = await save_sample(graph, sample_company, applicant)

    with pytest.raises(ProfileScopeError):
        await save_profile(
            graph,
            company_id=uuid4(),  # 다른 기업
            recruitment_id=applicant.info.recruitment_id,
            application_id=applicant.info.application_id,
            agent=ItemAgent.APPLICATION,
            items=[],
        )

    assert await reload(graph, applicant.info.application_id) == before


async def test_keeps_saved_order(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    sample = SAMPLE_PROFILES["정유민"]

    profile = await save_sample(graph, sample_company, applicant)

    assert [item.content for item in profile.items] == [item.content for item in sample.items]


async def test_stores_claims_with_evidence_and_source(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    sample = SAMPLE_PROFILES["정유민"]
    await save_sample(graph, sample_company, applicant)

    with_evidence = [item for item in sample.items if item.evidence is not None]
    assert await count(graph, "(:ApplicationProfile)-[:HAS_CLAIM]->(:Claim)") == len(sample.items)
    # 근거 문장이 있는 항목은 근거 문장을 거쳐, 없는 항목은 바로 자료에 이어진다
    assert await count(graph, "(:Claim)-[:SUPPORTED_BY]->(:Evidence)-[:FROM_SOURCE]->(:Source)") == len(with_evidence)
    assert await count(graph, "(:Claim)-[:FROM_SOURCE]->(:Source)") == len(sample.items) - len(with_evidence)
    # 분석 Agent가 기술, 업무 분야, 경험 이름을 넘기지 않은 항목은 내용 글로 노드를 만들지 않는다
    assert await count(graph, "(n) WHERE n:Skill OR n:Domain OR n:Experience") == 0


def card_items() -> list[NewItem]:
    """가상 지원자가 카드 결제 프로젝트와 사내 도구 프로젝트에서 한 일 (분석 Agent가 낸 것처럼 만든 항목)."""
    return [
        NewItem(
            category=ItemCategory.EXPERIENCE,
            content="카드 결제 승인 이벤트 처리 모듈을 Kafka로 개발했다.",
            source="이력서",
            experience="카드 결제 시스템",
            actor="SELF",
            action="DEVELOP",
            skills=("Kafka", "Java"),
            domains=("금융",),
        ),
        NewItem(
            category=ItemCategory.EXPERIENCE,
            content="팀이 승인 서버를 운영했다.",
            source="이력서",
            experience="카드 결제  시스템",  # 공백만 다른 같은 이름
            actor="TEAM",
            action="OPERATE",
            domains=("금융",),
        ),
        NewItem(
            category=ItemCategory.EXPERIENCE,
            content="사내 배포 도구를 Python으로 만들었다.",
            source="포트폴리오",
            experience="사내 배포 도구",
            actor="SELF",
            action="DEVELOP",
            skills=("Python",),
        ),
        NewItem(category=ItemCategory.SKILL, content="Redis를 다룰 수 있다.", source="이력서", skills=("Redis",)),
    ]


async def test_links_claims_of_one_experience_with_actor_skills_and_domains(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]

    profile = await save_sample(graph, sample_company, applicant, card_items())

    by_content = {item.content: item for item in profile.items}
    kafka = by_content["카드 결제 승인 이벤트 처리 모듈을 Kafka로 개발했다."]
    assert (kafka.experience, kafka.actor, kafka.action) == ("카드 결제 시스템", "SELF", "DEVELOP")
    assert (kafka.skills, kafka.domains) == (("Java", "Kafka"), ("금융",))
    # 경험이 없는 기술 항목은 경험 노드 없이 기술만 가리킨다
    redis = by_content["Redis를 다룰 수 있다."]
    assert (redis.experience, redis.actor, redis.skills) == (None, None, ("Redis",))
    # 공백만 다른 이름은 같은 경험이다. 경험은 그 지원서에 달린다
    assert await count(graph, "(:ApplicationProfile)-[:HAS_EXPERIENCE]->(:Experience)") == 2
    assert await count(graph, "(:Claim)-[:ABOUT_EXPERIENCE]->(:Experience {title: '카드 결제 시스템'})") == 2


async def test_graph_tells_same_experience_from_separate_ones(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    """관계 검색이 쓸 모양: 같은 경험 안에서 금융 분야와 본인이 쓴 기술이 이어지는지 따라갈 수 있다."""
    await save_sample(graph, sample_company, sample_applicants["정유민"], card_items())
    same_experience = (
        "(:Domain {name: '금융'})<-[:IN_DOMAIN]-(:Claim)-[:ABOUT_EXPERIENCE]->(:Experience)"
        "<-[:ABOUT_EXPERIENCE]-(:Claim {actor: 'SELF'})-[:USES_SKILL]->(:Skill {name: '%s'})"
    )

    # Kafka는 금융 경험(카드 결제 시스템)에서 본인이 썼고, Python은 다른 경험(사내 배포 도구)에서 썼다
    assert await count(graph, same_experience % "Kafka") > 0
    assert await count(graph, same_experience % "Python") == 0


async def test_same_skill_is_one_node_within_company_only(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    kafka = [NewItem(category=ItemCategory.SKILL, content="Kafka를 쓴다.", source="이력서", skills=("Kafka",))]
    spaced = [NewItem(category=ItemCategory.SKILL, content="kafka 경험", source="이력서", skills=(" kafka ",))]
    korean = [NewItem(category=ItemCategory.SKILL, content="카프카 경험", source="자기소개서", skills=("카프카",))]
    jung, kim = sample_applicants["정유민"], sample_applicants["김하늘"]
    await save_sample(graph, sample_company, jung, kafka)
    await save_sample(graph, sample_company, kim, spaced + korean)
    # 다른 기업의 지원자
    await save_profile(
        graph,
        company_id=uuid4(),
        recruitment_id=uuid4(),
        application_id=uuid4(),
        agent=ItemAgent.APPLICATION,
        items=kafka,
    )

    company = f"{{companyId: '{sample_company.company_id}'}}"
    # 대소문자와 공백만 다른 이름은 한 노드고, 표기가 다른 이름(카프카)은 합치지 않는다. 기업끼리는 나눈다
    assert await count(graph, f"(:Skill {company})") == 2
    assert (
        await count(
            graph, f"(:Claim)-[:USES_SKILL]->(:Skill {{name: 'Kafka', companyId: '{sample_company.company_id}'}})"
        )
        == 2
    )
    assert await count(graph, "(:Skill {name: 'Kafka'})") == 2


async def test_reanalysis_removes_unused_experiences(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    await save_sample(graph, sample_company, applicant, card_items())

    await save_sample(graph, sample_company, applicant, card_items()[2:])

    assert await count(graph, "(:Experience)") == 1
    assert await count(graph, "(:Experience {title: '사내 배포 도구'})") == 1


async def test_reanalysis_removes_old_evidence_and_unused_sources(
    graph: AsyncDriver, sample_company: CompanyProfile, sample_applicants: dict[str, ApplicantDetail]
) -> None:
    applicant = sample_applicants["정유민"]
    await save_sample(graph, sample_company, applicant)

    await save_sample(graph, sample_company, applicant, [])

    assert await count(graph, "(:Claim)") == 0
    assert await count(graph, "(:Evidence)") == 0
    assert await count(graph, "(:Source)") == 0


async def test_import_keeps_ids_and_times_and_is_repeatable(graph: AsyncDriver) -> None:
    created = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
    stage_id = uuid4()
    profile = CandidateProfile(
        application_id=uuid4(),
        company_id=uuid4(),
        recruitment_id=uuid4(),
        status=ProfileStatus.DONE,
        career_years=4,
        last_company="한빛카드",
        summary="서류 분석 완료",
        created_at=created,
        updated_at=created + timedelta(days=1),
        items=[
            ProfileItem(
                id=uuid4(),
                category=ItemCategory.DOMAIN,
                agent=ItemAgent.APPLICATION,
                content="금융 (카드 결제 · 정산)",
                evidence="한빛카드에서 카드 결제 승인 시스템을 개발했습니다.",
                verification=Verification.EVIDENCED,
                source="이력서",
                stage_id=None,
                created_at=created,
                experience="한빛카드",
                actor="SELF",
                action="DEVELOP",
                skills=("Java", "Kafka"),
                domains=("금융",),
            ),
            ProfileItem(
                id=uuid4(),
                category=ItemCategory.STRENGTH,
                agent=ItemAgent.INTERVIEW,
                content="장애 대응 당번 경험",
                evidence=None,
                verification=Verification.UNCONFIRMED,
                source="면접 답변",
                stage_id=stage_id,
                created_at=created + timedelta(hours=1),
            ),
        ],
    )

    assert await import_profile(graph, profile) == 2
    assert await import_profile(graph, profile) == 0

    assert await reload(graph, profile.application_id) == profile
    assert await count(graph, "(:Claim)") == 2
