import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from langfuse import Langfuse
from langsmith import tracing_context
from neo4j import AsyncDriver, AsyncGraphDatabase
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr, TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.community.neo4j import Neo4jContainer
from testcontainers.community.postgres import PostgresContainer

from hiremate_ai.config import settings
from hiremate_ai.contract import ApplicantDetail, CompanyProfile, PostingContent
from hiremate_ai.db import graph as graph_module
from hiremate_ai.db.graph import graph_driver
from hiremate_ai.llm import tracing
from hiremate_ai.llm.tracing import traced

AI_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = AI_DIR / "tests" / "samples"

# compose.yaml과 같은 이미지
POSTGRES_IMAGE = "pgvector/pgvector:pg17"
NEO4J_IMAGE = "neo4j:5.26-community"


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
    """async 테스트를 SelectorEventLoop에서 돌린다.

    Windows 기본 루프(ProactorEventLoop)에서는 psycopg가 async로 DB에 붙지 못한다.
    Linux · macOS는 원래 SelectorEventLoop라 동작이 같다.
    """
    return {"selector": asyncio.SelectorEventLoop}


@pytest.fixture(autouse=True)
def langfuse_tracing(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Langfuse 추적은 실제 모델 평가(tests/eval)만 LANGFUSE_TRACING_ENABLED를 따른다.

    다른 테스트는 ai/.env나 환경 변수로 추적을 켜 두어도 기록을 보내지 않는다.
    """
    if "eval" in request.path.relative_to(AI_DIR / "tests").parts and settings.llm_eval:
        with traced():
            yield
        return
    monkeypatch.setattr(settings, "langfuse_tracing_enabled", False)

    with tracing_context(enabled=False):
        yield


@pytest.fixture(autouse=True)
def isolate_poster_provider(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.path.parent.name != "eval":
        monkeypatch.setattr(settings, "poster_llm_model", "")
        monkeypatch.setattr(settings, "poster_api_key", SecretStr(""))
        monkeypatch.setattr(settings, "poster_local_python", "")
        monkeypatch.setattr(settings, "poster_local_release_ollama", False)


@pytest.fixture
def langfuse(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """실제 SDK가 만든 span을 메모리에 모은다. 모델과 Langfuse 서버에는 연결하지 않는다."""

    def no_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"단위 테스트의 외부 요청: {request.url}")

    exporter = InMemorySpanExporter()
    public_key = f"pk-test-{uuid4()}"
    # SDK는 명시적 생성자 인자와 별개로 이 환경 변수를 다시 확인한다.
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    client = Langfuse(
        public_key=public_key,
        secret_key="test-only",
        base_url="http://langfuse.test",
        environment="test",
        tracing_enabled=True,
        sample_rate=1,
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        httpx_client=httpx.Client(transport=httpx.MockTransport(no_http)),
    )
    monkeypatch.setattr(settings, "langfuse_tracing_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", public_key)
    monkeypatch.setattr(tracing, "_client", lambda: client)
    try:
        yield exporter
    finally:
        client.shutdown()


def _alembic_config(database_url: str | None = None) -> Config:
    config = Config(toml_file=AI_DIR / "pyproject.toml")
    if database_url is not None:
        config.attributes["database_url"] = database_url
    return config


@pytest.fixture
def alembic_config() -> Config:
    """DB 주소 없이 리비전 파일만 볼 때 쓰는 Alembic 설정."""
    return _alembic_config()


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """테스트용 PostgreSQL을 띄우고 ai 마이그레이션을 끝까지 적용한 주소. Docker가 필요하다."""
    with PostgresContainer(POSTGRES_IMAGE, driver="psycopg") as postgres:
        url = postgres.get_connection_url()
        command.upgrade(_alembic_config(url), "head")
        yield url


@pytest.fixture
async def db_session(database_url: str) -> AsyncIterator[AsyncSession]:
    """테스트마다 트랜잭션 안에서 쓰는 세션. 커밋해도 테스트가 끝나면 모두 되돌린다."""
    engine = create_async_engine(database_url)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
    await engine.dispose()


_open_graph = AsyncGraphDatabase.driver


def _closed_graph(uri: str, **options: Any) -> AsyncDriver:
    """연결할 수 없는 드라이버. 읽거나 쓰면 다시 시도하지 않고 바로 실패한다."""
    return _open_graph("bolt://127.0.0.1:1", max_transaction_retry_time=0, **options)


@pytest.fixture(autouse=True)
def no_graph(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """graph fixture를 받지 않은 테스트가 개발자의 로컬 프로필 그래프에 쓰지 못하게 닫힌 주소로 돌린다.

    드라이버는 만들 수 있지만(연결은 처음 쓸 때 한다) 그래프를 읽거나 쓰면 연결하지 못해 실패한다.
    """
    monkeypatch.setattr(graph_module.AsyncGraphDatabase, "driver", staticmethod(_closed_graph))
    graph_driver.cache_clear()
    yield
    graph_driver.cache_clear()


@pytest.fixture(scope="session")
def graph_server() -> Iterator[Neo4jContainer]:
    """테스트용 Neo4j. Docker가 필요하다."""
    with Neo4jContainer(NEO4J_IMAGE) as neo4j:
        yield neo4j


@pytest.fixture
async def graph(
    graph_server: Neo4jContainer, no_graph: None, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncDriver]:
    """테스트마다 비운 프로필 그래프. graph_driver()도 이 그래프를 돌려준다."""
    monkeypatch.setattr(graph_module.AsyncGraphDatabase, "driver", _open_graph)
    monkeypatch.setattr(settings, "graph_uri", graph_server.get_connection_url())
    monkeypatch.setattr(settings, "graph_user", graph_server.username)
    monkeypatch.setattr(settings, "graph_password", SecretStr(graph_server.password))
    driver = graph_driver()
    await driver.execute_query("MATCH (n) DETACH DELETE n")
    try:
        yield driver
    finally:
        await driver.close()


@pytest.fixture(scope="session")
def sample_posting() -> PostingContent:
    """가상 공고: 결제 플랫폼 백엔드 개발자 (tests/samples/README.md)."""
    return PostingContent.model_validate_json((SAMPLES_DIR / "posting.json").read_bytes())


@pytest.fixture(scope="session")
def sample_company() -> CompanyProfile:
    return CompanyProfile.model_validate_json((SAMPLES_DIR / "company.json").read_bytes())


@pytest.fixture(scope="session")
def sample_applicants() -> dict[str, ApplicantDetail]:
    """가상 지원자 8명. 이름으로 찾는다. 지원자별 정답은 tests/samples/README.md에 있다."""
    applicants = TypeAdapter(list[ApplicantDetail]).validate_json((SAMPLES_DIR / "applicants.json").read_bytes())
    return {applicant.info.name: applicant for applicant in applicants}
