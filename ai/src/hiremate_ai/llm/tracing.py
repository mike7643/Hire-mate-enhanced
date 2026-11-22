"""Langfuse 추적. 명시적으로 켠 요청과 작업만 지정한 서버에 기록한다.

ai 서버는 요청 하나(LlmDep을 받는 경로), AI 워커는 작업 하나를 기록(trace) 하나로 묶는다.
그 안의 그래프 노드, 모델 호출, 도구 호출은 LangChain이 하위 단계로 남긴다.
실제 모델 평가(tests/eval)는 묶지 않아 문항마다 그래프 실행이나 모델 호출 하나가 기록 하나다.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import cache
from urllib.parse import urlsplit

from langchain_core.tracers.context import register_configure_hook
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langsmith import tracing_context
from opentelemetry.sdk.trace import TracerProvider

from hiremate_ai.config import settings


class TracingNotConfiguredError(RuntimeError):
    """추적을 켰는데 Langfuse 연결 설정이 없을 때."""


_callback: ContextVar[CallbackHandler | None] = ContextVar("hiremate_langfuse_callback", default=None)
# 명시적 요청 범위에서만 LangGraph와 하위 LangChain 호출에 같은 콜백을 상속한다.
register_configure_hook(_callback, inheritable=True)


@contextmanager
def traced(name: str | None = None) -> Iterator[None]:
    """요청마다 콜백을 분리한다. SSE 스트림과 병렬 하위 호출이 끝난 뒤 상위 기록을 닫는다.

    name이 있으면 모두 name 기록 하나로 묶고, 없으면 바깥 LangChain 실행(그래프, 모델 호출)마다 기록 하나다.
    """
    # LangChain이 의존하는 LangSmith는 제거할 수 없지만, 이전 환경 변수로 이중 전송되면 안 된다.
    with tracing_context(enabled=False):
        if not settings.langfuse_tracing_enabled:
            yield
            return
        client = _client()
        token = _callback.set(CallbackHandler(public_key=settings.langfuse_public_key.strip()))
        try:
            scope = client.start_as_current_observation(name=name, as_type="agent") if name else nullcontext()
            with scope:
                yield
        finally:
            _callback.reset(token)


@cache
def _client() -> Langfuse:
    public_key = settings.langfuse_public_key.strip()
    secret_key = settings.langfuse_secret_key.get_secret_value()
    address = urlsplit(settings.langfuse_base_url.strip())
    if not public_key or not secret_key:
        raise TracingNotConfiguredError(
            "LANGFUSE_TRACING_ENABLED를 켰는데 LANGFUSE_PUBLIC_KEY 또는 LANGFUSE_SECRET_KEY가 비어 있다."
        )
    if address.scheme not in {"http", "https"} or not address.hostname or address.username or address.password:
        raise TracingNotConfiguredError("LANGFUSE_BASE_URL에 인증 정보 없는 Langfuse HTTP(S) 주소를 설정한다.")
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.langfuse_base_url.strip().rstrip("/"),
        environment=settings.langfuse_environment,
        tracing_enabled=True,
        sample_rate=1,
        # 다른 OpenTelemetry 사용처와 exporter를 공유하지 않는다.
        tracer_provider=TracerProvider(),
    )
