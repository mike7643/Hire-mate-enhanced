import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from hiremate_ai.config import Settings
from hiremate_ai.worker import __main__ as worker_main


async def test_local_worker_leaves_pending_external_batches_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    async def xlen(stream: str) -> int:
        assert stream.endswith(":application-batch")
        return 1

    async def close() -> None:
        pass

    async def run(stop: asyncio.Event) -> None:
        pass

    def forbid_external_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("local 워커가 외부 배치 클라이언트를 열었다")

    monkeypatch.setattr(worker_main, "settings", Settings(_env_file=None, llm_profile="local"))
    monkeypatch.setattr(worker_main, "connect", lambda *_: SimpleNamespace(xlen=xlen, aclose=close))
    monkeypatch.setattr(worker_main, "session_factory", lambda: None)
    monkeypatch.setattr(worker_main, "get_worker_llm", lambda: None)
    monkeypatch.setattr(worker_main, "job_handlers", lambda *_, **__: {})
    monkeypatch.setattr(worker_main, "Worker", lambda *_, **__: SimpleNamespace(run=run))
    monkeypatch.setattr(worker_main.httpx, "AsyncClient", forbid_external_client)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *_: None)

    await worker_main.main()
