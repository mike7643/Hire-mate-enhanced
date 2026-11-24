"""Ollama 문서 정제. 실험과 같은 응답 스키마를 변형 없이 보낸다."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from hiremate_ai.llm.call_log import CallLog, CallRecord
from hiremate_ai.llm.chat import CUT, LlmFailureReason, LlmOutputError, consume_model_call
from hiremate_ai.llm.tracing import native_generation


class _Message(BaseModel):
    content: str


class _OllamaReply(BaseModel):
    model: str
    message: _Message
    done: bool
    done_reason: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None

    def usage(self) -> dict[str, int]:
        tokens = {k: v for k, v in (("input", self.prompt_eval_count), ("output", self.eval_count)) if v is not None}
        if len(tokens) == 2:
            tokens["total"] = sum(tokens.values())
        return tokens


class OllamaNormalizationLlm:
    """OpenAI 호환 SDK를 통하지 않는 정제 전용 구조화 호출."""

    def __init__(self, endpoint: httpx.URL, model: str, call_log: CallLog, *, timeout: float) -> None:
        self._endpoint = endpoint
        self._model = model
        self._log = call_log
        self._timeout = timeout

    async def structured[T: BaseModel](
        self, output: type[T], *, agent: str, system: str, user: str, images: Sequence[str] = ()
    ) -> T:
        if images:
            raise ValueError("Ollama 문서 정제는 PDF에서 추출한 텍스트만 받는다")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": output.model_json_schema(by_alias=True),
            "options": {"temperature": 0, "num_ctx": 32768, "num_predict": -1, "seed": 20261116},
            "keep_alive": "15m",
        }
        started = time.perf_counter()
        reply: _OllamaReply | None = None
        consume_model_call()
        with native_generation(agent, self._model) as span:
            try:
                reply = await self._chat(payload)
                result = self._parse(output, reply)
            except Exception as error:
                if span is not None:
                    span.update(
                        level="ERROR",
                        status_message=type(error).__name__,
                        usage_details=reply.usage() if reply else None,
                    )
                await self._record(agent, system, user, started, reply, error=error)
                raise
            data = result.model_dump(mode="json")
            if span is not None:
                span.update(usage_details=reply.usage())
            await self._record(agent, system, user, started, reply, output=data)
        return result

    async def _chat(self, payload: dict[str, Any]) -> _OllamaReply:
        # 실험의 생성 조건은 그대로 두되, 모델이 끝나지 않으면 호출 시간으로 끊는다.
        async with (
            asyncio.timeout(self._timeout),
            httpx.AsyncClient(timeout=self._timeout, trust_env=False, follow_redirects=False) as client,
        ):
            response = await client.post(self._endpoint, json=payload)
            response.raise_for_status()
        try:
            return _OllamaReply.model_validate_json(response.content)
        except ValidationError:
            raise LlmOutputError("Ollama 문서 정제 응답 형식이 올바르지 않다") from None

    def _parse[T: BaseModel](self, output: type[T], reply: _OllamaReply) -> T:
        if reply.model != self._model:
            raise LlmOutputError("Ollama가 설정한 문서 정제 모델과 다른 모델로 응답했다")
        if reply.done_reason == "length":
            raise LlmOutputError(CUT)
        if not reply.done or reply.done_reason != "stop":
            raise LlmOutputError("Ollama 문서 정제 응답이 정상적으로 끝나지 않았다")
        try:
            return output.model_validate_json(reply.message.content)
        except ValidationError:
            raise LlmOutputError(
                "Ollama 문서 정제 결과가 스키마와 다르다", reason=LlmFailureReason.VALIDATION
            ) from None

    async def _record(
        self,
        agent: str,
        system: str,
        user: str,
        started: float,
        reply: _OllamaReply | None,
        *,
        output: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        await self._log.record(
            CallRecord(
                agent=agent,
                model=self._model,
                status="ERROR" if error else "OK",
                system=system,
                user=user,
                output=output,
                error=f"{type(error).__name__}: {error}" if error else None,
                input_tokens=reply.prompt_eval_count if reply is not None else None,
                output_tokens=reply.eval_count if reply is not None else None,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        )
