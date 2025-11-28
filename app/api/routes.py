"""API 라우터와 엔드포인트 정의."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.logger import get_logger
from app.auth import AuthenticatedUser, get_current_user
from app.rate_limit import enforce_daily_limit
from app.config import get_settings
from app.services import stream_graph
from app.services.langgraph import DEFAULT_MAX_TURNS

router = APIRouter()
logger = get_logger(__name__)


def _preview(text: str, limit: int = 80) -> str:
    """로그 출력을 위해 문자열을 요약한다."""

    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


class AskRequest(BaseModel):
    """질문을 포함하는 요청 스키마."""

    question: str
    turn: int | None = None
    max_turns: int | None = None
    history: list[dict[str, str]] | None = None


@router.get("/health")
async def health():
    """서비스 가용성을 확인하는 헬스 체크 응답을 반환한다.

    Returns:
        dict: `{"status": "ok"}` 구조의 간단한 상태 객체.
    """

    return {"status": "ok"}


@router.post("/api/ask")
async def ask_question(payload: AskRequest, user: AuthenticatedUser = Depends(get_current_user)):
    """LangGraph 워크플로우를 스트림 형태로 실행한다.

    Args:
        payload: 질문 문자열을 담은 요청 본문.
        user: 인증된 사용자 정보.

    Returns:
        StreamingResponse: partial/summary 이벤트가 줄 단위 JSON으로 전달되는 스트림 응답.
    """

    settings = get_settings()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    turn = payload.turn or 1
    max_turns = payload.max_turns or DEFAULT_MAX_TURNS
    history = payload.history or []
    if turn < 1:
        turn = 1

    logger.info("질문 수신: %s", _preview(question))

    if not user.get("bypass"):
        await enforce_daily_limit(user["sub"], settings.daily_usage_limit)

    async def response_stream():
        answers = {}
        api_status = {}
        durations_ms: dict[str, int] = {}
        messages = [{"role": "user", "content": question}]
        seen_messages = {("user", question)}
        completion_order: list[str] = []
        errors: list[dict[str, str | None]] = []

        def extend_messages(new_messages: list[dict[str, str]] | None):
            for message in new_messages or []:
                role = str(message.get("role"))
                content = str(message.get("content"))
                key = (role, content)
                if key in seen_messages:
                    continue
                seen_messages.add(key)
                messages.append({"role": role, "content": content})

        try:
            async for event in stream_graph(question, turn=turn, max_turns=max_turns, history=history):
                event_type = event.get("type", "partial")
                if event_type == "partial":
                    model = event.get("model")
                    if model:
                        if model not in completion_order:
                            completion_order.append(model)
                        answers[model] = event.get("answer")
                        status = event.get("status")
                        if status:
                            api_status[model] = status
                        elapsed_ms = event.get("elapsed_ms")
                        if elapsed_ms is not None:
                            durations_ms[model] = int(elapsed_ms)
                        logger.debug("부분 응답 누적: %s", model)
                    extend_messages(event.get("messages"))
                elif event_type == "error":
                    logger.warning(
                        "스트림 오류 이벤트 수신 (node=%s, model=%s): %s",
                        event.get("node"),
                        event.get("model"),
                        event.get("message"),
                    )
                    errors.append(
                        {
                            "message": event.get("message"),
                            "node": event.get("node"),
                            "model": event.get("model"),
                        }
                    )
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover
            error_event = {"type": "error", "message": str(exc), "node": None, "model": None}
            errors.append(
                {
                    "message": str(exc),
                    "node": None,
                    "model": None,
                }
            )
            logger.error("응답 스트림 처리 중 오류: %s", exc)
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
        finally:
            primary_model = next((model for model in completion_order if answers.get(model)), None)
            primary_answer = (
                {
                    "model": primary_model,
                    "answer": answers.get(primary_model),
                    "status": api_status.get(primary_model),
                }
                if primary_model
                else None
            )
            summary = {
                "type": "summary",
                "result": {
                    "question": question,
                    "answers": answers,
                    "api_status": api_status,
                    "durations_ms": durations_ms,
                    "messages": messages,
                    "order": completion_order,
                    "primary_model": primary_model,
                    "primary_answer": primary_answer,
                    "errors": errors,
                    "turn": turn,
                    "max_turns": max_turns,
                },
            }
            logger.info("요약 응답 전송 - 완료 모델 수: %d, 오류 수: %d", len(answers), len(errors))
            yield json.dumps(summary, ensure_ascii=False) + "\n"

    return StreamingResponse(response_stream(), media_type="application/json")
