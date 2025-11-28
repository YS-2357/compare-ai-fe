"""LangGraph 워크플로우와 각 LLM 호출 노드를 정의하고 스트림 이벤트를 노출한다."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, AsyncIterator, TypedDict, cast
import json

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.models import ChatPerplexity
from langchain_upstage import ChatUpstage
try:
    from langchain_mistralai.chat_models import ChatMistralAI
except ImportError:  # pragma: no cover
    ChatMistralAI = None  # type: ignore[assignment]

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None  # type: ignore[assignment]

try:
    from langchain_cohere import ChatCohere
except ImportError:  # pragma: no cover
    ChatCohere = None  # type: ignore[assignment]
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from app.logger import get_logger

# LangSmith UUID v7 지원
try:
    from langsmith import uuid7 as create_uuid
except ImportError:
    from uuid import uuid4 as create_uuid

# 환경변수 로드
load_dotenv()

# LangSmith 추적 설정 (노트북 파일 기준)
logging.langsmith("API-LangGraph-Test")

logger = get_logger(__name__)
DEFAULT_MAX_TURNS = 3


def _preview(text: str, limit: int = 80) -> str:
    """긴 문자열을 로그에 표시하기 위한 요약 버전으로 변환한다."""

    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


async def _summarize_content(llm: Any, content: str, label: str) -> str:
    """모델 응답을 짧게 요약한다."""

    summary_prompt = (
        "다음 답변을 핵심만 2문장 이하(400자 이내)로 매우 간결하게 요약하세요.\n\n"
        f"답변:\n{content}\n"
    )
    try:
        response = await _ainvoke(llm, summary_prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return str(text)
    except Exception as exc:
        logger.warning("%s 요약 실패: %s", label, exc)
        return content[:200]


def merge_dicts(existing: dict | None, new: dict | None) -> dict:
    """LangGraph 상태 병합 시 딕셔너리를 병합한다."""

    merged: dict = dict(existing or {})
    merged.update(new or {})
    return merged


class GraphState(TypedDict, total=False):
    """LangGraph 실행 시 공유되는 상태 정의."""

    question: Annotated[str, "Question"]
    max_turns: Annotated[int | None, "최대 턴 수"]
    conversation_history: Annotated[list[dict[str, str]] | None, "전체 대화 히스토리"]
    history_summary: Annotated[str | None, "요약된 대화"]
    turn: Annotated[int | None, "현재 턴 인덱스"]
    current_inputs: Annotated[dict[str, str] | None, merge_dicts]
    active_models: Annotated[list[str] | None, "활성화된 모델 목록"]

    openai_answer: Annotated[str | None, "OpenAI 응답"]
    gemini_answer: Annotated[str | None, "Google Gemini 응답"]
    anthropic_answer: Annotated[str | None, "Anthropic Claude 응답"]
    upstage_answer: Annotated[str | None, "Upstage 응답"]
    perplexity_answer: Annotated[str | None, "Perplexity 응답"]
    mistral_answer: Annotated[str | None, "Mistral AI 응답"]
    groq_answer: Annotated[str | None, "Groq 응답"]
    cohere_answer: Annotated[str | None, "Cohere 응답"]

    raw_responses: Annotated[dict[str, str] | None, merge_dicts]
    self_summaries: Annotated[dict[str, str] | None, merge_dicts]
    openai_summary: Annotated[str | None, "OpenAI 자기 요약"]
    gemini_summary: Annotated[str | None, "Gemini 자기 요약"]
    anthropic_summary: Annotated[str | None, "Anthropic 자기 요약"]
    upstage_summary: Annotated[str | None, "Upstage 자기 요약"]
    perplexity_summary: Annotated[str | None, "Perplexity 자기 요약"]
    mistral_summary: Annotated[str | None, "Mistral 자기 요약"]
    groq_summary: Annotated[str | None, "Groq 자기 요약"]
    cohere_summary: Annotated[str | None, "Cohere 자기 요약"]

    openai_status: Annotated[dict[str, Any] | None, "OpenAI 호출 상태"]
    gemini_status: Annotated[dict[str, Any] | None, "Gemini 호출 상태"]
    anthropic_status: Annotated[dict[str, Any] | None, "Anthropic 호출 상태"]
    upstage_status: Annotated[dict[str, Any] | None, "Upstage 호출 상태"]
    perplexity_status: Annotated[dict[str, Any] | None, "Perplexity 호출 상태"]
    mistral_status: Annotated[dict[str, Any] | None, "Mistral 호출 상태"]
    groq_status: Annotated[dict[str, Any] | None, "Groq 호출 상태"]
    cohere_status: Annotated[dict[str, Any] | None, "Cohere 호출 상태"]
    summary_model: Annotated[str | None, "요약에 사용한 모델"]

    messages: Annotated[list, add_messages]


def _default_active_models() -> list[str]:
    return list(NODE_CONFIG.keys())


def _model_label(node_name: str) -> str:
    meta = NODE_CONFIG.get(node_name)
    return meta["label"] if meta else node_name


def _simplify_error_message(error: Any) -> str:
    """예외 객체에서 핵심 메시지만 추출한다."""

    def from_mapping(data: dict[str, Any]) -> str | None:
        for key in ("detail", "message", "error"):
            value = data.get(key)
            if value:
                return str(value)
        body = data.get("body")
        if isinstance(body, dict):
            nested = from_mapping(body)
            if nested:
                return nested
        elif body:
            return str(body)
        return None

    response = getattr(error, "response", None)
    if response is not None:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                text = from_mapping(payload)
                if text:
                    return text
        except Exception:
            pass
        try:
            body = response.text
            if body:
                return body.strip().splitlines()[0][:200]
        except Exception:
            pass

    body_attr = getattr(error, "body", None)
    if body_attr:
        if isinstance(body_attr, dict):
            text = from_mapping(body_attr)
            if text:
                return text
        if isinstance(body_attr, str):
            return body_attr.strip().splitlines()[0][:200]

    if hasattr(error, "args") and error.args:
        candidate = error.args[0]
        if isinstance(candidate, dict):
            text = from_mapping(candidate)
            if text:
                return text
        if isinstance(candidate, str):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    text = from_mapping(parsed)
                    if text:
                        return text
            except Exception:
                pass
            return candidate.strip().splitlines()[0][:200]

    return str(error).strip().splitlines()[0][:200]


def build_status_from_response(
    response: Any, default_status: int = 200, detail: str = "success"
) -> dict[str, Any]:
    """LLM 응답 객체에서 상태 메타데이터를 추출한다.

    Args:
        response: LangChain LLM 응답 객체.
        default_status: 응답에 상태 코드가 없을 때 사용할 기본 값.
        detail: 응답에 finish reason이 없을 때 사용할 기본 메시지.

    Returns:
        dict[str, Any]: `status`, `detail` 키를 포함한 상태 정보.
    """
    metadata = getattr(response, "response_metadata", None) or {}
    status = metadata.get("status_code") or metadata.get("status") or metadata.get("http_status")
    detail_text = metadata.get("finish_reason") or metadata.get("reason") or detail
    return {"status": status or default_status, "detail": detail_text}


def build_status_from_error(error: Exception) -> dict[str, Any]:
    """예외 객체를 API 상태 표현으로 변환한다.

    Args:
        error: 발생한 예외 인스턴스.

    Returns:
        dict[str, Any]: 실패 상태와 메시지를 담은 상태 정보.
    """
    status = cast(int | None, getattr(error, "status_code", None))
    if status is None:
        response = getattr(error, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    detail = _simplify_error_message(error)
    return {"status": status or "error", "detail": detail}


def format_response_message(label: str, payload: Any) -> tuple[str, str]:
    """메시지 로그에 저장할 간단한 (role, content) 튜플을 생성한다.

    Args:
        label: 메시지 헤더(모델명 또는 오류 등).
        payload: 원본 응답 또는 예외 객체.

    Returns:
        tuple[str, str]: `("assistant", "[라벨] 내용")` 형태의 메시지.
    """
    return ("assistant", f"[{label}] {payload}")


def init_question(state: GraphState) -> GraphState:
    """그래프 초기 상태를 검증하고 기본 메시지를 설정한다."""

    question = state.get("question")
    if not question:
        raise ValueError("질문이 비어 있습니다.")

    max_turns = state.get("max_turns") or DEFAULT_MAX_TURNS
    active_models = state.get("active_models") or list(NODE_CONFIG.keys())
    current_inputs = state.get("current_inputs") or {
        NODE_CONFIG[node]["label"]: question for node in active_models
    }
    history = state.get("conversation_history")
    if not history:
        history = [{"role": "user", "content": question}]
    turn_value = state.get("turn") or 1

    logger.debug("질문 초기화: %s", _preview(question))
    return GraphState(
        question=question,
        max_turns=max_turns,
        turn=turn_value,
        conversation_history=history,
        history_summary=state.get("history_summary"),
        current_inputs=current_inputs,
        active_models=active_models,
        raw_responses=state.get("raw_responses") or {},
        self_summaries=state.get("self_summaries") or {},
        messages=state.get("messages") or [("user", question)],
    )


async def _ainvoke(llm: Any, question: str) -> Any:
    """주어진 LLM에서 비동기 호출을 수행한다."""

    if hasattr(llm, "ainvoke"):
        return await llm.ainvoke(question)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, llm.invoke, question)


async def call_openai(state: GraphState) -> GraphState:
    """OpenAI 모델을 호출하고 응답/상태를 반환한다.

    Args:
        state: 질문을 포함한 그래프 상태.

    Returns:
        GraphState: OpenAI 응답/상태/메시지를 담은 상태 델타.
    """
    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("OpenAI") or question
    logger.debug("OpenAI 호출 시작")
    try:
        llm = ChatOpenAI(model="gpt-5-nano")
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "OpenAI")
        logger.info("OpenAI 응답 완료: %s", status.get("detail"))
        return GraphState(
            openai_answer=content,
            openai_status=status,
            openai_summary=summary,
            raw_responses={"OpenAI": content},
            self_summaries={"OpenAI": summary},
            messages=[format_response_message("OpenAI", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("OpenAI 호출 실패: %s", exc)
        return GraphState(
            openai_status=status,
            messages=[format_response_message("OpenAI 오류", exc)],
        )


async def call_gemini(state: GraphState) -> GraphState:
    """Google Gemini 모델을 호출한다.

    Args:
        state: 질문을 포함한 그래프 상태.

    Returns:
        GraphState: Gemini 응답/상태/메시지를 담은 상태 델타.
    """
    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Gemini") or question
    logger.debug("Gemini 호출 시작")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Gemini")
        logger.info("Gemini 응답 완료: %s", status.get("detail"))
        return GraphState(
            gemini_answer=content,
            gemini_status=status,
            gemini_summary=summary,
            raw_responses={"Gemini": content},
            self_summaries={"Gemini": summary},
            messages=[format_response_message("Gemini", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Gemini 호출 실패: %s", exc)
        return GraphState(
            gemini_status=status,
            messages=[format_response_message("Gemini 오류", exc)],
        )


async def call_anthropic(state: GraphState) -> GraphState:
    """Anthropic Claude 모델을 호출한다.

    Args:
        state: 질문을 포함한 그래프 상태.

    Returns:
        GraphState: Claude 응답/상태/메시지를 담은 상태 델타.
    """
    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Anthropic") or question
    logger.debug("Anthropic 호출 시작")
    try:
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Anthropic")
        logger.info("Anthropic 응답 완료: %s", status.get("detail"))
        return GraphState(
            anthropic_answer=content,
            anthropic_status=status,
            anthropic_summary=summary,
            raw_responses={"Anthropic": content},
            self_summaries={"Anthropic": summary},
            messages=[format_response_message("Anthropic", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Anthropic 호출 실패: %s", exc)
        return GraphState(
            anthropic_status=status,
            messages=[format_response_message("Anthropic 오류", exc)],
        )


async def call_upstage(state: GraphState) -> GraphState:
    """Upstage Solar 모델을 호출한다.

    Args:
        state: 질문을 포함한 그래프 상태.

    Returns:
        GraphState: Upstage 응답/상태/메시지를 담은 상태 델타.
    """
    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Upstage") or question
    logger.debug("Upstage 호출 시작")
    try:
        llm = ChatUpstage(model="solar-mini")
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Upstage")
        logger.info("Upstage 응답 완료: %s", status.get("detail"))
        return GraphState(
            upstage_answer=content,
            upstage_status=status,
            upstage_summary=summary,
            raw_responses={"Upstage": content},
            self_summaries={"Upstage": summary},
            messages=[format_response_message("Upstage", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Upstage 호출 실패: %s", exc)
        return GraphState(
            upstage_status=status,
            messages=[format_response_message("Upstage 오류", exc)],
        )


async def call_perplexity(state: GraphState) -> GraphState:
    """Perplexity Sonar 모델을 호출한다.

    Args:
        state: 질문을 포함한 그래프 상태.

    Returns:
        GraphState: Perplexity 응답/상태/메시지를 담은 상태 델타.
    """
    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Perplexity") or question
    logger.debug("Perplexity 호출 시작")
    try:
        llm = ChatPerplexity(
            model="sonar",
            temperature=0.2,
            top_p=0.9,
            search_domain_filter=["perplexity.ai"],
            return_images=False,
            return_related_questions=True,
            top_k=0,
            stream=False,
        )
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Perplexity")
        logger.info("Perplexity 응답 완료: %s", status.get("detail"))
        return GraphState(
            perplexity_answer=content,
            perplexity_status=status,
            perplexity_summary=summary,
            raw_responses={"Perplexity": content},
            self_summaries={"Perplexity": summary},
            messages=[format_response_message("Perplexity", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Perplexity 호출 실패: %s", exc)
        return GraphState(
            perplexity_status=status,
            messages=[format_response_message("Perplexity 오류", exc)],
        )


async def call_mistral(state: GraphState) -> GraphState:
    """Mistral AI 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Mistral") or question
    if ChatMistralAI is None:
        error = RuntimeError("langchain-mistralai 패키지가 설치되어 있지 않습니다.")
        logger.warning("Mistral AI 사용 불가: %s", error)
        status = build_status_from_error(error)
        return GraphState(
            mistral_status=status,
            messages=[format_response_message("Mistral 오류", error)],
        )
    try:
        llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Mistral")
        logger.info("Mistral 응답 완료: %s", status.get("detail"))
        return GraphState(
            mistral_answer=content,
            mistral_status=status,
            mistral_summary=summary,
            raw_responses={"Mistral": content},
            self_summaries={"Mistral": summary},
            messages=[format_response_message("Mistral", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Mistral 호출 실패: %s", exc)
        return GraphState(
            mistral_status=status,
            messages=[format_response_message("Mistral 오류", exc)],
        )


async def call_groq(state: GraphState) -> GraphState:
    """Groq 기반 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Groq") or question
    if ChatGroq is None:
        error = RuntimeError("langchain-groq 패키지가 설치되어 있지 않습니다.")
        logger.warning("Groq 사용 불가: %s", error)
        status = build_status_from_error(error)
        return GraphState(
            groq_status=status,
            messages=[format_response_message("Groq 오류", error)],
        )
    try:
        llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Groq")
        logger.info("Groq 응답 완료: %s", status.get("detail"))
        return GraphState(
            groq_answer=content,
            groq_status=status,
            groq_summary=summary,
            raw_responses={"Groq": content},
            self_summaries={"Groq": summary},
            messages=[format_response_message("Groq", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Groq 호출 실패: %s", exc)
        return GraphState(
            groq_status=status,
            messages=[format_response_message("Groq 오류", exc)],
        )


async def call_cohere(state: GraphState) -> GraphState:
    """Cohere Command 모델을 호출한다."""

    inputs = state.get("current_inputs") or {}
    question = state["question"]
    prompt = inputs.get("Cohere") or question
    if ChatCohere is None:
        error = RuntimeError("langchain-cohere 패키지가 설치되어 있지 않습니다.")
        logger.warning("Cohere 사용 불가: %s", error)
        status = build_status_from_error(error)
        return GraphState(
            cohere_status=status,
            messages=[format_response_message("Cohere 오류", error)],
        )
    try:
        llm = ChatCohere(model="command-r-plus", temperature=0)
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        status = build_status_from_response(response)
        summary = await _summarize_content(llm, content, "Cohere")
        logger.info("Cohere 응답 완료: %s", status.get("detail"))
        return GraphState(
            cohere_answer=content,
            cohere_status=status,
            cohere_summary=summary,
            raw_responses={"Cohere": content},
            self_summaries={"Cohere": summary},
            messages=[format_response_message("Cohere", response)],
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("Cohere 호출 실패: %s", exc)
        return GraphState(
            cohere_status=status,
            messages=[format_response_message("Cohere 오류", exc)],
        )


NODE_CONFIG: dict[str, dict[str, str]] = {
    "call_openai": {"label": "OpenAI", "answer_key": "openai_answer", "status_key": "openai_status"},
    "call_gemini": {"label": "Gemini", "answer_key": "gemini_answer", "status_key": "gemini_status"},
    "call_anthropic": {"label": "Anthropic", "answer_key": "anthropic_answer", "status_key": "anthropic_status"},
    "call_perplexity": {"label": "Perplexity", "answer_key": "perplexity_answer", "status_key": "perplexity_status"},
    "call_upstage": {"label": "Upstage", "answer_key": "upstage_answer", "status_key": "upstage_status"},
    "call_mistral": {"label": "Mistral", "answer_key": "mistral_answer", "status_key": "mistral_status"},
    "call_groq": {"label": "Groq", "answer_key": "groq_answer", "status_key": "groq_status"},
    "call_cohere": {"label": "Cohere", "answer_key": "cohere_answer", "status_key": "cohere_status"},
}


def dispatch_llm_calls(state: GraphState) -> list[Send]:
    """Send API를 활용해 각 LLM 노드를 동시에 실행할 태스크 목록을 생성한다."""

    question = state.get("question")
    if not question:
        raise ValueError("질문이 비어 있습니다.")
    active_models = state.get("active_models") or _default_active_models()
    logger.info("LLM fan-out 실행: %s", ", ".join(active_models))
    return [Send(node_name, state) for node_name in active_models]


def build_workflow():
    """StateGraph를 구성하고 LangGraph 앱으로 컴파일한다.

    Returns:
        Any: 컴파일된 LangGraph 애플리케이션.
    """
    logger.debug("LangGraph 워크플로우 컴파일 시작")
    workflow = StateGraph(GraphState)
    workflow.add_node("init_question", init_question)
    workflow.add_node("call_openai", call_openai)
    workflow.add_node("call_gemini", call_gemini)
    workflow.add_node("call_anthropic", call_anthropic)
    workflow.add_node("call_upstage", call_upstage)
    workflow.add_node("call_perplexity", call_perplexity)
    workflow.add_node("call_mistral", call_mistral)
    workflow.add_node("call_groq", call_groq)
    workflow.add_node("call_cohere", call_cohere)

    workflow.add_conditional_edges("init_question", dispatch_llm_calls)

    workflow.add_edge("call_openai", END)
    workflow.add_edge("call_gemini", END)
    workflow.add_edge("call_anthropic", END)
    workflow.add_edge("call_upstage", END)
    workflow.add_edge("call_perplexity", END)
    workflow.add_edge("call_mistral", END)
    workflow.add_edge("call_groq", END)
    workflow.add_edge("call_cohere", END)

    workflow.set_entry_point("init_question")
    compiled = workflow.compile()
    logger.info("LangGraph 워크플로우 컴파일 완료")
    return compiled


_app = None


def get_app():
    """싱글턴 형태로 컴파일된 LangGraph 앱을 반환한다.

    Returns:
        Any: 재사용 가능한 LangGraph 애플리케이션 인스턴스.
    """
    global _app
    if _app is None:
        _app = build_workflow()
    return _app


async def _summarize_history(history: list[dict[str, str]] | None, limit: int = 400) -> str | None:
    """과거 대화 이력을 2문장 이내로 요약한다."""

    if not history:
        return None

    text_lines = [f"{item.get('role')}: {item.get('content')}" for item in history if item.get("content")]
    history_text = "\n".join(text_lines)
    prompt = (
        "다음 대화 이력을 2문장 이하, 400자 이내로 요약하세요. 핵심 논점만 남기고 세부사항은 생략합니다.\n\n"
        f"{history_text}"
    )
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    try:
        response = await _ainvoke(llm, prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return str(content)[:limit]
    except Exception as exc:  # pragma: no cover - 요약 실패 시 안전 폴백
        logger.warning("대화 요약 실패, 원본을 절단해 사용합니다: %s", exc)
        compact = " ".join(history_text.split())
        return compact[:limit]


def _build_current_inputs(question: str, active_models: list[str]) -> dict[str, str]:
    """현 턴에서 사용할 모델별 입력 프롬프트를 생성한다."""

    prompts: dict[str, str] = {}
    for node_name in active_models:
        label = NODE_CONFIG[node_name]["label"]
        sections = [question]
        sections.append(
            "사용자 질문에 최신 답변을 제시하세요. 필요하면 이전 맥락을 반영하세요."
        )
        sections.append("응답은 5문장 이하, 600자 이내로 간결하게 작성하세요.")
        prompts[label] = "\n\n".join([part for part in sections if part])
    return prompts


def _normalize_messages(messages: list | None) -> list[dict[str, str]]:
    """Streamlit 표시를 위해 메시지를 표준화한다.

    Args:
        messages: LangGraph에서 누적된 메시지 리스트.

    Returns:
        list[dict[str, str]]: `{"role": ..., "content": ...}` 형태 리스트.
    """
    normalized: list[dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, (list, tuple)) and len(message) == 2:
            role, content = message
            normalized.append({"role": str(role), "content": str(content)})
        else:
            normalized.append({"role": "system", "content": str(message)})
    return normalized


def _extend_unique_messages(
    target: list[dict[str, str]], new_messages: list[dict[str, str]] | None, seen: set[tuple[str, str]]
) -> None:
    """중복 없이 메시지를 추가한다.

    Args:
        target: 메시지를 누적할 리스트.
        new_messages: 새로 추가할 메시지 목록.
        seen: (role, content) 조합을 저장한 중복 체크 세트.
    """
    for message in new_messages or []:
        role = str(message.get("role"))
        content = str(message.get("content"))
        key = (role, content)
        if key in seen:
            continue
        seen.add(key)
        target.append({"role": role, "content": content})


async def stream_graph(
    question: str, *, turn: int = 1, max_turns: int | None = None, history: list[dict[str, str]] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """질문을 받아 LangGraph 워크플로우에서 발생하는 이벤트를 스트리밍한다.

    Args:
        question: 사용자 질문 문자열.
        turn: 현재 턴 인덱스(사용자 입력 횟수).
        max_turns: 허용되는 최대 턴 수.
        history: 이전 대화 이력(`role`, `content`).

    Yields:
        dict[str, Any]: `type=partial` 이벤트(모델명/응답/상태/메시지).

    Raises:
        ValueError: 질문이 비어 있는 경우.
    """
    if not question or not question.strip():
        raise ValueError("질문을 입력해주세요.")

    resolved_max_turns = max_turns or DEFAULT_MAX_TURNS
    if turn > resolved_max_turns:
        warning = f"최대 턴({resolved_max_turns})을 초과했습니다. 새 질문으로 시작해주세요."
        logger.warning("턴 초과 - 실행 중단: turn=%s, max=%s", turn, resolved_max_turns)
        yield {"type": "error", "message": warning, "node": None, "model": None, "turn": turn}
        return

    logger.info("LangGraph 스트림 실행: %s", _preview(question))
    base_question = question.strip()
    app = get_app()
    start_time = time.perf_counter()
    active_models = list(NODE_CONFIG.keys())
    current_inputs = _build_current_inputs(base_question, active_models)
    conversation_history = list(history or [])
    conversation_history.append({"role": "user", "content": base_question})
    state_inputs: GraphState = {
        "question": base_question,
        "max_turns": resolved_max_turns,
        "turn": turn,
        "conversation_history": conversation_history,
        "current_inputs": current_inputs,
        "active_models": active_models,
    }

    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": str(create_uuid())})
    try:
        async for event in app.astream(state_inputs, config=config):
            turn_index = state_inputs.get("turn") or 1
            for node_name, state in event.items():
                if node_name == "__end__":
                    continue
                if node_name not in NODE_CONFIG:
                    continue
                meta = NODE_CONFIG[node_name]
                logger.debug("이벤트 수신: %s (turn=%s)", meta["label"], turn_index)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield {
                    "model": meta["label"],
                    "node": node_name,
                    "answer": state.get(meta["answer_key"]),
                    "status": state.get(meta["status_key"]) or {},
                    "messages": _normalize_messages(state.get("messages")),
                    "type": "partial",
                    "turn": turn_index,
                    "elapsed_ms": elapsed_ms,
                }
    except Exception as exc:
        logger.error("LangGraph 스트림 오류: %s", exc)
        yield {
            "type": "error",
            "message": str(exc),
            "node": None,
            "model": None,
            "turn": turn,
        }
