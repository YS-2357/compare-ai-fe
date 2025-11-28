"""Streamlit UI 엔트리포인트."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL_FILE = Path(__file__).resolve().parents[2] / ".fastapi_url"
DEFAULT_FASTAPI_BASE = FASTAPI_URL_FILE.read_text().strip() if FASTAPI_URL_FILE.exists() else ""

st.set_page_config(page_title="Compare-AI", page_icon="🤖", layout="wide")


def _load_base_url() -> str:
    saved = (
        st.session_state.get("fastapi_base_url")
        or DEFAULT_FASTAPI_BASE
        or os.getenv("FASTAPI_URL", "")
        or st.secrets.get("FASTAPI_URL", "")
    )
    if saved.endswith("/api/ask"):
        return saved.rsplit("/api/ask", 1)[0]
    return saved


def _get_admin_token() -> str:
    # 환경변수에 있어도 UI에 노출되지 않도록 세션 값만 사용
    return st.session_state.get("admin_token", "")


def _get_admin_env_token() -> str:
    return os.getenv("ADMIN_BYPASS_TOKEN", "") or st.secrets.get("ADMIN_BYPASS_TOKEN", "")


def _get_usage_limit() -> str:
    return os.getenv("DAILY_USAGE_LIMIT") or "3"


def _usage_limit_int() -> int:
    try:
        return int(_get_usage_limit())
    except Exception:
        return 3


def _sync_usage_from_headers(resp: requests.Response) -> None:
    """서버 헤더의 사용량 정보를 세션에 반영한다."""

    limit = resp.headers.get("X-Usage-Limit")
    remaining = resp.headers.get("X-Usage-Remaining")
    if limit is not None and limit.isdigit():
        st.session_state["usage_limit"] = int(limit)
    if remaining is not None and remaining.isdigit():
        st.session_state["usage_remaining"] = int(remaining)


def main() -> None:
    st.title("Compare-AI")
    st.caption("여러 LLM 중 내 질문에 가장 잘 답하는 모델을 찾아보세요.")

    with st.sidebar:
        st.header("Backend 설정")
        base_url = st.text_input("FastAPI Base URL", value=_load_base_url(), placeholder="http://127.0.0.1:8000")
        base_url = base_url.rstrip("/")
        st.session_state["fastapi_base_url"] = base_url
        st.caption("예: http://127.0.0.1:8000")

        st.subheader("관리자 우회 토큰 (선택)")
        admin_token = st.text_input("x-admin-bypass", value=_get_admin_token(), type="password")
        st.session_state["admin_token"] = admin_token
        use_admin = st.checkbox("우회 토큰 사용", value=False, help="체크 시 인증/레이트리밋 우회")
        st.session_state["use_admin_bypass"] = use_admin

    ask_url = f"{base_url}/api/ask" if base_url else ""

    # 인증/회원가입 뷰
    if not st.session_state.get("auth_token"):
        st.header("로그인 또는 회원가입")
        email = st.text_input("이메일")
        password = st.text_input("비밀번호", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("회원가입"):
                if not base_url:
                    st.error("FastAPI Base URL을 입력하세요.")
                elif not email or not password:
                    st.warning("이메일/비밀번호를 입력하세요.")
                else:
                    try:
                        resp = requests.post(
                            f"{base_url}/auth/register",
                            json={"email": email, "password": password},
                            timeout=15,
                        )
                        st.write(f"회원가입 상태: {resp.status_code}")
                        st.json(resp.json())
                    except Exception as exc:
                        st.error(f"회원가입 실패: {exc}")
        with col2:
            if st.button("로그인"):
                if not base_url:
                    st.error("FastAPI Base URL을 입력하세요.")
                elif not email or not password:
                    st.warning("이메일/비밀번호를 입력하세요.")
                else:
                    try:
                        resp = requests.post(
                            f"{base_url}/auth/login",
                            json={"email": email, "password": password},
                            timeout=15,
                        )
                        data = resp.json()
                        st.write(f"로그인 상태: {resp.status_code}")
                        if resp.ok and data.get("access_token"):
                            token = f"{data.get('token_type', 'bearer')} {data['access_token']}"
                            st.session_state["auth_token"] = token
                            st.session_state["auth_user"] = data.get("user")
                            st.success("로그인 성공: 토큰 저장 완료")
                            st.rerun()
                        st.json(data)
                    except Exception as exc:
                        st.error(f"로그인 실패: {exc}")
        st.stop()

    # 로그인 후 질문 뷰
    st.header("대화")
    if "usage_remaining" not in st.session_state:
        st.session_state["usage_remaining"] = _usage_limit_int()
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []
    if user := st.session_state.get("auth_user"):
        st.caption(f"로그인됨: {user.get('email')}")
    remaining = st.session_state.get("usage_remaining", _usage_limit_int())
    st.info(f"남은 일일 사용 횟수: **{remaining}회** (관리자 우회 시 제한 없음)")
    if st.button("로그아웃"):
        st.session_state.pop("auth_token", None)
        st.session_state.pop("auth_user", None)
        st.session_state.pop("usage_remaining", None)
        st.session_state.pop("chat_log", None)
        st.rerun()

    chat_area = st.container()
    if st.session_state["chat_log"]:
        for idx, item in enumerate(reversed(st.session_state["chat_log"])):
            with chat_area:
                st.markdown(f"**Q{len(st.session_state['chat_log'])-idx}:** {item.get('question')}")
                answers = item.get("answers") or {}
                sources = item.get("sources") or {}
                events = item.get("events") or []
                if events:
                    st.caption("응답 스트림 (수신 순서)")
                    for ev in events:
                        model = ev.get("model") or "unknown"
                        ans = ev.get("answer")
                        src = ev.get("source")
                        status = ev.get("status") or {}
                        st.write(f"[{model}] {ans}")
                        st.caption(f"status: {status}")
                        if src:
                            st.caption(f"출처: {src}")
                else:
                    for model, answer in answers.items():
                        src = sources.get(model)
                        st.write(f"[{model}] {answer}")
                        if src:
                            st.caption(f"출처: {src}")
                st.divider()
    else:
        chat_area.info("아직 대화가 없습니다. 질문을 입력해보세요.")

    question = st.text_area("질문을 입력하세요", height=120, placeholder="여기에 질문을 입력하세요")
    send_col, reset_col = st.columns([3, 1])
    with send_col:
        send_clicked = st.button("질문하기", use_container_width=True)
    with reset_col:
        if st.button("입력 지우기", use_container_width=True):
            st.session_state["current_question"] = ""
            st.rerun()

    if send_clicked:
        if not question.strip():
            st.warning("질문을 입력해주세요.")
            return
        if not ask_url:
            st.error("FastAPI Base URL을 설정해주세요.")
            return
        headers = {"Content-Type": "application/json"}
        if token := st.session_state.get("auth_token"):
            headers["Authorization"] = token

        # 관리자 우회 토큰 검증: 환경/secret에 설정된 값과 일치할 때만 헤더 추가
        admin_env = _get_admin_env_token()
        admin_input = st.session_state.get("admin_token")
        allow_admin = False
        if st.session_state.get("use_admin_bypass"):
            if admin_env and admin_input and admin_input == admin_env:
                allow_admin = True
            elif admin_input:
                st.warning("관리자 우회 토큰이 일치하지 않아 일반 요청으로 진행합니다.")
            elif not admin_env:
                st.warning("서버에 관리자 우회 토큰이 설정되지 않아 우회할 수 없습니다.")
        if allow_admin:
            headers["x-admin-bypass"] = admin_input
        with st.spinner("질문 보내는 중..."):
            try:
                answers_acc: dict[str, str] = {}
                sources_acc: dict[str, str | None] = {}
                resp = requests.post(
                    ask_url,
                    headers=headers,
                    data=json.dumps({"question": question}),
                    stream=True,
                    timeout=60,
                )
                _sync_usage_from_headers(resp)
                stream_lines = []
                events = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line.decode("utf-8"))
                    except Exception:
                        parsed = line
                    stream_lines.append(parsed)
                    if isinstance(parsed, dict) and parsed.get("type") == "partial":
                        model = parsed.get("model")
                        if model:
                            answers_acc[model] = parsed.get("answer")
                            sources_acc[model] = parsed.get("source")
                            events.append(
                                {
                                    "model": model,
                                    "answer": parsed.get("answer"),
                                    "source": parsed.get("source"),
                                    "status": parsed.get("status"),
                                }
                            )
                # 스트림 완료 후 summary 저장
                summary = None
                for item in stream_lines[::-1]:
                    if isinstance(item, dict) and item.get("type") == "summary":
                        summary = item
                        break
                if summary:
                    result = summary.get("result") or {}
                    answers_acc = result.get("answers") or answers_acc
                    sources_acc = result.get("sources") or sources_acc
                st.session_state["chat_log"].append(
                    {"question": question, "answers": answers_acc, "sources": sources_acc, "events": events}
                )

                # 응답 완료 후 남은 횟수 갱신
                if resp.status_code == 429:
                    st.session_state["usage_remaining"] = 0
                elif resp.ok and not st.session_state.get("use_admin_bypass"):
                    # 서버가 헤더로 내려준 값을 우선 사용하고, 없으면 클라이언트 감소
                    if "X-Usage-Remaining" not in resp.headers:
                        new_value = max(0, st.session_state.get("usage_remaining", _usage_limit_int()) - 1)
                        st.session_state["usage_remaining"] = new_value
                st.rerun()
            except Exception as exc:  # pragma: no cover - UI 예외
                st.error(f"요청 실패: {exc}")


if __name__ == "__main__":
    main()
