# Compare-AI (Frontend: Streamlit)

백엔드(`compare-ai-be`)의 Streamlit UI만 별도 레포로 분리한 버전입니다.  
> **최종 업데이트: 2025-11-28** — 로그인/회원가입, 사용량 헤더 동기화, 대화 로그 유지

## 📋 프로젝트 개요

Supabase Auth로 로그인/회원가입 → JWT로 FastAPI 백엔드 `/api/ask` 호출 → 스트리밍 응답을 화면에 표시하는 경량 FE입니다.

## 🚀 빠른 시작 (로컬)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows는 .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app/ui/streamlit_app.py
```

## ⚙️ 필수 환경변수

`.env` 또는 Streamlit secrets에 설정합니다.

```
# 백엔드 주소
FASTAPI_URL=https://your-backend.onrender.com

# 인증 없이 우회하려면(선택)
ADMIN_BYPASS_TOKEN=...

# 선택: 일일 사용량 기본값 (백엔드가 헤더로 내려주면 덮어씀)
DAILY_USAGE_LIMIT=3
```

## 🖥️ 사용 방법

1) 사이드바에서 FastAPI Base URL 입력 (예: `https://.../api/ask` 없이 베이스 주소)  
2) 이메일/비밀번호로 회원가입 또는 로그인 → 토큰 자동 저장  
3) 질문 입력 후 “질문하기” 클릭 → 모델별 partial 이벤트와 summary 표시  
4) 응답 헤더 `X-Usage-Limit`/`X-Usage-Remaining`을 읽어 남은 횟수 동기화  
5) 필요하면 `x-admin-bypass` 토글 후 관리자 토큰으로 우회 가능

## 📁 프로젝트 구조

```
app/
├── __init__.py
└── ui/
    ├── README.md
    └── streamlit_app.py  # Streamlit 엔트리포인트
docs/                     # 백엔드 문서와 동기화된 FE 문서
requirements.txt          # Streamlit UI 전용 최소 의존성
runtime.txt               # Python 버전 고정(선택)
```

## ✏️ 문서

- 변경 로그: `docs/changelog/`
- 개발 로그/로드맵: `docs/development/`

## 🧪 체크리스트

- FastAPI 백엔드가 실행 중인지 확인
- `.env`에 `FASTAPI_URL`을 넣어두면 자동 로드
- 헤더로 내려오는 사용량 정보가 UI에 반영되는지 확인
