# app/ui/

- Streamlit 로컬/개발용 UI (별도 Render 서비스로도 사용 가능).
- 로그인/회원가입 → 질문 → 스트림(partial/summary) 표시, 모델별 답변과 source 표시.
- 사용량 헤더(`X-Usage-Limit`, `X-Usage-Remaining`)를 읽어 카운터를 동기화.

실행 (로컬):
```bash
streamlit run app/ui/streamlit_app.py
```

환경변수:
- `FASTAPI_URL` (예: https://<render-be>.onrender.com)
- `ADMIN_BYPASS_TOKEN` (옵션)
- `STREAMLIT_SERVER_HEADLESS=true` (Render 등 배포 시)

