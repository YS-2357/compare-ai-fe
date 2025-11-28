# app/ui/

- Streamlit 로컬/개발용 UI (별도 Render 서비스로도 사용 가능).
- 로그인/회원가입 → 질문 → 스트림(partial/summary) 표시, 모델별 답변과 source 표시.
- 사용량 헤더(`X-Usage-Limit`, `X-Usage-Remaining`)를 읽어 카운터를 동기화.
- 현재 한계: 재로그인 직후 초기 화면에서 남은 횟수가 항상 3으로 보일 수 있으며, `/api/ask` 호출 후 응답 헤더/summary로 실제 값이 반영된다(초기 표시를 서버 값으로 맞추는 패치 예정).

실행 (로컬):
```bash
streamlit run app/ui/streamlit_app.py
```

환경변수:
- `FASTAPI_URL` (예: https://<render-be>.onrender.com)
- `ADMIN_BYPASS_TOKEN` (옵션, 입력값이 환경/secret의 토큰과 일치할 때만 우회 적용)
- `STREAMLIT_SERVER_HEADLESS=true` (Render 등 배포 시)
