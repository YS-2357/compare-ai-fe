"""FastAPI 애플리케이션 팩토리."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI 애플리케이션을 구성해 반환한다.

    Args:
        settings: 외부에서 주입할 `Settings` 인스턴스. 생략 시 `.env`/환경변수를 읽어 생성한다.

    Returns:
        FastAPI: 라우터와 미들웨어가 등록된 FastAPI 인스턴스.
    """

    settings = settings or get_settings()
    app = FastAPI(title="API LangGraph Test", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.state.settings = settings
    return app


app = create_app()
