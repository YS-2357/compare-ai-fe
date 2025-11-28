"""애플리케이션 설정 모듈."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """환경 변수를 통해 주입되는 기본 설정."""

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000

    env: Literal["local", "test", "prod"] = "local"
    langsmith_project: str = "Compare-AI-FE"
    supabase_url: str | None = None
    supabase_jwks_url: str | None = None
    supabase_aud: str = "authenticated"
    admin_bypass_token: str | None = None
    upstash_redis_url: str | None = None
    upstash_redis_token: str | None = None
    daily_usage_limit: int = 100

    @staticmethod
    def from_env() -> Settings:
        """환경 변수와 `.env` 값을 기반으로 Settings를 생성한다.

        Returns:
            Settings: 현재 실행 환경에 맞춘 설정 인스턴스.
        """

        return Settings(
            fastapi_host=os.getenv("FASTAPI_HOST", "127.0.0.1"),
            fastapi_port=int(os.getenv("FASTAPI_PORT", "8000")),
            env=os.getenv("APP_ENV", "local"),  # type: ignore[assignment]
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "API-LangGraph-Test"),
            supabase_url=os.getenv("SUPABASE_URL"),
            supabase_jwks_url=os.getenv("SUPABASE_JWKS_URL"),
            supabase_aud=os.getenv("SUPABASE_JWT_AUD", "authenticated"),
            admin_bypass_token=os.getenv("ADMIN_BYPASS_TOKEN"),
            upstash_redis_url=os.getenv("UPSTASH_REDIS_URL"),
            upstash_redis_token=os.getenv("UPSTASH_REDIS_TOKEN"),
            daily_usage_limit=int(os.getenv("DAILY_USAGE_LIMIT", "100")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """전역적으로 재사용 가능한 Settings 인스턴스를 반환한다.

    Returns:
        Settings: 캐싱된 설정 인스턴스.
    """

    return Settings.from_env()
