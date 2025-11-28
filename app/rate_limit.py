"""Upstash Redis를 이용한 사용자별 일일 사용량 제한."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.config import get_settings


class UpstashClient:
    """간단한 Upstash REST 클라이언트."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self._client = httpx.AsyncClient(timeout=5)
        self._lock = asyncio.Lock()

    async def incr_with_expiry(self, key: str, ttl_seconds: int) -> int:
        """INCR 후 키에 만료를 설정한다."""

        payload: list[list[Any]] = [["INCR", key], ["EXPIRE", key, ttl_seconds]]
        async with self._lock:
            resp = await self._client.post(
                f"{self.url}/pipeline",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )
        if resp.status_code >= 400:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limit backend error")

        data = resp.json()
        # Pipeline 결과는 [{"result": int}, {"result": int}] 형태
        try:
            return int(data[0]["result"])
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limit parse error") from exc


_client: UpstashClient | None = None


def get_rate_limiter() -> UpstashClient:
    """환경변수를 기반으로 Upstash 클라이언트를 생성/재사용한다."""

    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.upstash_redis_url or not settings.upstash_redis_token:
        raise RuntimeError("Upstash Redis 설정이 없습니다.")
    _client = UpstashClient(settings.upstash_redis_url, settings.upstash_redis_token)
    return _client


def _seconds_until_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow - now).total_seconds())


async def enforce_daily_limit(user_id: str, limit: int) -> None:
    """사용자별 일일 호출 제한을 적용한다."""

    client = get_rate_limiter()
    key = f"usage:{user_id}:{datetime.now(timezone.utc).date().isoformat()}"
    count = await client.incr_with_expiry(key, _seconds_until_midnight_utc())
    if count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="daily usage limit exceeded",
        )
