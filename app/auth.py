"""Supabase JWT 검증 및 관리자 바이패스 지원."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx
from fastapi import Depends, HTTPException, Header, status
from jose import JWTError, jwt

from app.config import get_settings


class AuthenticatedUser(TypedDict):
    """검증된 사용자 정보를 표현한다."""

    sub: str
    email: str | None
    role: str | None
    bypass: bool


class _JWKSCache:
    """Supabase JWKS를 주기적으로 캐시한다."""

    def __init__(self, jwks_url: str, cache_ttl: int = 300) -> None:
        self.jwks_url = jwks_url
        self.cache_ttl = cache_ttl
        self._keys: dict[str, dict[str, Any]] | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get_key(self, kid: str) -> dict[str, Any]:
        async with self._lock:
            if self._keys is None or self._expires_at <= asyncio.get_event_loop().time():
                await self._refresh()
            if not self._keys or kid not in self._keys:
                raise KeyError("JWKS key not found")
            return self._keys[kid]

    async def _refresh(self) -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(self.jwks_url)
            resp.raise_for_status()
            data = resp.json()
            keys = {item["kid"]: item for item in data.get("keys", []) if "kid" in item}
            self._keys = keys
            self._expires_at = asyncio.get_event_loop().time() + self.cache_ttl


@dataclass(frozen=True)
class SupabaseVerifier:
    """Supabase JWT 검증기."""

    jwks_cache: _JWKSCache
    audience: str
    issuer: str

    async def verify(self, token: str) -> AuthenticatedUser:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if not kid:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token header")
            key = await self.jwks_cache.get_key(kid)
            claims = jwt.decode(
                token,
                key,
                algorithms=[key.get("alg", "RS256")],
                audience=self.audience,
                issuer=self.issuer,
            )
        except (JWTError, KeyError) as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token") from exc

        return AuthenticatedUser(
            sub=str(claims.get("sub")),
            email=claims.get("email"),
            role=claims.get("role"),
            bypass=False,
        )


_verifier: SupabaseVerifier | None = None


def _get_verifier() -> SupabaseVerifier:
    global _verifier
    if _verifier is not None:
        return _verifier

    settings = get_settings()
    jwks_url = settings.supabase_jwks_url
    if not jwks_url:
        if not settings.supabase_url:
            raise RuntimeError("Supabase URL이 설정되지 않았습니다.")
        jwks_url = settings.supabase_url.rstrip("/") + "/auth/v1/jwks"
    issuer = (settings.supabase_url or "").rstrip("/") + "/auth/v1"
    _verifier = SupabaseVerifier(_JWKSCache(jwks_url), audience=settings.supabase_aud, issuer=issuer)
    return _verifier


async def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    admin_bypass: str | None = Header(default=None, alias="x-admin-bypass"),
) -> AuthenticatedUser:
    """JWT를 검증하고 사용자 정보를 반환한다. 관리자 바이패스를 지원한다."""

    settings = get_settings()

    # 관리자 바이패스 토큰 우선
    if settings.admin_bypass_token and admin_bypass and admin_bypass == settings.admin_bypass_token:
        return AuthenticatedUser(sub="admin-bypass", email=None, role="admin", bypass=True)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authorization header missing")

    token = authorization.split(" ", 1)[1].strip()
    verifier = _get_verifier()
    return await verifier.verify(token)
