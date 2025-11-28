"""프로젝트 전역에서 재사용할 이모지 기반 로거 유틸리티."""

from __future__ import annotations

import logging

LEVEL_EMOJI: dict[int, str] = {
    logging.DEBUG: "🛠️",
    logging.INFO: "✅",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}


class EmojiFormatter(logging.Formatter):
    """로그 레코드에 로그 레벨에 따른 이모지를 추가한다."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        record.emoji = LEVEL_EMOJI.get(record.levelno, "")  # type: ignore[attr-defined]
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """표준 출력으로 기록하는 프로젝트 전용 로거를 반환한다."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = EmojiFormatter(
            "%(emoji)s [%(levelname)s | %(asctime)s | %(name)s] - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
