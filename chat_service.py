import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from services import llm_chat
from clause_followup_service import answer_clause_followup_with_meta
from intent_detector import detect_intent

logger = logging.getLogger(__name__)

_TIMEOUT_MSG = "後端分析逾時，可能是模型回應卡住。請縮小問題範圍後再試一次。"


def _is_timeout_exc(exc: BaseException) -> bool:
    type_name = type(exc).__name__
    if "Timeout" in type_name or "timeout" in type_name:
        return True

    module = getattr(type(exc), "__module__", "") or ""
    if any(m in module for m in ("ollama", "httpx", "requests", "urllib")):
        return True

    return False


def _get_latest_user_input(messages: List[Dict[str, str]]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "") or "").strip()
    return ""


def _infer_fallback_tool_name(
    *,
    draft_text: str = "",
    should_use_review_context: bool = False,
) -> str:
    if should_use_review_context:
        return "review_context_chat"

    if draft_text and draft_text.strip():
        return "contract_draft_chat"

    return "general_chat"


def answer_contract_chat(
    *,
    messages: List[Dict[str, str]],
    draft_text: str = "",
    review_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    合約聊天服務統一入口。

    回傳格式：
    {
        "reply": "...",
        "tool_name": "general_chat | contract_draft_chat | review_context_chat | clause_followup | backend_timeout | chat_error",
        "latency_sec": 1.23,
        "extra": {...}
    }

    設計重點：
    1. 補條款問題優先由 clause_followup_service deterministic 處理。
    2. 其他聊天才交給 llm_chat。
    3. 統一提供 timeout、latency、tool_name、extra。
    """
    timeout_sec = float(os.getenv("CHAT_TIMEOUT_SEC", "120").strip() or "120")
    t0 = time.perf_counter()

    latest_user_input = _get_latest_user_input(messages)
    detected_intent = detect_intent(latest_user_input)

    used_review_context = bool(review_context)
    has_draft = bool(draft_text and draft_text.strip())
    should_use_review_context = used_review_context and detected_intent in {
        "contract_review",
        "law_check",
        "historical_compare",
    }

    if should_use_review_context:
        try:
            clause_result = answer_clause_followup_with_meta(
                user_input=latest_user_input,
                review_context=review_context or {},
            )

            if clause_result:
                extra = clause_result.get("extra") or {}
                extra.update(
                    {
                        "used_review_context": should_use_review_context,
                        "review_context_available": used_review_context,
                        "detected_intent": detected_intent,
                        "has_draft": has_draft,
                        "latest_user_input": latest_user_input[:200],
                    }
                )

                return {
                    "reply": clause_result.get("reply", ""),
                    "tool_name": clause_result.get("tool_name", "clause_followup"),
                    "latency_sec": time.perf_counter() - t0,
                    "extra": extra,
                }

        except Exception as exc:
            logger.exception("clause_followup_service failed, fallback to llm_chat: %s", exc)

    fallback_tool_name = _infer_fallback_tool_name(
        draft_text=draft_text,
        should_use_review_context=should_use_review_context,
    )

    def _run():
        return llm_chat(
            messages=messages,
            draft_text=draft_text,
            review_context=review_context if should_use_review_context else None,
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)

            try:
                reply = future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                future.cancel()
                logger.warning("llm_chat timed out after %.1fs", timeout_sec)

                return {
                    "reply": _TIMEOUT_MSG,
                    "tool_name": "backend_timeout",
                    "latency_sec": time.perf_counter() - t0,
                    "extra": {
                        "timed_out": True,
                        "timeout_sec": timeout_sec,
                        "used_review_context": should_use_review_context,
                        "review_context_available": used_review_context,
                        "detected_intent": detected_intent,
                        "has_draft": has_draft,
                        "latest_user_input": latest_user_input[:200],
                    },
                }

        return {
            "reply": reply,
            "tool_name": fallback_tool_name,
            "latency_sec": time.perf_counter() - t0,
            "extra": {
                "used_review_context": should_use_review_context,
                "review_context_available": used_review_context,
                "detected_intent": detected_intent,
                "has_draft": has_draft,
                "latest_user_input": latest_user_input[:200],
            },
        }

    except Exception as exc:
        if _is_timeout_exc(exc):
            logger.warning("llm_chat raised timeout exception: %s", exc)

            return {
                "reply": _TIMEOUT_MSG,
                "tool_name": "backend_timeout",
                "latency_sec": time.perf_counter() - t0,
                "extra": {
                    "timed_out": True,
                    "timeout_sec": timeout_sec,
                    "used_review_context": should_use_review_context,
                    "review_context_available": used_review_context,
                    "detected_intent": detected_intent,
                    "has_draft": has_draft,
                    "latest_user_input": latest_user_input[:200],
                },
            }

        logger.exception("answer_contract_chat failed")

        return {
            "reply": f"對話服務發生錯誤：{exc}",
            "tool_name": "chat_error",
            "latency_sec": time.perf_counter() - t0,
            "extra": {
                "error": str(exc),
                "used_review_context": should_use_review_context,
                "review_context_available": used_review_context,
                "detected_intent": detected_intent,
                "has_draft": has_draft,
                "latest_user_input": latest_user_input[:200],
            },
        }