import logging
import os
import time
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
        # 進一步檢查 message 是否包含 timeout 字樣，避免誤判其他 client 錯誤
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
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

    重要設計變更：
    - 不再用 ThreadPoolExecutor.future.result(timeout=...) 假 timeout。
      該方式無法 kill 已啟動的 worker thread，只是讓主執行緒提前回傳，
      Ollama 那端的呼叫仍會繼續吃資源並阻塞下一個請求。
    - 真正的 timeout 已下放到 ollama Client 層 (services._ollama_client)，
      由 LLM_TIMEOUT_SEC 控制。client 在 HTTP 層丟 timeout exception 時
      這裡只負責認得並轉成統一的回應格式。
    """
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

    base_extra = {
        "used_review_context": should_use_review_context,
        "review_context_available": used_review_context,
        "detected_intent": detected_intent,
        "has_draft": has_draft,
        "latest_user_input": latest_user_input[:200],
    }

    if should_use_review_context:
        try:
            clause_result = answer_clause_followup_with_meta(
                user_input=latest_user_input,
                review_context=review_context or {},
            )

            if clause_result:
                extra = clause_result.get("extra") or {}
                extra.update(base_extra)

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

    try:
        # 不再包 ThreadPoolExecutor — timeout 由 ollama Client 真正 enforce。
        reply = llm_chat(
            messages=messages,
            draft_text=draft_text,
            review_context=review_context if should_use_review_context else None,
        )

        return {
            "reply": reply,
            "tool_name": fallback_tool_name,
            "latency_sec": time.perf_counter() - t0,
            "extra": base_extra,
        }

    except Exception as exc:
        if _is_timeout_exc(exc):
            logger.warning("llm_chat timed out at client layer: %s", exc)

            timeout_extra = dict(base_extra)
            timeout_extra["timed_out"] = True

            return {
                "reply": _TIMEOUT_MSG,
                "tool_name": "backend_timeout",
                "latency_sec": time.perf_counter() - t0,
                "extra": timeout_extra,
            }

        logger.exception("answer_contract_chat failed")

        error_extra = dict(base_extra)
        error_extra["error"] = str(exc)

        return {
            "reply": f"對話服務發生錯誤：{exc}",
            "tool_name": "chat_error",
            "latency_sec": time.perf_counter() - t0,
            "extra": error_extra,
        }
