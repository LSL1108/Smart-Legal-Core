from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---- Optional dependency guard ----
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    logger.info("sentence-transformers 未安裝，reranker 將自動停用: %s", exc)
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None  # type: ignore


_model_singleton: Optional[Any] = None
_model_init_lock = threading.Lock()
_model_load_failed = False  

_consecutive_timeouts: int = 0
_consecutive_timeouts_lock = threading.Lock()
_AUTO_DISABLE_AFTER_TIMEOUTS = 3
_runtime_disabled: bool = False

_predict_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")


def is_available() -> bool:

    return _SENTENCE_TRANSFORMERS_AVAILABLE and not _model_load_failed and not _runtime_disabled


def _record_timeout() -> None:
    global _runtime_disabled, _consecutive_timeouts
    with _consecutive_timeouts_lock:
        _consecutive_timeouts += 1
        if _consecutive_timeouts >= _AUTO_DISABLE_AFTER_TIMEOUTS and not _runtime_disabled:
            _runtime_disabled = True
            logger.error(
                "Reranker 連續 timeout 達 %d 次，自動禁用整層以避免空轉。"
                "排查方向：(1) device 是否為 CPU？(2) 降低 RERANKER_INPUT_K；"
                "(3) 設環境變數 ENABLE_RERANKER=false 永久關閉",
                _consecutive_timeouts,
            )


def _record_success() -> None:
    global _consecutive_timeouts
    with _consecutive_timeouts_lock:
        _consecutive_timeouts = 0


def _detect_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        # macOS Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_model(model_name: str, max_length: int = 1024) -> Optional[Any]:
    global _model_singleton, _model_load_failed

    if _model_load_failed:
        return None
    if _model_singleton is not None:
        return _model_singleton
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None

    with _model_init_lock:
        if _model_load_failed:
            return None
        if _model_singleton is not None:
            return _model_singleton

        try:
            device = _detect_device()
            if device == "cpu":
                logger.warning(
                    "Reranker 載入於 CPU。重排 ~20 對候選將花 2~5 秒，長合約上有感。"
                    "建議：(a) 在 macOS Apple Silicon 安裝支援 MPS 的 torch；"
                    "(b) 降低 RERANKER_INPUT_K（如 8）；"
                    "(c) 不需要時設環境變數 ENABLE_RERANKER=false 關掉"
                )
            logger.info("載入 reranker 模型 %s 於 device=%s（首次載入會花幾秒）...", model_name, device)
            t0 = time.perf_counter()
            _model_singleton = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device,
            )
            logger.info("Reranker 模型載入完成，耗時 %.2fs", time.perf_counter() - t0)
            return _model_singleton
        except Exception as exc:
            logger.error("Reranker 模型載入失敗，後續將跳過重排：%s", exc)
            _model_load_failed = True
            return None


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit]


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    enabled: bool = True,
    model_name: Optional[str] = None,
    max_chars: Optional[int] = None,
    timeout_sec: Optional[float] = None,
) -> List[Dict[str, Any]]:

    from config import (
        ENABLE_RERANKER,
        RERANKER_MODEL,
        RERANKER_MAX_CHARS,
        RERANKER_TIMEOUT_SEC,
    )

    effective_enabled = enabled and ENABLE_RERANKER and not _runtime_disabled
    effective_model = model_name or RERANKER_MODEL
    effective_max_chars = max_chars or RERANKER_MAX_CHARS
    effective_timeout = timeout_sec or RERANKER_TIMEOUT_SEC

    if not effective_enabled:
        return candidates if top_k is None else candidates[:top_k]

    if not candidates or not query:
        return candidates if top_k is None else candidates[:top_k]

    # 單一候選不需要重排
    if len(candidates) == 1:
        return list(candidates)

    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return candidates if top_k is None else candidates[:top_k]

    model = _get_model(effective_model)
    if model is None:
        return candidates if top_k is None else candidates[:top_k]

    # 準備 (query, content) pairs，截到 max_chars
    short_query = _truncate(query, effective_max_chars)
    pairs: List[Tuple[str, str]] = []
    for c in candidates:
        content = str(c.get("content", "") or "")
        pairs.append((short_query, _truncate(content, effective_max_chars)))


    def _do_predict() -> List[float]:
        scores = model.predict(pairs, show_progress_bar=False)
        try:
            return [float(s) for s in scores]
        except Exception:
            return [float(s) for s in list(scores)]

    try:
        t0 = time.perf_counter()
        future = _predict_executor.submit(_do_predict)
        scores = future.result(timeout=effective_timeout)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "rerank: %d 對候選，耗時 %.3fs，topk=%s",
            len(pairs), elapsed, top_k,
        )
        _record_success()
    except FuturesTimeoutError:
        _record_timeout()
        logger.warning(
            "Reranker timeout (>%.1fs)，回退到原排序；建議檢查模型 device 或減少 RERANKER_INPUT_K",
            effective_timeout,
        )
        return candidates if top_k is None else candidates[:top_k]
    except Exception as exc:
        logger.warning("Reranker 預測失敗，回退到原排序：%s", exc)
        return candidates if top_k is None else candidates[:top_k]

    enriched: List[Tuple[float, Dict[str, Any]]] = []
    for c, s in zip(candidates, scores):
        c_copy = dict(c)
        c_copy["_rerank_score"] = s
        enriched.append((s, c_copy))

    enriched.sort(key=lambda x: x[0], reverse=True)
    sorted_cands = [c for _, c in enriched]

    if top_k is None:
        return sorted_cands
    return sorted_cands[:top_k]


def warmup() -> None:
    from config import ENABLE_RERANKER, RERANKER_MODEL
    if not ENABLE_RERANKER:
        return
    _get_model(RERANKER_MODEL)
