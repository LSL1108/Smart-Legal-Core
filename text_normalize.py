from __future__ import annotations

import logging
import unicodedata
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


try:
    from opencc import OpenCC  # type: ignore
    _opencc_s2t = OpenCC("s2t")                 
    _opencc_t2s = OpenCC("t2s")                 
    try:
        _opencc_s2twp = OpenCC("s2twp")         
    except Exception:
        _opencc_s2twp = _opencc_s2t              
    _OPENCC_AVAILABLE = True
except Exception as exc:
    logger.info("opencc 未安裝，簡繁正規化將略過（建議 pip install opencc-python-reimplemented）: %s", exc)
    _opencc_s2t = None
    _opencc_t2s = None
    _opencc_s2twp = None
    _OPENCC_AVAILABLE = False


def normalize_for_match(text: str) -> str:
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text)
    s = s.lower()

    if _OPENCC_AVAILABLE and _opencc_s2t:
        try:
            s = _opencc_s2t.convert(s)
        except Exception as exc:
            logger.warning("opencc 轉換失敗，使用原文比對: %s", exc)

    s = " ".join(s.split())
    return s


def find_all_occurrences(haystack: str, needle: str) -> List[Tuple[int, int]]:
    if not haystack or not needle:
        return []
    out: List[Tuple[int, int]] = []
    needle_len = len(needle)
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        out.append((idx, idx + needle_len))
        start = idx + 1  # 允許重疊命中（如「合作合作」）
    return out


def find_any_occurrence(haystack_norm: str, needle: str) -> Optional[Tuple[int, int]]:
    if not haystack_norm or not needle:
        return None
    idx = haystack_norm.find(needle)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


def is_normalize_available() -> bool:
    return True


def is_opencc_available() -> bool:

    return _OPENCC_AVAILABLE


def ensure_traditional(text: str) -> str:
    if not text or not _OPENCC_AVAILABLE:
        return text
    converter = _opencc_s2twp or _opencc_s2t
    if converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception as exc:
        logger.warning("ensure_traditional 轉換失敗，回傳原文：%s", exc)
        return text


def ensure_traditional_in_obj(obj):

    if obj is None:
        return obj
    if isinstance(obj, str):
        return ensure_traditional(obj)
    if isinstance(obj, dict):
        return {k: ensure_traditional_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_traditional_in_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(ensure_traditional_in_obj(v) for v in obj)
    return obj



def expand_aliases_for_match(aliases: List[str]) -> List[Tuple[str, str]]:

    out: List[Tuple[str, str]] = []
    seen_norm: set = set()

    for raw in aliases:
        if not raw:
            continue
        norm = normalize_for_match(raw)
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            out.append((raw, norm))
        if _OPENCC_AVAILABLE:
            try:
                # 原字串若是繁體，加上其簡體形式
                simp = _opencc_t2s.convert(raw) if _opencc_t2s else raw
                simp_norm = normalize_for_match(simp)
                if simp_norm and simp_norm not in seen_norm:
                    seen_norm.add(simp_norm)
                    out.append((raw, simp_norm))
            except Exception:
                pass

    return out
