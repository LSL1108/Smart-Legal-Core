from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---- Optional dependencies guard ----
try:
    import jieba  
    jieba.setLogLevel(logging.WARNING)
    _JIEBA_AVAILABLE = True
except Exception as exc:  
    logger.warning("jieba not available, BM25 retrieval will be disabled: %s", exc)
    _JIEBA_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi  
    _BM25_AVAILABLE = True
except Exception as exc:  
    logger.warning("rank_bm25 not available, BM25 retrieval will be disabled: %s", exc)
    _BM25_AVAILABLE = False


HYBRID_RETRIEVAL_AVAILABLE = _JIEBA_AVAILABLE and _BM25_AVAILABLE


# ---- Tokenization ----

# 法律 / 合約專用詞彙，加進 jieba 字典後切詞會更準確。
_CUSTOM_TERMS = [
    "資安檢測", "弱點掃描", "滲透測試", "弱掃", "資安事件",
    "管轄法院", "準據法", "智慧財產權", "違約金", "懲罰性違約金",
    "保密義務", "個資保護", "個人資料保護", "個人資料保護法",
    "保險代理人", "保險業務員", "保險商品", "保戶", "要保人", "被保險人",
    "佣酬", "佣酬返還", "招攬", "招攬義務", "理賠協助",
    "廣告文宣", "洗錢防制", "打擊資恐", "複委託", "利益衝突", "績效考核",
    "法規遵循", "合規", "稽核", "金管會", "主管機關",
    "備份", "災難復原", "事件通報", "弱點修補",
    "驗收", "交付", "上線", "測試版",
    "維護時間", "維護人力", "回覆時限", "修復時限",
    "原始碼", "程式碼", "開源", "GitHub",
]

_jieba_initialized = False
_jieba_init_lock = threading.Lock()


def _ensure_jieba_terms() -> None:
    global _jieba_initialized
    if _jieba_initialized or not _JIEBA_AVAILABLE:
        return
    with _jieba_init_lock:
        if _jieba_initialized:
            return
        for term in _CUSTOM_TERMS:
            jieba.add_word(term)
        _jieba_initialized = True


# 通用停用字，避免污染 BM25 排序
_STOPWORDS = {
    "的", "了", "是", "在", "及", "與", "或", "和", "之", "其", "並",
    "由", "為", "以", "對", "於", "至", "從", "給", "向", "等",
    "甲方", "乙方", "丙方", "雙方", "本條", "本合約", "本契約",
    "應", "得", "可", "需", "須", "予以", "予", "本", "前", "後",
    "第", "條", "款", "項", "目", "章", "節",
    " ", "\n", "\t", "，", "。", "、", "（", "）", "「", "」", "：", "；",
}


def tokenize_zh(text: str) -> List[str]:
    if not text or not _JIEBA_AVAILABLE:
        return []
    _ensure_jieba_terms()
    tokens = jieba.cut_for_search(text)
    return [t.strip() for t in tokens if t.strip() and t not in _STOPWORDS and len(t) >= 1]


# ---- Reciprocal Rank Fusion ----

def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:

    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---- HybridIndex ----

class HybridIndex:

    def __init__(self) -> None:
        self._chunks: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[Any] = None
        self._dirty: bool = True
        self._lock = threading.RLock()

    # --- Lifecycle ---

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def build(self, chunks: Iterable[Dict[str, Any]]) -> None:

        if not HYBRID_RETRIEVAL_AVAILABLE:
            with self._lock:
                self._chunks = list(chunks)
                self._tokenized = []
                self._bm25 = None
                self._dirty = False
            return

        with self._lock:
            self._chunks = []
            self._tokenized = []
            for c in chunks:
                content = str((c.get("content") or "")).strip()
                if not content:
                    continue
                self._chunks.append(c)
                self._tokenized.append(tokenize_zh(content))

            if self._tokenized:
                self._bm25 = BM25Okapi(self._tokenized)
            else:
                self._bm25 = None
            self._dirty = False

    def is_empty(self) -> bool:
        with self._lock:
            return not self._chunks

    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    # --- Search ---

    def bm25_search(
        self,
        query: str,
        top_k: int = 30,
        candidate_doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not HYBRID_RETRIEVAL_AVAILABLE:
            return []

        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []

            tokens = tokenize_zh(query)
            if not tokens:
                return []

            scores = self._bm25.get_scores(tokens)
            # 過濾候選文件
            allow = set(candidate_doc_ids) if candidate_doc_ids else None

            indexed = []
            for i, score in enumerate(scores):
                if score <= 0:
                    continue
                chunk = self._chunks[i]
                if allow is not None:
                    meta = chunk.get("metadata", {}) or {}
                    if str(meta.get("doc_id", "")) not in allow:
                        continue
                indexed.append((i, score))

            indexed.sort(key=lambda x: x[1], reverse=True)
            return [(self._chunks[i], float(s)) for i, s in indexed[:top_k]]

    # --- Helpers ---

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for c in self._chunks:
                if c.get("id") == chunk_id:
                    return c
        return None


# ---- Module-level singleton ----

_index_singleton: Optional[HybridIndex] = None
_index_singleton_lock = threading.Lock()


def get_hybrid_index() -> HybridIndex:
    global _index_singleton
    if _index_singleton is None:
        with _index_singleton_lock:
            if _index_singleton is None:
                _index_singleton = HybridIndex()
    return _index_singleton
