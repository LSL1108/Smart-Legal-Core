import os
import sqlite3
import json
import logging
import datetime
import hashlib
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple
import re
from difflib import SequenceMatcher
import chromadb
from chromadb.utils import embedding_functions
from utils import detect_topics, parse_core_topics_field, detect_clause_type, normalize_text
from config import SQLITE_DB_PATH, CHROMA_DIR, EMBED_MODEL, TOPIC_KEYWORDS, TOPIC_ALIAS

# ★ 預設法遵規則庫 
DEFAULT_COMPLIANCE_RULES: Dict[str, List[str]] = {
    "資安檢測與掃描": [
        "廠商應協助進行弱點掃描",
        "廠商應配合甲方進行系統資安掃描",
        "廠商應提供資安健檢報告",
    ],
    "個資保護與保密": [
        "廠商應遵循個人資料保護法",
        "廠商應簽署保密協議",
        "廠商人員不得洩漏甲方資料",
    ],
    "備份與災難復原": [
        "廠商應提供資料備份機制",
        "廠商應定期進行備份復原演練",
    ],
    "事件通報與應變": [
        "廠商應於發現資安事件後24小時內通報甲方",
        "廠商應提供事件應變支援",
    ],
    "弱點修補與維護": [
        "廠商應於規定期限內完成弱點修補",
        "廠商應提供免費修補服務",
    ],
    "代理人資格": [
        "保險代理人應具備保險代理人執業證照及相關法定資格",
        "從事招攬行為之人員應具備保險業務員資格",
    ],
    "授權範圍": [
        "保險代理人之授權事項應以書面明確約定，不得以口頭同意擴張服務範圍",
    ],
    "文件轉送期限": [
        "保險代理人收受要保文件後，應於約定期限內轉送保險業核辦",
    ],
    "廣告文宣控管": [
        "保險商品廣告、文宣、簡報及商品說明資料應經保險業事前書面同意後始得使用",
    ],
    "個人資料保護": [
        "保險代理人處理保戶、要保人或被保險人個人資料時，應採取安全維護措施並接受保險業監督",
    ],
    "複委託監督": [
        "保險代理人複委託第三人處理資料或服務事項時，應取得保險業事前書面同意並負同等責任",
    ],
    "洗錢防制與打擊資恐": [
        "保險代理人應配合保險業辦理客戶身分確認、風險辨識、資料驗證及洗錢防制與打擊資恐作業",
    ],
    "終止事由": [
        "保險代理人有證照撤銷、主管機關重大裁罰、重大違反保險法令或損害保戶權益情事時，保險業得暫停或終止合約",
    ],
}
# === Topic Filter Helper Functions ===

_TOPIC_FILTER_MIN_MATCH = 1


def _topic_filter_terms(target_topic: str) -> List[str]:
    """取得 target_topic 對應的 topic、alias、keywords，用於 RAG chunk topic guard。"""
    target_topic = str(target_topic or "").strip()
    if not target_topic:
        return []

    terms: List[str] = [target_topic]
    terms.extend(TOPIC_ALIAS.get(target_topic, []))
    terms.extend(TOPIC_KEYWORDS.get(target_topic, []))

    seen = set()
    normalized_terms: List[str] = []
    for term in terms:
        t = normalize_text(str(term or "").strip())
        if not t or t in seen:
            continue
        seen.add(t)
        normalized_terms.append(t)
    return normalized_terms


def _ref_topic_text(ref: Dict[str, Any]) -> str:
    """彙整 chunk metadata 與內容中可判斷 topic 的文字。"""
    parts: List[str] = []
    for key in ["topics", "core_topics"]:
        value = ref.get(key, "")
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        else:
            parts.append(str(value or ""))

    for key in [
        "topics_text",
        "article_title",
        "clause_type",
        "chunk_label",
        "content",
    ]:
        parts.append(str(ref.get(key, "") or ""))

    return normalize_text(" ".join(parts))


def _topic_match_score(ref: Dict[str, Any], target_topic: str) -> int:
    """計算 chunk 是否與指定 topic 相符。metadata topic 命中給較高權重。"""
    terms = _topic_filter_terms(target_topic)
    if not terms:
        return 0

    score = 0
    topic_fields = normalize_text(" ".join([
        str(ref.get("topics_text", "") or ""),
        " ".join(ref.get("topics", []) if isinstance(ref.get("topics"), list) else [str(ref.get("topics", ""))]),
        " ".join(ref.get("core_topics", []) if isinstance(ref.get("core_topics"), list) else [str(ref.get("core_topics", ""))]),
        str(ref.get("article_title", "") or ""),
        str(ref.get("clause_type", "") or ""),
        str(ref.get("chunk_label", "") or ""),
    ]))
    full_text = _ref_topic_text(ref)

    for term in terms:
        if term in topic_fields:
            score += 3
        elif term in full_text:
            score += 1
    return score


def filter_chunks_by_topic(refs: List[Dict[str, Any]], target_topic: str) -> List[Dict[str, Any]]:
    """
    依 target_topic 優先篩選 RAG 片段。

    目的：避免「文件對了、條文錯了」，例如缺漏保密措施卻引用到招攬義務。
    若完全沒有命中，保留原 refs 作為 fallback，避免因 topic metadata 不完整而查不到資料。
    """
    target_topic = str(target_topic or "").strip()
    if not refs or not target_topic:
        return refs

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for ref in refs:
        score = _topic_match_score(ref, target_topic)
        if score >= _TOPIC_FILTER_MIN_MATCH:
            scored.append((score, ref))

    if not scored:
        return refs

    scored.sort(key=lambda item: item[0], reverse=True)
    return [ref for _, ref in scored]


# Connection

@contextmanager
def get_db():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Schema

def init_sqlite():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS templates (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id           TEXT UNIQUE NOT NULL,
                file_name        TEXT NOT NULL,
                file_type        TEXT,
                storage_path     TEXT,
                sha256           TEXT UNIQUE,
                byte_size        INTEGER,
                created_at       TEXT,
                contract_type    TEXT,
                summary          TEXT,
                keywords         TEXT,
                template_role    TEXT,
                core_topics      TEXT,
                source_text      TEXT,
                vendor_name      TEXT,
                system_name      TEXT,
                service_scope    TEXT,
                maintenance_type TEXT,
                industry         TEXT,
                contract_name    TEXT
            );

            CREATE TABLE IF NOT EXISTS contract_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor_name TEXT NOT NULL,
                amount      INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                username  TEXT NOT NULL,
                action    TEXT NOT NULL,
                target    TEXT,
                detail    TEXT
            );

            CREATE TABLE IF NOT EXISTS compliance_rules (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                topic      TEXT NOT NULL,
                example    TEXT NOT NULL,
                is_active  INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_compliance_rules_topic
                ON compliance_rules (topic, is_active);
        """)
    
    with get_db() as conn:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(templates)").fetchall()}
        for col_name in ["vendor_name", "system_name", "service_scope", "maintenance_type", "industry", "contract_name"]:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE templates ADD COLUMN {col_name} TEXT")

    seed_compliance_rules_if_empty()


# Helpers

def row_to_template_dict(row: Optional[sqlite3.Row]) -> Dict[str, Any]:
    if row is None:
        return {}
    d = dict(row)
    d["keywords"]    = json.loads(d["keywords"])    if d.get("keywords")    else []
    d["core_topics"] = json.loads(d["core_topics"]) if d.get("core_topics") else []
    return d


# Templates CRUD

def template_exists_by_sha256(sha256_val: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM templates WHERE sha256 = ?", (sha256_val,)
        ).fetchone()
        return row_to_template_dict(row) if row else None


def insert_template_doc(doc: Dict[str, Any]):
    created_at = doc.get("created_at")
    if isinstance(created_at, (datetime.datetime, datetime.date)):
        created_at = created_at.isoformat()
    else:
        created_at = str(created_at or "")

    with get_db() as conn:
        conn.execute("""
            INSERT INTO templates (
                doc_id, file_name, file_type, storage_path, sha256, byte_size, created_at,
                contract_type, summary, keywords, template_role, core_topics, source_text,
                vendor_name, system_name, service_scope, maintenance_type, industry, contract_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.get("doc_id"),
            doc.get("file_name"),
            doc.get("file_type"),
            doc.get("storage_path"),
            doc.get("sha256"),
            doc.get("byte_size"),
            created_at,
            doc.get("contract_type", "其他"),
            doc.get("summary", ""),
            json.dumps(doc.get("keywords",    []), ensure_ascii=False),
            doc.get("template_role", "歷史基準與規範"),
            json.dumps(doc.get("core_topics", []), ensure_ascii=False),
            doc.get("source_text", ""),
            doc.get("vendor_name", ""),
            doc.get("system_name", ""),
            doc.get("service_scope", ""),
            doc.get("maintenance_type", ""),
            doc.get("industry", ""),
            doc.get("contract_name", ""),
        ))


def get_template_by_doc_id(doc_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM templates WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row_to_template_dict(row) if row else None


def get_all_templates() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM templates ORDER BY created_at DESC"
        ).fetchall()
        return [row_to_template_dict(r) for r in rows]


def delete_template_by_doc_id(doc_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM templates WHERE doc_id = ?", (doc_id,))


def search_templates_sql(
    query_text: str = "",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 50,
    contract_type: str = "",
    query: str = "",
) -> List[Dict[str, Any]]:
    """
    支援新舊兩種呼叫方式：
    - 舊版：search_templates_sql(contract_type=..., query=..., limit=...)
    - 新版：search_templates_sql(query_text=..., filters={...}, limit=...)
    """
    effective_query = (query_text or query or "").strip()
    filters = dict(filters or {})

    if contract_type and not filters.get("contract_type"):
        filters["contract_type"] = contract_type

    where_clauses = ["file_type = 'docx'"]
    params: List[Any] = []

    field_map = {
        "contract_type": "contract_type",
        "vendor_name": "vendor_name",
        "system_name": "system_name",
        "service_scope": "service_scope",
        "maintenance_type": "maintenance_type",
        "industry": "industry",
        "contract_name": "contract_name",
    }

    for key, col in field_map.items():
        value = (filters.get(key) or "").strip() if isinstance(filters.get(key), str) else filters.get(key)
        if value:
            where_clauses.append(f"COALESCE({col}, '') LIKE ?")
            params.append(f"%{value}%")

    sql = "SELECT * FROM templates WHERE " + " AND ".join(where_clauses) + " ORDER BY created_at DESC LIMIT ?"

    with get_db() as conn:
        rows = conn.execute(sql, (*params, limit)).fetchall()

        if not rows:
            fallback_clauses = ["file_type = 'docx'"]
            fallback_params: List[Any] = []
            if filters.get("contract_type"):
                fallback_clauses.append("COALESCE(contract_type, '') LIKE ?")
                fallback_params.append(f"%{str(filters['contract_type']).strip()}%")

            fallback_sql = "SELECT * FROM templates WHERE " + " AND ".join(fallback_clauses) + " ORDER BY created_at DESC LIMIT ?"
            rows = conn.execute(fallback_sql, (*fallback_params, limit)).fetchall()

        if not rows:
            rows = conn.execute(
                "SELECT * FROM templates WHERE file_type = 'docx' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        docs = [row_to_template_dict(r) for r in rows]

    q = effective_query

    def _score(doc: Dict[str, Any]) -> int:
        blob_parts = [
            doc.get("file_name", "") or "",
            doc.get("summary", "") or "",
            " ".join(doc.get("keywords", [])),
            " ".join(doc.get("core_topics", [])),
            doc.get("vendor_name", "") or "",
            doc.get("system_name", "") or "",
            doc.get("service_scope", "") or "",
            doc.get("maintenance_type", "") or "",
            doc.get("industry", "") or "",
            doc.get("contract_name", "") or "",
        ]
        text = " ".join(blob_parts)
        score = sum(2 for token in q.split() if len(token) >= 2 and token in text)

        for k, field in [("vendor_name", 8), ("system_name", 7), ("service_scope", 6), ("maintenance_type", 5), ("contract_type", 5)]:
            fv = (filters.get(k) or "").strip() if isinstance(filters.get(k), str) else ""
            dv = (doc.get(k) or "").strip() if isinstance(doc.get(k), str) else ""
            if fv and dv and fv == dv:
                score += field
        return score

    return sorted(docs, key=_score, reverse=True)


def get_template_by_file_name_like(file_name: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute("""
            SELECT * FROM templates
            WHERE file_name LIKE ?
            ORDER BY created_at DESC LIMIT 1
        """, (f"%{file_name}%",)).fetchone()
        return row_to_template_dict(row) if row else None


def get_template_by_selector(selector: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if not selector:
        return None
    if selector.get("file_name"):
        return get_template_by_file_name_like(selector["file_name"])
    return None


# Contract History

def count_history_records() -> int:
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM contract_history").fetchone()
        return int(row["cnt"])


def insert_history_records(records: List[Dict[str, Any]]):
    with get_db() as conn:
        conn.executemany(
            "INSERT INTO contract_history (vendor_name, amount) VALUES (?, ?)",
            [(r["vendor_name"], r["amount"]) for r in records],
        )


def find_history_by_vendor_keyword(keyword: str) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM contract_history WHERE vendor_name LIKE ?",
            (f"%{keyword}%",),
        ).fetchall()
        return [dict(r) for r in rows]

# Audit Log

def insert_audit_log(username: str, action: str, target: str = "", detail: str = ""):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO audit_log (timestamp, username, action, target, detail)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            username or "未知使用者",
            action,
            target or "",
            detail or "",
        ))


def get_audit_logs(limit: int = 200) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ★ Compliance Rules CRUD

def get_compliance_rules() -> Dict[str, List[str]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT topic, example FROM compliance_rules "
            "WHERE is_active = 1 ORDER BY topic, id"
        ).fetchall()

    result: Dict[str, List[str]] = {}
    for topic, example in rows:
        result.setdefault(topic, []).append(example)
    return result


def insert_compliance_rule(topic: str, example: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO compliance_rules (topic, example) VALUES (?, ?)",
            (topic.strip(), example.strip()),
        )
        return cur.lastrowid


def upsert_compliance_rule(topic: str, examples: List[str]):
    """前端傳入一個主題與多個範例，更新該主題的所有規則（先軟刪除舊的再新增）。"""
    with get_db() as conn:
        conn.execute(
            "UPDATE compliance_rules SET is_active = 0, updated_at = datetime('now','localtime') WHERE topic = ?",
            (topic.strip(),)
        )
        for ex in examples:
            if ex.strip():
                conn.execute(
                    "INSERT INTO compliance_rules (topic, example) VALUES (?, ?)",
                    (topic.strip(), ex.strip())
                )


def delete_compliance_rule(topic: str) -> bool:
    """軟刪除：將該主題的所有規則標記為停用，保留歷史紀錄。"""
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE compliance_rules SET is_active = 0, "
            "updated_at = datetime('now','localtime') WHERE topic = ?",
            (topic.strip(),)
        )
        return cur.rowcount > 0


def list_all_compliance_rules() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, topic, example, is_active, created_at, updated_at "
            "FROM compliance_rules ORDER BY topic, id"
        ).fetchall()
        return [dict(r) for r in rows]


def seed_compliance_rules_if_empty():
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM compliance_rules"
        ).fetchone()[0]

    if count > 0:
        return

    with get_db() as conn:
        conn.executemany(
            "INSERT INTO compliance_rules (topic, example) VALUES (?, ?)",
            [
                (topic, example)
                for topic, examples in DEFAULT_COMPLIANCE_RULES.items()
                for example in examples
            ],
        )
    logging.info("✅ compliance_rules 初始化完成，已自動載入預設法遵規則。")


# ChromaDB

chroma_client     = None
template_collection = None
chunk_collection    = None
_db_initialized   = False


def get_chroma():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )

    t_col = client.get_or_create_collection(
        name="contract_templates_fulltext",
        embedding_function=ollama_ef,
    )
    c_col = client.get_or_create_collection(
        name="contract_template_chunks",
        embedding_function=ollama_ef,
    )
    return client, t_col, c_col


def init_db():
    global chroma_client, template_collection, chunk_collection, _db_initialized
    os.makedirs(os.path.dirname(SQLITE_DB_PATH) or ".", exist_ok=True)
    init_sqlite()
    chroma_client, template_collection, chunk_collection = get_chroma()
    _db_initialized = True


def ensure_db():
    global _db_initialized
    if not _db_initialized:
        init_db()


if not os.environ.get("SKIP_DB_INIT"):
    ensure_db()



# === Chunk Helper Functions ===

def _stable_chunk_id(doc_id: str, parent_article_key: str, chunk_index: int, content: str) -> str:
    """產生穩定 chunk id，避免同一文件重建索引時追蹤來源失準。"""
    seed = "\n".join([
        str(doc_id or ""),
        str(parent_article_key or ""),
        str(chunk_index),
        normalize_text(content or ""),
    ])
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
    return f"{doc_id}_chunk_{digest}"


def _metadata_str(value: Any) -> str:
    """Chroma metadata 僅存純量；此函式統一轉成安全字串。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _normalize_topics_value(value: Any, content: str = "") -> List[str]:
    if isinstance(value, list):
        topics = [str(t).strip() for t in value if str(t).strip()]
    else:
        topics = parse_core_topics_field(value)

    if not topics:
        topics = detect_topics(content)

    seen = set()
    normalized: List[str] = []
    for topic in topics:
        t = str(topic).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        normalized.append(t)
    return normalized


def _chunk_display_label(metadata: Dict[str, Any]) -> str:
    article_no = _metadata_str(metadata.get("article_no"))
    article_title = _metadata_str(metadata.get("article_title"))
    chunk_index = metadata.get("chunk_index", 0)

    parts = []
    if article_no:
        parts.append(article_no)
    if article_title:
        parts.append(article_title)
    if chunk_index not in [None, "", 0, "0"]:
        parts.append(f"片段{chunk_index}")
    return "｜".join(parts) if parts else "未標示條文"


# === Deduplication & MMR Helper Functions ===

_DEDUP_HIGH_SIMILARITY_THRESHOLD = 0.98


def _normalize_text_for_dedup(text: str) -> str:
    """供去重使用：壓縮空白與換行，避免同一段文字因格式差異被視為不同。"""
    return " ".join(normalize_text(text or "").split())


def _text_similarity(a: str, b: str) -> float:
    """兩段文字相似度，供高相似度去重與 MMR 使用。"""
    a = _normalize_text_for_dedup(a)
    b = _normalize_text_for_dedup(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _ref_dedup_key(ref: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """以文件、條文、片段與內容摘要作為去重基礎。"""
    return (
        str(ref.get("doc_id", "")),
        str(ref.get("parent_article_key", "")),
        str(ref.get("chunk_index", "")),
        _normalize_text_for_dedup(ref.get("content", ""))[:120],
    )


def _dedup_chunk_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    移除重複或幾乎相同的檢索片段。

    目的：避免同一條歷史合約因向量檢索與 keyword fallback 同時命中，
    造成後續 LLM 看到大量重複 context，進而放大單一來源的影響。
    """
    seen_keys = set()
    kept: List[Dict[str, Any]] = []
    seen_hashes = set()

    for ref in refs:
        content = str(ref.get("content", "") or "").strip()
        if not content:
            continue

        key = _ref_dedup_key(ref)
        if key in seen_keys:
            continue

        norm = _normalize_text_for_dedup(content)
        content_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue

        too_similar = False
        for existing in kept:
            if _text_similarity(content, existing.get("content", "")) >= _DEDUP_HIGH_SIMILARITY_THRESHOLD:
                too_similar = True
                break
        if too_similar:
            continue

        seen_keys.add(key)
        seen_hashes.add(content_hash)
        kept.append(ref)

    return kept


def _metadata_relevance_bonus(ref: Dict[str, Any], keywords: List[str]) -> int:
    """根據條文標題、主題、條款類型、chunk_label 給額外分數。"""
    if not keywords:
        return 0

    topic_text = normalize_text(" ".join(ref.get("topics", []) if isinstance(ref.get("topics"), list) else [str(ref.get("topics", ""))]))
    core_topic_text = normalize_text(" ".join(ref.get("core_topics", []) if isinstance(ref.get("core_topics"), list) else [str(ref.get("core_topics", ""))]))
    title_text = normalize_text(str(ref.get("article_title", "")))
    clause_type_text = normalize_text(str(ref.get("clause_type", "")))
    label_text = normalize_text(str(ref.get("chunk_label", "")))

    bonus = 0
    for kw in keywords:
        if kw in title_text or kw in label_text:
            bonus += 2
        if kw in topic_text or kw in core_topic_text or kw in clause_type_text:
            bonus += 3
    return bonus


def _mmr_select_refs(
    refs: List[Dict[str, Any]],
    *,
    keywords: List[str],
    top_n: int,
    lambda_: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    用簡化版 MMR 從檢索結果中挑出兼具相關性與多樣性的片段。

    relevance 來源：
    1. 既有檢索順序越前面分數越高；
    2. 內容、條文標題、主題、條款類型命中關鍵詞會加分。
    diversity 來源：
    與已選片段文字越相似，越降低排序。
    """
    if not refs or top_n <= 0:
        return []

    refs = _dedup_chunk_refs(refs)
    if len(refs) <= top_n:
        return refs[:top_n]

    lambda_ = max(0.0, min(1.0, lambda_))
    texts = [str(ref.get("content", "") or "") for ref in refs]
    base_scores: List[float] = []
    total = max(len(refs), 1)

    for idx, ref in enumerate(refs):
        order_score = (total - idx) / total
        norm_content = normalize_text(texts[idx])
        keyword_hits = sum(1 for kw in keywords if kw and kw in norm_content)
        meta_bonus = _metadata_relevance_bonus(ref, keywords)
        base_scores.append(order_score + keyword_hits * 0.15 + meta_bonus * 0.2)

    min_s, max_s = min(base_scores), max(base_scores)
    if max_s > min_s:
        relevance = [(s - min_s) / (max_s - min_s) for s in base_scores]
    else:
        relevance = [1.0 for _ in base_scores]

    selected: List[int] = []
    remaining = list(range(len(refs)))

    first = max(remaining, key=lambda i: relevance[i])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_n and remaining:
        best_i = remaining[0]
        best_score = -999.0
        for i in remaining:
            max_sim = max(_text_similarity(texts[i], texts[j]) for j in selected) if selected else 0.0
            mmr_score = lambda_ * relevance[i] - (1.0 - lambda_) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)

    return [refs[i] for i in selected]

# Vector DB Operations

def upsert_template_vectors(meta: Dict[str, Any], full_text: str, chunks: List[Any]):
    """
    支援兩種 chunks 格式：
    1. 舊版 List[str]
    2. 新版 article-aware List[Dict[str, Any]]
    """
    ensure_db()
    try:
        template_collection.upsert(
            ids=[meta["doc_id"]],
            documents=[full_text[:12000]],
            metadatas=[{
                "doc_id":           meta["doc_id"],
                "file_name":        meta["file_name"],
                "contract_type":    meta.get("contract_type", "其他"),
                "summary":          meta.get("summary", ""),
                "keywords":         ",".join(meta.get("keywords", [])),
                "core_topics":      ",".join(meta.get("core_topics", [])),
                "vendor_name":      meta.get("vendor_name", ""),
                "system_name":      meta.get("system_name", ""),
                "service_scope":    meta.get("service_scope", ""),
                "maintenance_type": meta.get("maintenance_type", ""),
                "industry":         meta.get("industry", ""),
                "contract_name":    meta.get("contract_name", ""),
            }],
        )
    except Exception as e:
        logging.error("歷史基準全文向量入庫失敗: %s", e)

    if not chunks:
        return

    normalized_chunks: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            content = str(chunk.get("content", "") or "").strip()
            if not content:
                continue

            topics_val = chunk.get("topics", [])
            if isinstance(topics_val, list):
                topics = [str(t).strip() for t in topics_val if str(t).strip()]
            else:
                topics = parse_core_topics_field(topics_val)

            if not topics:
                topics = detect_topics(content)

            article_title = str(chunk.get("article_title", "") or chunk.get("title", "") or "")
            clause_type = str(chunk.get("clause_type", "") or "").strip()
            if not clause_type:
                clause_type = detect_clause_type(content, article_title)

            article_no = _metadata_str(chunk.get("article_no"))
            article_title = _metadata_str(article_title)
            parent_article_key = _metadata_str(chunk.get("parent_article_key")) or article_no or f"ARTICLE_{i}"
            chunk_index = int(chunk.get("chunk_index", i) or 0)

            metadata = {
                "doc_id": meta["doc_id"],
                "file_name": meta["file_name"],
                "contract_type": meta.get("contract_type", "其他"),
                "template_role": meta.get("template_role", "歷史基準與規範"),
                "topics": ",".join(topics),
                "core_topics": ",".join(meta.get("core_topics", [])),
                "article_no": article_no,
                "article_title": article_title,
                "chunk_index": chunk_index,
                "parent_article_key": parent_article_key,
                "clause_type": clause_type,
                "vendor_name": meta.get("vendor_name", ""),
                "system_name": meta.get("system_name", ""),
                "service_scope": meta.get("service_scope", ""),
                "maintenance_type": meta.get("maintenance_type", ""),
                "industry": meta.get("industry", ""),
                "contract_name": meta.get("contract_name", ""),
            }
            metadata["chunk_label"] = _chunk_display_label(metadata)

            normalized_chunks.append({
                "id": _stable_chunk_id(meta["doc_id"], parent_article_key, chunk_index, content),
                "content": content,
                "metadata": metadata,
            })
        else:
            content = str(chunk or "").strip()
            if not content:
                continue

            clause_type = detect_clause_type(content, "")

            topics = detect_topics(content)
            metadata = {
                "doc_id": meta["doc_id"],
                "file_name": meta["file_name"],
                "contract_type": meta.get("contract_type", "其他"),
                "template_role": meta.get("template_role", "歷史基準與規範"),
                "topics": ",".join(topics),
                "core_topics": ",".join(meta.get("core_topics", [])),
                "article_no": "",
                "article_title": "",
                "chunk_index": i,
                "parent_article_key": f"ARTICLE_{i}",
                "clause_type": clause_type,
                "vendor_name": meta.get("vendor_name", ""),
                "system_name": meta.get("system_name", ""),
                "service_scope": meta.get("service_scope", ""),
                "maintenance_type": meta.get("maintenance_type", ""),
                "industry": meta.get("industry", ""),
                "contract_name": meta.get("contract_name", ""),
            }
            metadata["chunk_label"] = _chunk_display_label(metadata)

            normalized_chunks.append({
                "id": _stable_chunk_id(meta["doc_id"], metadata["parent_article_key"], i, content),
                "content": content,
                "metadata": metadata,
            })

    if not normalized_chunks:
        return

    try:
        chunk_collection.upsert(
            ids=[x["id"] for x in normalized_chunks],
            documents=[x["content"] for x in normalized_chunks],
            metadatas=[x["metadata"] for x in normalized_chunks],
        )
    except Exception as e:
        logging.error("歷史基準片段向量入庫失敗: %s", e)

def query_templates_fulltext(draft_text: str, n_results: int = 12) -> List[Dict[str, Any]]:
    ensure_db()
    try:
        results = template_collection.query(
            query_texts=[draft_text[:3500]],
            n_results=n_results,
        )
    except Exception as e:
        logging.error("歷史基準全文檢索失敗: %s", e)
        return []

    docs  = (results or {}).get("documents", [[]])
    metas = (results or {}).get("metadatas", [[]])

    if not docs or not docs[0]:
        return []

    refs = []
    for i, doc_text in enumerate(docs[0]):
        meta   = metas[0][i] if metas and metas[0] and i < len(metas[0]) and metas[0][i] else {}
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        refs.append({
            "doc_id":           doc_id,
            "file_name":        meta.get("file_name",     "未知檔案"),
            "contract_type":    meta.get("contract_type", "其他"),
            "summary":          meta.get("summary",       ""),
            "source_text":      doc_text,
            "vendor_name":      meta.get("vendor_name", ""),
            "system_name":      meta.get("system_name", ""),
            "service_scope":    meta.get("service_scope", ""),
            "maintenance_type": meta.get("maintenance_type", ""),
            "industry":         meta.get("industry", ""),
            "contract_name":    meta.get("contract_name", ""),
            "core_topics":      meta.get("core_topics", "").split(",") if meta.get("core_topics") else [],
            "keywords":         meta.get("keywords", "").split(",") if meta.get("keywords") else [],
        })
    return refs


def query_template_chunks_by_query(
    query_text: str,
    candidate_doc_ids: List[str],
    n_results: int = 12,
    target_topic: str = "",
) -> List[Dict[str, Any]]:
    if not candidate_doc_ids:
        return []

    ensure_db()

    def _normalize_chunk_ref(doc_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        topics_raw = meta.get("topics", "")
        topics = [t for t in str(topics_raw).split(",") if t.strip()]
        core_topics_raw = meta.get("core_topics", "")
        core_topics = [t for t in str(core_topics_raw).split(",") if t.strip()]
        return {
            "doc_id": meta.get("doc_id"),
            "file_name": meta.get("file_name", "未知檔案"),
            "content": doc_text,
            "contract_type": meta.get("contract_type", "其他"),
            "template_role": meta.get("template_role", "歷史基準與規範"),
            "topics": topics,
            "topics_text": meta.get("topics", ""),
            "core_topics": core_topics,
            "article_no": meta.get("article_no", ""),
            "article_title": meta.get("article_title", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "parent_article_key": meta.get("parent_article_key", ""),
            "clause_type": meta.get("clause_type", ""),
            "chunk_label": meta.get("chunk_label", ""),
            "vendor_name": meta.get("vendor_name", ""),
            "system_name": meta.get("system_name", ""),
            "service_scope": meta.get("service_scope", ""),
            "maintenance_type": meta.get("maintenance_type", ""),
            "industry": meta.get("industry", ""),
            "contract_name": meta.get("contract_name", ""),
        }

    def _extract_keywords(text: str) -> List[str]:
        text = normalize_text(text)
        raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
        stopwords = {
            "甲方", "乙方", "雙方", "條款", "本條", "本合約", "約定",
            "應", "應於", "以及", "相關", "內容", "方式", "廠商", "草稿",
            "第一條", "第二條", "第三條", "第四條", "第五條", "第六條", "第七條", "第八條", "第九條", "第十條"
        }
        keywords = []
        seen = set()
        for tok in raw_tokens:
            if tok in stopwords:
                continue
            if tok not in seen:
                seen.add(tok)
                keywords.append(tok)
        legal_terms = [
            "資安檢測", "弱點掃描", "滲透測試", "管轄法院", "準據法", "爭議處置",
            "維護人力", "維護時間", "違約金", "損害賠償", "保密義務", "個資保護",
            "智慧財產權", "驗收", "付款", "終止", "解除", "備份", "災難復原", "事件通報",
            "保險代理人", "招攬", "保險商品", "保戶", "要保人", "被保險人", "保險費",
            "佣酬", "佣金", "核保", "理賠", "保險業務員", "廣告文宣", "洗錢防制",
            "打擊資恐", "複委託", "個人資料", "利益衝突", "績效考核"
        ]
        for term in legal_terms:
            if term in text and term not in seen:
                seen.add(term)
                keywords.insert(0, term)
        return keywords[:12]

    refs: List[Dict[str, Any]] = []
    seen_keys = set()

    try:
        results = chunk_collection.query(
            query_texts=[query_text[:2000]],
            n_results=max(n_results * 2, 12),
            where={"doc_id": {"$in": candidate_doc_ids}},
        )

        docs = (results or {}).get("documents", [[]])
        metas = (results or {}).get("metadatas", [[]])

        if docs and docs[0]:
            for i, doc_text in enumerate(docs[0]):
                meta = metas[0][i] if metas and metas[0] and i < len(metas[0]) and metas[0][i] else {}
                ref = _normalize_chunk_ref(doc_text, meta)
                key = (ref["doc_id"], ref["content"][:120])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                refs.append(ref)
    except Exception as e:
        logging.error("歷史基準片段向量檢索失敗: %s", e)

    keywords = _extract_keywords(query_text)
    if keywords:
        try:
            kw_results = chunk_collection.get(
                where={"doc_id": {"$in": candidate_doc_ids}},
                include=["documents", "metadatas"],
            )
            kw_docs = (kw_results or {}).get("documents", [])
            kw_metas = (kw_results or {}).get("metadatas", [])

            lexical_hits = []
            for i, doc_text in enumerate(kw_docs):
                meta = kw_metas[i] if kw_metas and i < len(kw_metas) and kw_metas[i] else {}
                norm_doc = normalize_text(doc_text)
                topic_text = normalize_text(str(meta.get("topics", "")))
                title_text = normalize_text(str(meta.get("article_title", "")))
                clause_type_text = normalize_text(str(meta.get("clause_type", "")))
                label_text = normalize_text(str(meta.get("chunk_label", "")))
                searchable = " ".join([norm_doc, topic_text, title_text, clause_type_text, label_text])
                hit_count = sum(1 for kw in keywords if kw in searchable)
                if hit_count <= 0:
                    continue

                ref = _normalize_chunk_ref(doc_text, meta)
                title_bonus = sum(2 for kw in keywords if kw in title_text or kw in label_text)
                topic_bonus = sum(3 for kw in keywords if kw in topic_text or kw in clause_type_text)
                lexical_hits.append((hit_count + title_bonus + topic_bonus, ref))

            lexical_hits.sort(key=lambda x: x[0], reverse=True)

            for _, ref in lexical_hits[: max(n_results, 8)]:
                key = (ref["doc_id"], ref["content"][:120])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                refs.append(ref)

        except Exception as e:
            logging.warning("歷史基準片段 keyword 補召回失敗: %s", e)

    refs = _dedup_chunk_refs(refs)

    if target_topic:
        refs = filter_chunks_by_topic(refs, target_topic)
        topic_terms = _topic_filter_terms(target_topic)
        keywords = list(dict.fromkeys(topic_terms + keywords))[:16]

    refs = _mmr_select_refs(refs, keywords=keywords, top_n=n_results, lambda_=0.65)
    return refs[:n_results]