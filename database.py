import os
import sqlite3
import json
import logging
import datetime
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from utils import detect_topics
from config import SQLITE_DB_PATH, CHROMA_DIR, EMBED_MODEL

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
}


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


# Vector DB Operations

def upsert_template_vectors(meta: Dict[str, Any], full_text: str, chunks: List[str]):
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
                "keywords":         ",".join(meta.get("keywords",    [])),
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

    try:
        chunk_collection.upsert(
            ids=[f"{meta['doc_id']}_chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{
                "doc_id":        meta["doc_id"],
                "file_name":     meta["file_name"],
                "contract_type": meta.get("contract_type", "其他"),
                "topics":        ",".join(detect_topics(chunk)),
            } for chunk in chunks],
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
    query_text: str, candidate_doc_ids: List[str], n_results: int = 12
) -> List[Dict[str, Any]]:
    if not candidate_doc_ids:
        return []

    ensure_db()
    try:
        results = chunk_collection.query(
            query_texts=[query_text[:2000]],
            n_results=max(n_results, 12),
            where={"doc_id": {"$in": candidate_doc_ids}},
        )
    except Exception as e:
        logging.error("歷史基準片段檢索失敗: %s", e)
        return []

    docs  = (results or {}).get("documents", [[]])
    metas = (results or {}).get("metadatas", [[]])

    if not docs or not docs[0]:
        return []

    refs = []
    for i, doc_text in enumerate(docs[0]):
        if len(refs) >= n_results:
            break
        meta = metas[0][i] if metas and metas[0] and i < len(metas[0]) and metas[0][i] else {}
        refs.append({
            "doc_id":        meta.get("doc_id"),
            "file_name":     meta.get("file_name",     "未知檔案"),
            "content":       doc_text,
            "contract_type": meta.get("contract_type", "其他"),
            "topics":        meta.get("topics",        ""),
        })
    return refs