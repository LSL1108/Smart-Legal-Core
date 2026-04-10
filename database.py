import os
import sqlite3
import json
import logging
import datetime
from typing import Dict, Any, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from config import SQLITE_DB_PATH, CHROMA_DIR, EMBED_MODEL

# SQLite
def get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_sqlite():
    conn = get_sqlite_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            file_name TEXT NOT NULL,
            file_type TEXT,
            storage_path TEXT,
            sha256 TEXT UNIQUE,
            byte_size INTEGER,
            created_at TEXT,
            contract_type TEXT,
            summary TEXT,
            keywords TEXT,
            template_role TEXT,
            core_topics TEXT,
            source_text TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS contract_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT NOT NULL,
            amount INTEGER NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def row_to_template_dict(row: Optional[sqlite3.Row]) -> Dict[str, Any]:
    if row is None:
        return {}
    d = dict(row)
    d["keywords"] = json.loads(d["keywords"]) if d.get("keywords") else []
    d["core_topics"] = json.loads(d["core_topics"]) if d.get("core_topics") else []
    return d

def template_exists_by_sha256(sha256_val: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM templates WHERE sha256 = ?", (sha256_val,))
    row = cur.fetchone()
    conn.close()
    return row_to_template_dict(row) if row else None

def insert_template_doc(doc: Dict[str, Any]):
    conn = get_sqlite_conn()
    cur = conn.cursor()

    created_at = doc.get("created_at")
    if isinstance(created_at, (datetime.datetime, datetime.date)):
        created_at = created_at.isoformat()
    else:
        created_at = str(created_at or "")

    cur.execute("""
        INSERT INTO templates (
            doc_id, file_name, file_type, storage_path, sha256, byte_size, created_at,
            contract_type, summary, keywords, template_role, core_topics, source_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        json.dumps(doc.get("keywords", []), ensure_ascii=False),
        doc.get("template_role", "標準模板"),
        json.dumps(doc.get("core_topics", []), ensure_ascii=False),
        doc.get("source_text", "")
    ))

    conn.commit()
    conn.close()

def get_template_by_doc_id(doc_id: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM templates WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
    return row_to_template_dict(row) if row else None

def get_all_templates() -> List[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM templates ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [row_to_template_dict(r) for r in rows]

def delete_template_by_doc_id(doc_id: str):
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM templates WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()

def search_templates_sql(contract_type: str, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """用於合約生成 (DocxTemplate) 的快速尋找"""
    conn = get_sqlite_conn()
    cur = conn.cursor()

    if contract_type.strip():
        cur.execute("""
            SELECT * FROM templates
            WHERE file_type = 'docx' AND contract_type LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{contract_type.strip()}%", limit))
        rows = cur.fetchall()
    else:
        rows = []

    if not rows:
        cur.execute("""
            SELECT * FROM templates
            WHERE file_type = 'docx'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()

    conn.close()
    docs = [row_to_template_dict(r) for r in rows]

    q = (query or "").strip()
    def score(doc):
        score_val = 0
        text = " ".join(doc.get("keywords", [])) + " " + (doc.get("summary", "") or "")
        for token in q.split():
            if len(token) >= 2 and token in text:
                score_val += 2
        return score_val

    return sorted(docs, key=score, reverse=True)

def get_template_by_file_name_like(file_name: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM templates
        WHERE file_name LIKE ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (f"%{file_name}%",))
    row = cur.fetchone()
    conn.close()
    return row_to_template_dict(row) if row else None

def get_template_by_selector(selector: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """根據選擇器條件從資料庫尋找特定的模板"""
    if not selector:
        return None
    if selector.get("file_name"):
        return get_template_by_file_name_like(selector["file_name"])
    return None

# 歷史報價相關操作
def count_history_records() -> int:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM contract_history")
    row = cur.fetchone()
    conn.close()
    return int(row["cnt"])

def insert_history_records(records: List[Dict[str, Any]]):
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO contract_history (vendor_name, amount)
        VALUES (?, ?)
    """, [(r["vendor_name"], r["amount"]) for r in records])
    conn.commit()
    conn.close()

def find_history_by_vendor_keyword(keyword: str) -> List[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM contract_history
        WHERE vendor_name LIKE ?
    """, (f"%{keyword}%",))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# 初始化 SQLite
os.makedirs(os.path.dirname(SQLITE_DB_PATH) or ".", exist_ok=True)
init_sqlite()



# ChromaDB
def get_chroma():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # 使用 Ollama 的 embedding 模型
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )

    template_collection = chroma_client.get_or_create_collection(
        name="contract_templates_fulltext",
        embedding_function=ollama_ef
    )

    chunk_collection = chroma_client.get_or_create_collection(
        name="contract_template_chunks",
        embedding_function=ollama_ef
    )

    return chroma_client, template_collection, chunk_collection

# 獲取 Chroma 例項
chroma_client, template_collection, chunk_collection = get_chroma()

def upsert_template_vectors(meta: Dict[str, Any], full_text: str, chunks: List[str]):
    """將完整文本與切塊(Chunks)存入向量資料庫"""
    try:
        template_collection.upsert(
            ids=[meta["doc_id"]],
            documents=[full_text[:12000]], # 避免過長
            metadatas=[{
                "doc_id": meta["doc_id"],
                "file_name": meta["file_name"],
                "contract_type": meta.get("contract_type", "其他"),
                "summary": meta.get("summary", ""),
                "keywords": ",".join(meta.get("keywords", [])),
                "core_topics": ",".join(meta.get("core_topics", [])),
            }]
        )
    except Exception as e:
        logging.error(f"模板全文向量入庫失敗: {e}")

    if chunks:
        ids = [f"{meta['doc_id']}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "doc_id": meta["doc_id"],
            "file_name": meta["file_name"],
            "contract_type": meta.get("contract_type", "其他"),
        } for _ in chunks]

        try:
            chunk_collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas
            )
        except Exception as e:
            logging.error(f"模板片段向量入庫失敗: {e}")

def query_templates_fulltext(draft_text: str, n_results: int = 12) -> List[Dict[str, Any]]:
    """根據草稿全文，搜尋最相關的標準模板"""
    try:
        results = template_collection.query(
            query_texts=[draft_text[:3500]],
            n_results=n_results
        )
    except Exception as e:
        logging.error(f"模板全文檢索失敗: {e}")
        return []

    refs = []
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])

    if not docs or not docs[0]:
        return refs

    for i, doc_text in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue

        refs.append({
            "doc_id": doc_id,
            "file_name": meta.get("file_name", "未知模板"),
            "contract_type": meta.get("contract_type", "其他"),
            "summary": meta.get("summary", ""),
            "source_text": doc_text,
        })
    return refs

def query_template_chunks_by_query(query_text: str, candidate_doc_ids: List[str], n_results: int = 12) -> List[Dict[str, Any]]:
    """根據單條草稿，從選定的候選模板中搜尋最相關的條文片段 (Chunks)"""
    if not candidate_doc_ids:
        return []

    try:
        results = chunk_collection.query(
            query_texts=[query_text[:2000]],
            n_results=max(n_results, 12)
        )
    except Exception as e:
        logging.error(f"模板片段檢索失敗: {e}")
        return []

    refs = []
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])

    if not docs or not docs[0]:
        return refs

    for i, doc_text in enumerate(docs[0]):
        meta = metas[0][i] if metas and metas[0] else {}
        doc_id = meta.get("doc_id")
        
        # 過濾：只抓取我們先前選定為「高關聯」的標準模板
        if doc_id not in candidate_doc_ids:
            continue

        refs.append({
            "doc_id": doc_id,
            "file_name": meta.get("file_name", "未知模板"),
            "content": doc_text,
            "contract_type": meta.get("contract_type", "其他"),
        })

        if len(refs) >= n_results:
            break

    return refs