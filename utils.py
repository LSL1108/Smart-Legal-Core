import os
import re
import json
import uuid
import hashlib
import datetime
import logging
from typing import Dict, Any, List, Optional
from pypdf import PdfReader
from docx import Document as DocxReader

# 引入 config 中的常數
from config import UPLOAD_DIR, TOPIC_KEYWORDS, TOPIC_ALIAS, TOPIC_MIN_MATCHES

# 檔案與路徑工具
def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def make_output_path(filename: str) -> str:
    ensure_upload_dir()
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")

# 🛡️ 資安防護：過濾檔名，防止路徑穿越攻擊 (Path Traversal)
def secure_filename(filename: str) -> str:
    """
    只保留英數字、中文字、點(.)、底線(_)和橫線(-)，
    過濾掉如 ../ 等可能導致目錄穿越的危險符號。
    """
    if not filename:
        return "unnamed_file"
    safe_name = re.sub(r'[^\w\u4e00-\u9fa5\.\-]', '_', filename)
    return safe_name.lstrip('.')

def save_upload_file(up_file) -> Dict[str, Any]:
    ensure_upload_dir()
    data = up_file.getvalue()
    if not data:
        raise ValueError("上傳檔案大小為 0 bytes，請重新上傳。")

    ext = os.path.splitext(up_file.name)[-1].lower().lstrip(".")
    file_id = str(uuid.uuid4())

    safe_name = secure_filename(up_file.name)
    storage_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

    with open(storage_path, "wb") as f:
        f.write(data)

    return {
        "doc_id": file_id,
        "file_name": safe_name,
        "file_type": ext,
        "storage_path": storage_path,
        "sha256": sha256_bytes(data),
        "byte_size": len(data),
        "created_at": datetime.datetime.now(),
    }

# 文本萃取工具
def extract_text_from_pdf(file_or_path, max_pages: int = 30) -> str:
    try:
        reader = PdfReader(file_or_path)
        texts = [page.extract_text() or "" for page in reader.pages[:max_pages]]
        return normalize_text("\n".join(texts))
    except Exception as e:
        logging.error(f"PDF 讀取失敗: {e}")
        return ""

def extract_text_from_docx(file_or_path) -> str:
    try:
        doc = DocxReader(file_or_path)
        paras = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                paras.append(t)
        return normalize_text("\n".join(paras))
    except Exception as e:
        logging.error(f"DOCX 讀取失敗: {e}")
        return ""

# 字串與正規化工具
def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def short_text(text: str, limit: int = 320) -> str:
    text = normalize_text(text)
    return text if len(text) <= limit else text[:limit] + "..."

def safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except Exception:
            pass
    return {}

def normalize_term(term: str) -> str:
    term = (term or "").strip()
    if not term:
        return term
    if re.search(r"(一年|1\s*年)", term) and ("至" not in term):
        start = datetime.date.today()
        try:
            end = datetime.date(start.year + 1, start.month, start.day) - datetime.timedelta(days=1)
        except ValueError:
            end = datetime.date(start.year + 1, start.month, start.day - 1) - datetime.timedelta(days=1)
        return f"{start.strftime('%Y年%m月%d日')}至{end.strftime('%Y年%m月%d日')}"
    return term

def parse_template_selector(text: str) -> Dict[str, str]:
    m = re.search(r'template\s*=\s*"([^"]+)"', text)
    if m:
        return {"file_name": m.group(1).strip()}
    return {}

# Semantic Chunking
def chunk_text(text: str, chunk_size: int = 650, overlap: int = 120) -> List[str]:
    """
    合約專用語義切塊演算法：
    優先依照「第X條」進行切割，確保法條語義完整。若單一條文過長，再依賴長度切塊。
    """
    text = normalize_text(text)
    if not text:
        return []

    pattern = r"(?=\n?第[一二三四五六七八九十百0-9]+條[：:\s])"
    raw_chunks = re.split(pattern, text)

    chunks = []
    current_chunk = ""

    for part in raw_chunks:
        part = part.strip()
        if not part:
            continue
        if len(current_chunk) + len(part) <= chunk_size:
            current_chunk += ("\n\n" + part if current_chunk else part)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(part) > chunk_size:
                start = 0
                step = max(1, chunk_size - overlap)
                while start < len(part):
                    chunks.append(part[start:start + chunk_size])
                    start += step
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# 業務邏輯與主題判定工具
def normalize_topic_name(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        return ""
    for canonical, aliases in TOPIC_ALIAS.items():
        if topic == canonical:
            return canonical
        for alias in aliases:
            if topic == alias or alias in topic or topic in alias:
                return canonical
    return topic

def detect_contract_mode_from_text(text: str) -> str:
    text = text or ""
    flags = {
        "維護": any(k in text for k in ["維護", "維運", "故障", "SLA", "修復"]),
        "開發": any(k in text for k in ["開發", "系統設計", "原始碼", "程式碼", "平台", "智慧財產權"]),
        "保密": any(k in text for k in ["保密", "機密", "揭露", "GitHub", "開源"]),
    }
    active = [k for k, v in flags.items() if v]
    if len(active) >= 2:
        return "混合型"
    return active[0] if active else "其他"

def detect_topics(text: str) -> List[str]:
    """
    ⚠️ v2：改用 TOPIC_MIN_MATCHES 設定各主題最低命中門檻，
    避免單一寬泛關鍵字導致 False Positive。
    """
    text = normalize_text(text)
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        min_hits = TOPIC_MIN_MATCHES.get(topic, 1)
        hit_count = sum(1 for k in keywords if k in text)
        if hit_count >= min_hits:
            found.append(topic)
    return found

def score_topic_overlap(a: List[str], b: List[str]) -> int:
    return len(set(a or []) & set(b or []))

def lexical_score(text: str, query: str) -> int:
    text = normalize_text(text)
    query = normalize_text(query)
    score = 0
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", query):
        if token in text:
            score += 1
    return score

def parse_core_topics_field(val: Any) -> List[str]:
    if isinstance(val, list):
        return [normalize_topic_name(str(x)) for x in val if str(x).strip()]
    if isinstance(val, str):
        parts = re.split(r"[、,，;；\s]+", val)
        return [normalize_topic_name(x) for x in parts if x.strip()]
    return []

# =========================
# ✂️ 草稿解析工具
# =========================
def split_draft_into_articles(text: str) -> List[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return []

    pattern = r"(第[一二三四五六七八九十百0-9]+條[：:]\s*.*?)(?=(?:\n?第[一二三四五六七八九十百0-9]+條[：:])|$)"
    matches = re.findall(pattern, text, flags=re.S)

    articles = []
    if matches:
        for raw in matches:
            raw = normalize_text(raw)
            m = re.match(r"(第[一二三四五六七八九十百0-9]+條)[：:]\s*([^\n ]+)?\s*(.*)", raw, flags=re.S)
            if m:
                content = normalize_text(raw)
                articles.append({
                    "article_no": m.group(1).strip(),
                    "title": (m.group(2) or "").strip(),
                    "content": content,
                    "topics": detect_topics(content),
                })
            else:
                articles.append({
                    "article_no": "",
                    "title": "",
                    "content": raw,
                    "topics": detect_topics(raw),
                })
        return articles

    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return [{
        "article_no": "",
        "title": "",
        "content": p,
        "topics": detect_topics(p),
    } for p in paras]

def build_article_map(articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    article_map = {}
    for idx, a in enumerate(articles, start=1):
        no = (a.get("article_no", "") or "").strip()
        title = (a.get("title", "") or "").strip()
        if no:
            article_map[no] = a
        article_map[f"ARTICLE_{idx}"] = a
        if title:
            article_map[title] = a
    return article_map

def article_to_key(article: Dict[str, Any], idx: int) -> str:
    no = (article.get("article_no", "") or "").strip()
    return no if no else f"ARTICLE_{idx}"