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

import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import UPLOAD_DIR, TOPIC_KEYWORDS, TOPIC_ALIAS, TOPIC_MIN_MATCHES, ALL_TOPICS_FOR_PROMPT, MODEL

def ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def make_output_path(filename: str) -> str:
    ensure_upload_dir()
    return os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")


def secure_filename(filename: str) -> str:

    if not filename:
        return "unnamed_file"
    safe_name = re.sub(r'[^\w\u4e00-\u9fa5\.\-]', '_', filename)
    return safe_name.lstrip('.')

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

def save_upload_file(up_file) -> Dict[str, Any]:
    ensure_upload_dir()
    data = up_file.getvalue()
    if not data:
        raise ValueError("上傳檔案大小為 0 bytes，請重新上傳。")
    if len(data) > MAX_UPLOAD_SIZE:
        raise ValueError(
            f"檔案「{up_file.name}」超過 {MAX_UPLOAD_SIZE // (1024 * 1024)} MB 上限，請壓縮後重試。"
        )

    ext       = os.path.splitext(up_file.name)[-1].lower().lstrip(".")
    file_id   = str(uuid.uuid4())
    safe_name = secure_filename(up_file.name)
    storage_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

    with open(storage_path, "wb") as f:
        f.write(data)

    return {
        "doc_id":       file_id,
        "file_name":    safe_name,
        "file_type":    ext,
        "storage_path": storage_path,
        "sha256":       sha256_bytes(data),
        "byte_size":    len(data),
        "created_at":   datetime.datetime.now(),
    }

# 文本萃取工具
def extract_text_from_pdf(file_or_path, max_pages: Optional[int] = None) -> str:

    from config import MAX_PDF_PAGES
    if max_pages is None:
        max_pages = MAX_PDF_PAGES

    try:
        reader = PdfReader(file_or_path, strict=False)
    except Exception as e:
        logging.error(f"PDF 開啟失敗: {e}")
        return ""

    texts: List[str] = []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            logging.warning(
                f"PDF 超過 {max_pages} 頁（總計 {total} 頁），已截斷後續內容；"
                f"如需處理更長文件，請調高 MAX_PDF_PAGES 環境變數。"
            )
            break
        try:
            texts.append(page.extract_text() or "")
        except Exception as e:
            logging.warning(f"PDF 第 {i + 1} 頁解析失敗，已 skip: {e}")
            continue
    return normalize_text("\n".join(texts))

def extract_text_from_docx(file_or_path) -> str:

    try:
        doc   = DocxReader(file_or_path)
        parts = []

        # 段落文字
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                parts.append(t)

        for table in doc.tables:
            seen_cells: set = set()
            for row in table.rows:
                for cell in row.cells:
                    cell_text = (cell.text or "").strip()
                    if cell_text and cell_text not in seen_cells:
                        seen_cells.add(cell_text)
                        parts.append(cell_text)

        return normalize_text("\n".join(parts))
    except Exception as e:
        logging.error(f"DOCX 讀取失敗: {e}")
        return ""

# 字串與正規化工具
def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n?",   "\n",   text)
    text = re.sub(r"\n{3,}",  "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ",  text)
    return text.strip()

def short_text(text: str, limit: int = 320) -> str:
    text = normalize_text(text)
    return text if len(text) <= limit else text[:limit] + "..."

def safe_json_load(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```",        "", s).strip()
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


_ARTICLE_NUM_RE = r"[一二三四五六七八九十百千萬壹貳參肆伍陸柒捌玖拾0-9]+"

ARTICLE_HEADER_RE = re.compile(
    rf"(?=(?:^|\n)\s*第\s*{_ARTICLE_NUM_RE}\s*條(?:[：:、\s]|$))",
    flags=re.M,
)

ARTICLE_NO_TITLE_RE = re.compile(
    rf"^\s*(第\s*({_ARTICLE_NUM_RE})\s*條)\s*[：:、]?\s*([^\n]{{0,60}})?",
    flags=re.M,
)

_HEADING_PATTERN = re.compile(
    rf"^(?:#+\s+|[一二三四五六七八九十百千萬壹貳參肆伍陸柒捌玖拾]+、|第\s*{_ARTICLE_NUM_RE}\s*條)"
)

def _clean_article_title(title: str) -> str:
    """清理條文標題，避免把整段條文誤當成標題。"""
    title = normalize_text(title or "")
    title = re.sub(r"^[：:、\s]+", "", title).strip()
    if not title:
        return ""

    # 標題通常很短；若含句號、分號或過長，代表可能抓到正文。
    if len(title) > 24:
        return ""
    if re.search(r"[。；;]", title):
        return ""
    return title


def _article_header_label(article_no: str, article_title: str = "") -> str:
    article_no = normalize_text(article_no or "")
    article_title = normalize_text(article_title or "")
    if article_no and article_title:
        return f"{article_no} {article_title}"
    if article_no:
        return article_no
    if article_title:
        return article_title
    return ""

def detect_topics_fast(text: str) -> List[str]:

    text_n = normalize_text(text)
    if not text_n:
        return []

    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        min_hits = TOPIC_MIN_MATCHES.get(topic, 1)
        hits = sum(1 for k in keywords if k and k in text_n)
        if hits >= min_hits:
            found.append(topic)

    dedup = []
    seen = set()
    for t in found:
        nt = normalize_topic_name(t)
        if nt and nt not in seen:
            seen.add(nt)
            dedup.append(nt)
    return dedup

def _split_text_into_article_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    raw_parts = re.split(ARTICLE_HEADER_RE, text)
    parts = [normalize_text(p) for p in raw_parts if normalize_text(p)]

    # 若條文數量不足，退回通用單段，避免把一般文件切得太碎。
    return parts if len(parts) >= 2 else [text]

def _parse_article_block(raw: str, idx: int) -> Dict[str, Any]:
    raw = normalize_text(raw)
    first_line = raw.split("\n", 1)[0].strip()

    m = ARTICLE_NO_TITLE_RE.match(first_line)
    if m:
        article_no = re.sub(r"\s+", "", m.group(1))
        possible_title = _clean_article_title(m.group(3) or "")

        return {
            "article_no": article_no,
            "title": possible_title,
            "content": raw,
            "topics": [],
            "parent_article_key": article_no or f"ARTICLE_{idx}",
        }

    return {
        "article_no": "",
        "title": "",
        "content": raw,
        "topics": [],
        "parent_article_key": f"ARTICLE_{idx}",
    }

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:

    articles = split_draft_into_articles(text)
    if not articles:
        return []

    chunks: List[Dict[str, Any]] = []

    for idx, article in enumerate(articles, start=1):
        article_no = (article.get("article_no") or "").strip()
        article_title = (article.get("title") or "").strip()
        article_text = normalize_text(article.get("content", ""))
        article_topics = article.get("topics", []) or []

        parent_article_key = (
            article.get("parent_article_key")
            or article_no
            or article_title
            or f"ARTICLE_{idx}"
        )

        if not article_text:
            continue

        if len(article_text) <= chunk_size:
            chunks.append({
                "article_no": article_no,
                "article_title": article_title,
                "content": article_text,
                "chunk_index": 0,
                "parent_article_key": parent_article_key,
                "topics": article_topics,
            })
            continue

        step = max(1, chunk_size - overlap)
        start = 0
        chunk_index = 0

        header_label = _article_header_label(article_no, article_title)

        while start < len(article_text):
            piece = article_text[start:start + chunk_size].strip()
            if piece:
                # 長條款被切成多段時，續段補上條號／標題，避免檢索後失去所屬條文脈絡。
                if chunk_index > 0 and header_label and not piece.startswith(article_no):
                    piece = f"{header_label}（續）\n{piece}"

                chunks.append({
                    "article_no": article_no,
                    "article_title": article_title,
                    "content": piece,
                    "chunk_index": chunk_index,
                    "parent_article_key": parent_article_key,
                    "topics": article_topics,
                })
                chunk_index += 1
            start += step

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
    text  = text or ""
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

    text = normalize_text(text)
    if not text or len(text) < 10:
        return []

    fast_topics = detect_topics_fast(text)
    if fast_topics:
        return fast_topics

    prompt = f"""
    你是一個專業的法務合約分類系統。請閱讀以下合約條文，並判斷它涉及哪些主題。

    【強制規定】：
    1. 只能從以下「標準主題清單」中挑選，絕對不能自己發明新詞彙：
    {", ".join(ALL_TOPICS_FOR_PROMPT)}
    2. 若條文提及「對價、匯款、費用」，請歸類為「付款價金」。
    3. 如果該條文沒有涉及清單中的任何主題，請讓陣列保持空白。
    4. 必須輸出合法的 JSON 物件格式，範例：{{"topics": ["付款價金", "違約金"]}}。絕對不要輸出其他說明文字。

    條文內容：
    {text[:800]}
    """

    raw_response = ""
    try:
        res = ollama.generate(
            model=MODEL,
            prompt=prompt.strip(),
            format="json",
            options={"temperature": 0.0, "top_p": 0.1}
        )

        raw_response = (res or {}).get("response", "{}").strip()
        raw_response = re.sub(r"```json\s*", "", raw_response)
        raw_response = re.sub(r"```", "", raw_response).strip()

        data = json.loads(raw_response)
        topics = data.get("topics", [])

        if not isinstance(topics, list):
            topics = [t for t in ALL_TOPICS_FOR_PROMPT if t in raw_response]

        valid_topics = []
        seen = set()
        for t in topics:
            t_str = normalize_topic_name(str(t).strip())
            if t_str in ALL_TOPICS_FOR_PROMPT and t_str not in seen:
                seen.add(t_str)
                valid_topics.append(t_str)

        return valid_topics

    except Exception as e:
        logging.warning(f"LLM 判斷主題失敗: {e} | 退回字串暴力比對模式")
        return [t for t in ALL_TOPICS_FOR_PROMPT if t in text]

def score_topic_overlap(a: List[str], b: List[str]) -> int:
    return len(set(a or []) & set(b or []))

def lexical_score(text: str, query: str) -> int:
    text  = normalize_text(text)
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

# 草稿解析工具
def split_draft_into_articles(text: str) -> List[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return []

    raw_blocks = _split_text_into_article_blocks(text)
    raw_articles = [_parse_article_block(raw, idx + 1) for idx, raw in enumerate(raw_blocks)]

    def _detect_and_return(idx, content):
        fast_topics = detect_topics_fast(content)
        if fast_topics:
            return idx, fast_topics
        return idx, detect_topics(content)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_detect_and_return, i, a["content"]): i for i, a in enumerate(raw_articles)}
        for future in as_completed(futures):
            idx, topics = future.result()
            raw_articles[idx]["topics"] = topics

    return raw_articles

def build_article_map(articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    article_map: Dict[str, Dict[str, Any]] = {}
    for idx, a in enumerate(articles, start=1):
        no    = (a.get("article_no", "") or "").strip()
        title = (a.get("title",      "") or "").strip()
        parent_key = (a.get("parent_article_key", "") or "").strip()

        if no:
            article_map[no] = a
        if title:
            article_map[title] = a
        if parent_key:
            article_map[parent_key] = a

        article_map[f"ARTICLE_{idx}"] = a
    return article_map

def article_to_key(article: Dict[str, Any], idx: int) -> str:
    no = (article.get("article_no", "") or "").strip()
    if no:
        return no

    parent_key = (article.get("parent_article_key", "") or "").strip()
    if parent_key:
        return parent_key

    return f"ARTICLE_{idx}"

def detect_clause_type(article_text: str, article_title: str = "") -> str:
    text = normalize_text(f"{article_title} {article_text}")

    rules = [
        ("付款條款", ["付款", "價金", "費用", "匯款", "發票", "請款", "報酬", "對價"]),
        ("違約責任", ["違約", "違約金", "逾期", "遲延", "損害賠償", "賠償責任"]),
        ("維護服務/SLA", ["維護", "維運", "故障", "修復", "回覆時限", "服務水準", "SLA", "弱點修補"]),
        ("保密條款", ["保密", "機密", "秘密資訊", "不得揭露", "揭露", "保密義務"]),
        ("智慧財產條款", ["智慧財產", "著作權", "專利", "原始碼", "程式碼", "技術成果", "授權"]),
        ("驗收交付條款", ["驗收", "交付", "測試", "上線", "交付物", "成果交付"]),
        ("爭議解決條款", ["爭議", "準據法", "管轄", "法院", "仲裁", "合意裁判"]),
        ("人力配置條款", ["人力", "專責", "工程師", "人員", "窗口", "團隊成員"]),
    ]

    scores = []
    for clause_type, keywords in rules:
        hit_count = sum(1 for kw in keywords if kw in text)
        if hit_count > 0:
            scores.append((hit_count, clause_type))

    if not scores:
        return "一般條款"

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]