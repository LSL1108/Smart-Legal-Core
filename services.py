import os
import re
import time
import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
from models import UserRequestIntent, ReviewReport
import ollama
from docxtpl import DocxTemplate

from config import (
    MODEL,
    CHAT_MODEL,
    REVIEW_MODEL,
    JSON_MODEL,
    ALL_TOPICS_FOR_PROMPT,
    TOPIC_KEYWORDS,
    TOPIC_ALIAS,
    TOPIC_MIN_MATCHES,
    CRITICAL_RISK_TRIGGERS,
    HIGH_RISK_TRIGGERS,
    CONTRACT_TYPE_RULESET,
    OLLAMA_URL,
    LLM_NUM_CTX,
    LLM_TIMEOUT_SEC,
    OLLAMA_MAX_CONCURRENCY,
    MAX_ARTICLE_CHARS,
    MAX_CHUNK_CHARS,
    MAX_PROMPT_CHARS,
    MAX_QUERY_CHARS,
    MAX_INGEST_CHARS,
)

from utils import (
    safe_json_load, normalize_text, detect_topics, lexical_score, score_topic_overlap,
    detect_contract_mode_from_text, build_article_map, normalize_topic_name,
    article_to_key, short_text, split_draft_into_articles, extract_text_from_pdf,
    extract_text_from_docx, make_output_path, normalize_term, save_upload_file,
    parse_template_selector, chunk_text, detect_clause_type
)

from database import (
    query_templates_fulltext,
    query_template_chunks_by_query,
    find_history_by_vendor_keyword,
    template_exists_by_sha256,
    insert_template_doc,
    upsert_template_vectors,
    get_template_by_selector,
    search_templates_sql,
    template_collection,
    get_compliance_rules,
    DEFAULT_COMPLIANCE_RULES,
)

from clause_followup_service import answer_clause_followup

from text_normalize import ensure_traditional, ensure_traditional_in_obj

from rule_engine import (
    load_rules,
    evaluate_all_rules,
    render_issue_from_hit,
    merge_issues_by_topic,
)



_ollama_client = ollama.Client(host=OLLAMA_URL, timeout=LLM_TIMEOUT_SEC)
_ollama_semaphore = threading.Semaphore(OLLAMA_MAX_CONCURRENCY)


def _truncate(text: str, limit: int, suffix: str = "...(已截斷)") -> str:
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + suffix


def ollama_json(
    prompt: str,
    model: str = JSON_MODEL,
    temperature: float = 0.0,
    top_p: float = 0.1,
    retries: int = 2,
    num_ctx: Optional[int] = None,
) -> Dict[str, Any]:
    prompt = _truncate(prompt.strip(), MAX_PROMPT_CHARS)
    effective_num_ctx = num_ctx or LLM_NUM_CTX

    for attempt in range(retries + 1):
        try:
            logging.info(f"ollama_json 使用模型：{model}")

            with _ollama_semaphore:
                res = _ollama_client.generate(
                    model=model,
                    prompt=prompt,
                    format="json",
                    options={
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_ctx": effective_num_ctx,
                    },
                )

            return ensure_traditional_in_obj(
                safe_json_load((res or {}).get("response", "{}"))
            )

        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                logging.warning(
                    f"Ollama JSON 失敗（第 {attempt + 1} 次），{wait}s 後重試：{e}"
                )
                time.sleep(wait)
            else:
                logging.error(f"Ollama JSON 失敗（已重試 {retries} 次）：{e}")
                return {}

# 文件入庫與意圖分析

def llm_ingest_contract(text: str) -> Dict[str, Any]:
    prompt = f"""
你是企業法遵與文件入庫助理。請只輸出合法 JSON。
請根據以下合約或規範內容抽取 metadata。
core_topics 請優先從下列主題中選 3~10 個：
{", ".join(ALL_TOPICS_FOR_PROMPT)}

內容：
{_truncate(text, MAX_INGEST_CHARS, "")}

JSON 格式：
{{
"contract_type": "請根據內容真實屬性，精煉出標準名稱（例如：維護合約、資安規範、採購合約等，限10字以內）",
"summary": "80~160字摘要",
"keywords": ["關鍵字1", "關鍵字2"],
"template_role": "歷史基準與規範",
"core_topics": ["違約金", "資安檢測與掃描"]
}}
"""
    obj = ollama_json(prompt)
    if "keywords" not in obj or not isinstance(obj["keywords"], list):
        obj["keywords"] = []
    if "core_topics" not in obj or not isinstance(obj["core_topics"], list):
        obj["core_topics"] = detect_topics(text)

    obj["contract_type"] = str(obj.get("contract_type", "其他") or "其他").strip()
    obj["summary"] = str(obj.get("summary", "") or "").strip()
    obj["template_role"] = str(obj.get("template_role", "歷史基準與規範") or "歷史基準與規範").strip()
    obj["core_topics"] = [normalize_topic_name(x) for x in obj.get("core_topics", []) if str(x).strip()]
    return obj



def handle_upload(files):
    inserted = 0
    skipped = 0
    for f in files:
        meta = save_upload_file(f)
        exist = template_exists_by_sha256(meta["sha256"])
        if exist:
            skipped += 1
            continue

        if meta["file_type"] == "pdf":
            text = extract_text_from_pdf(meta["storage_path"])
        else:
            text = extract_text_from_docx(meta["storage_path"])

        if not text.strip():
            logging.warning(f"檔案無法解析文字：{meta['file_name']}")
            skipped += 1
            continue

        ing = llm_ingest_contract(text)
        doc = {
            **meta,
            "contract_type": ing.get("contract_type", "其他"),
            "summary": ing.get("summary", ""),
            "keywords": ing.get("keywords", []),
            "template_role": ing.get("template_role", "歷史基準與規範"),
            "core_topics": ing.get("core_topics", []),
            "source_text": text[:20000],
        }
        chunks = chunk_text(text)

        insert_template_doc(doc)
        upsert_template_vectors(doc, text, chunks)
        inserted += 1

    return inserted, skipped



def llm_parse_user_request(message: str) -> Dict[str, Any]:
    prompt = f"""
你是合約助理。請把需求整理成 JSON，不要輸出其他文字。
JSON 結構：
{{
"intent": "generate | review",
"contract_type": "維護合約/其他",
"fields": {{
  "party_a": "",
  "party_b": "",
  "amount": "",
  "term": "",
  "system_name": "",
  "vendor_name": "",
  "service_scope": "",
  "maintenance_type": "",
  "industry": "",
  "contract_name": ""
}},
"notes": ""
}}
使用者訊息：{message}
"""
    raw_dict = ollama_json(prompt)
    raw_fields = raw_dict.get("fields") if isinstance(raw_dict, dict) else {}
    if not isinstance(raw_fields, dict):
        raw_fields = {}

    raw_fields.setdefault("party_a", "")
    raw_fields.setdefault("party_b", "")
    raw_fields.setdefault("amount", "")
    raw_fields.setdefault("term", "")
    raw_fields.setdefault("system_name", "")
    raw_fields.setdefault("vendor_name", "")
    raw_fields.setdefault("service_scope", "")
    raw_fields.setdefault("maintenance_type", "")
    raw_fields.setdefault("industry", "")
    raw_fields.setdefault("contract_name", "")
    raw_dict["fields"] = raw_fields

    try:
        validated_data = UserRequestIntent(**raw_dict)
        result = validated_data.model_dump()
        result.setdefault("fields", {})
        result["fields"].update({
            "vendor_name": raw_fields.get("vendor_name", ""),
            "service_scope": raw_fields.get("service_scope", ""),
            "maintenance_type": raw_fields.get("maintenance_type", ""),
            "industry": raw_fields.get("industry", ""),
            "contract_name": raw_fields.get("contract_name", ""),
        })
        return result
    except Exception as e:
        logging.error(f"意圖解析失敗，回傳預設值: {e}")
        fallback = UserRequestIntent().model_dump()
        fallback.setdefault("fields", {})
        fallback["fields"].update({
            "vendor_name": "",
            "service_scope": "",
            "maintenance_type": "",
            "industry": "",
            "contract_name": "",
        })
        return fallback

def _normalize_contract_type_label(label: str) -> str:
    label = str(label or "").strip()
    mapping = {
        "軟體系統開發合約": "開發合約",
        "系統開發合約": "開發合約",
        "軟體開發合約": "開發合約",
        "網頁設計開發合約": "開發合約",
        "資訊系統開發合約": "開發合約",
        "維運合約": "維護合約",
        "軟體維護合約": "維護合約",
        "資訊維護合約": "維護合約",
        "保密合約": "保密協定",
        "NDA": "保密協定",
        "保密協議": "保密協定",
        "保險代理人合約": "保險代理合約",
        "保險代理合約": "保險代理合約",
        "保險商品代理合約": "保險代理合約",
        "保險招攬合約": "保險代理合約",
        "代理招攬合約": "保險代理合約",
        "保險業與保險代理人合約": "保險代理合約",
    }

    if label in mapping:
        return mapping[label]

    for k, v in mapping.items():
        if k in label:
            return v

    return label or "其他"

# RAG 檢索與關聯模板選擇

def guess_draft_contract_type(draft_text: str) -> Dict[str, Any]:
    text = str(draft_text or "")
    mode = detect_contract_mode_from_text(text)

    normalized = normalize_text(text)
    insurance_agency_hits = sum(1 for k in [

    "保險代理人",

    "招攬保險",

    "招攬",

    "保險商品",

    "保戶",

    "要保人",

    "被保險人",

    "保險費",

    "佣酬",

    "佣金",

    "核保",

    "理賠",

    "保險業務員",

    "保險代理人管理規則",

    "保險業招攬及核保理賠辦法",

    "金融消費者保護法",

    ] if k in normalized)

    if insurance_agency_hits >= 3:

        return {

            "primary_type": "保險代理合約",

            "secondary_types": ["保密協定"] if any(k in normalized for k in ["保密", "個人資料", "個資", "機密"]) else [],

            "mode": "保險代理",

            "reason": "保險商品招攬與代理服務",

        }
    has_system = any(k in normalized for k in ["資訊系統", "系統", "平台", "網站", "網頁", "軟體", "程式"])
    has_build = any(k in normalized for k in ["建置", "開發", "設計", "導入", "客製", "實作", "交付", "驗收"])
    has_maintain = any(k in normalized for k in ["維護", "維運", "保固", "修補", "故障排除", "技術支援", "服務維運"])
    has_security = any(k in normalized for k in ["資安", "弱點掃描", "弱點修補", "滲透測試", "資安檢測"])

    if has_system and has_build and has_maintain:
        return {
            "primary_type": "開發合約",
            "secondary_types": ["維護合約"] + (["保密協定"] if "保密" in normalized or "機密" in normalized else []),
            "mode": "混合型",
            "reason": "資訊系統建置兼維運",
        }

    # 系統開發 / 網頁設計 / 軟體建置
    if has_system and has_build:
        secondary = []
        if has_security:
            secondary.append("維護合約")
        return {
            "primary_type": "開發合約",
            "secondary_types": secondary[:3],
            "mode": "開發",
            "reason": "系統建置或開發",
        }

    # 單純維護 / 維運
    if has_system and has_maintain:
        return {
            "primary_type": "維護合約",
            "secondary_types": [],
            "mode": "維護",
            "reason": "系統維護或維運",
        }

    # 保密協定
    if any(k in normalized for k in ["保密協定", "保密合約", "nda", "機密資訊", "不得揭露"]):
        return {
            "primary_type": "保密協定",
            "secondary_types": [],
            "mode": "保密",
            "reason": "保密義務為主",
        }

    prompt = f"""
你是合約分類助理。請判斷以下合約草稿的主要類型與次要類型。
只能輸出 JSON，不要輸出其他文字。

可用類型請盡量從以下挑選：
["維護合約", "開發合約", "保險代理合約", "保密協定", "採購合約", "租賃合約", "委任合約", "承攬合約", "其他"]

分類規則：
1. 若內容大量涉及保險代理人、保險商品、招攬、保戶、要保人、被保險人、保險費、佣酬、核保、理賠、保險業務員或保險代理人管理規則，應優先判斷為「保險代理合約」。
2. 若內容同時包含資訊系統建置、開發、交付、驗收，以及維護、維運、修補、技術支援，應優先判斷為「開發合約」，並將「維護合約」列為 secondary_types。
3. 不要只因為有承攬、建置、交付就泛稱為承攬合約。
4. 企業資訊系統、軟體、網頁、平台相關合約，應優先在「開發合約」或「維護合約」中判斷；但若資訊系統只是保險代理後勤作業的一部分，仍應判斷為「保險代理合約」。

請輸出：
{{
  "primary_type": "主要類型",
  "secondary_types": ["次要類型1", "次要類型2"],
  "reason": "20字內簡短理由"
}}

草稿內容：
{text[:1800]}
"""
    try:
        obj = ollama_json(prompt)
        primary = _normalize_contract_type_label(obj.get("primary_type", "其他"))
        secondary = obj.get("secondary_types", [])
        if not isinstance(secondary, list):
            secondary = []

        secondary = [
            _normalize_contract_type_label(x)
            for x in secondary
            if str(x).strip()
        ]
        secondary = [x for x in secondary if x and x != primary]

        # 防止資訊系統合約被 LLM 泛化成承攬合約
        if primary == "承攬合約" and has_system:
            if has_build and has_maintain:
                primary = "開發合約"
                if "維護合約" not in secondary:
                    secondary.insert(0, "維護合約")
                mode = "混合型"
            elif has_build:
                primary = "開發合約"
                mode = "開發"
            elif has_maintain:
                primary = "維護合約"
                mode = "維護"

        if mode == "混合型":
            if has_maintain and "維護合約" not in [primary] + secondary:
                secondary.append("維護合約")
            if has_build and "開發合約" not in [primary] + secondary:
                secondary.append("開發合約")
            if any(k in normalized for k in ["保密", "機密", "揭露", "開源", "github"]) and "保密協定" not in [primary] + secondary:
                secondary.append("保密協定")

        secondary = [x for x in secondary if x and x != primary]
        secondary = list(dict.fromkeys(secondary))[:3]

        return {
            "primary_type": primary or "其他",
            "secondary_types": secondary,
            "mode": mode,
            "reason": str(obj.get("reason", "")).strip(),
        }

    except Exception:
        fallback_primary = "其他"
        fallback_secondary = []

        if has_system and has_build and has_maintain:
            fallback_primary = "開發合約"
            fallback_secondary = ["維護合約"]
            mode = "混合型"
        elif has_system and has_build:
            fallback_primary = "開發合約"
            mode = "開發"
        elif has_system and has_maintain:
            fallback_primary = "維護合約"
            mode = "維護"

        return {
            "primary_type": fallback_primary,
            "secondary_types": list(dict.fromkeys(fallback_secondary))[:3],
            "mode": mode,
            "reason": "",
        }

def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _safe_topics(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_topic_name(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [normalize_topic_name(x) for x in value.split(",") if x.strip()]
    return []
def _chunk_topics_list(chunk: Dict[str, Any]) -> List[str]:
    raw = chunk.get("topics", [])
    if isinstance(raw, list):
        return [normalize_topic_name(str(x)) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [normalize_topic_name(str(x)) for x in raw.split(",") if str(x).strip()]
    return []



def _chunk_label(chunk: Dict[str, Any]) -> str:
    article_no = _safe_str(chunk.get("article_no"))
    article_title = _safe_str(chunk.get("article_title") or chunk.get("title"))
    parent_key = _safe_str(chunk.get("parent_article_key"))
    chunk_index = chunk.get("chunk_index")
    parts = []
    if article_no:
        parts.append(article_no)
    elif parent_key:
        parts.append(parent_key)
    if article_title:
        parts.append(article_title)
    if chunk_index not in [None, "", 0]:
        parts.append(f"片段{chunk_index}")
    return "｜".join(parts) if parts else "未標示條文"


# ===== 新增輔助函式 =====
def _article_display_label(article: Dict[str, Any], fallback_index: Optional[int] = None) -> str:
    """產生穩定的草稿條文顯示名稱，避免 issue 標題與展開原文錯配。"""
    article_no = _safe_str(article.get("article_no"))
    article_title = _safe_str(article.get("title") or article.get("article_title"))

    if article_no and article_title:
        return f"{article_no}：{article_title}"
    if article_no:
        return article_no
    if article_title:
        return article_title
    if fallback_index is not None:
        return f"第 {fallback_index} 條"
    return "未標示條文"


def _sanitize_issues_for_article(
    issues: Any,
    article: Dict[str, Any],
    article_key: str,
    fallback_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(issues, list):
        return []

    article_label = _article_display_label(article, fallback_index)
    article_content = _safe_str(article.get("content"))
    article_no = _safe_str(article.get("article_no"))
    article_title = _safe_str(article.get("title") or article.get("article_title"))
    article_topics = [normalize_topic_name(x) for x in article.get("topics", []) or [] if str(x).strip()]
    clause_type = _safe_str(article.get("clause_type"))

    sanitized: List[Dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue

        # 完全沒有分析與建議的 issue 不呈現，避免空卡片。
        if not any(_safe_str(issue.get(k)) for k in ["analysis", "suggestion", "adjusted_clause", "template_snippet"]):
            continue

        item = dict(issue)
        item["article_key"] = article_key
        item["clause"] = article_label
        item["draft_text"] = article_content
        item["draft_article_no"] = article_no
        item["draft_article_title"] = article_title
        item["draft_clause_type"] = clause_type
        item["draft_topics"] = article_topics

        if item.get("issue_topic"):
            item["issue_topic"] = normalize_topic_name(item.get("issue_topic"))

        # 如果 LLM 沒給 source，至少標示為系統審查結果；後續 normalize 階段仍會再補來源。
        if not _safe_str(item.get("source")):
            item["source"] = "系統逐條審查"

        sanitized.append(item)

    return sanitized

def _build_article_evidence_packet(
    article: Dict[str, Any],
    candidate_chunks: List[Dict[str, Any]],
) -> str:
    article_label = _article_display_label(article)
    article_content = _safe_str(article.get("content"))
    article_topics = "、".join([
        normalize_topic_name(x)
        for x in article.get("topics", []) or []
        if str(x).strip()
    ]) or "未明確標示"
    clause_type = _safe_str(article.get("clause_type")) or "未分類"

    parts = [
        "## 原始條文脈絡（廠商草稿）",
        f"條文定位：{article_label}",
        f"條款類型：{clause_type}",
        f"草稿主題：{article_topics}",
        "草稿原文：",
        article_content or "（草稿條文內容為空）",
        "",
        "## 知識庫參考依據（歷史合約 / 法遵規範）",
    ]

    if not candidate_chunks:
        parts.append("檢索內容中未含可供比對的歷史基準或法遵片段。")
        return "\n".join(parts)

    for i, chunk in enumerate(candidate_chunks, start=1):
        fname = _safe_str(chunk.get("file_name")) or "未知檔案"
        topics = "、".join(_chunk_topics_list(chunk)) or _safe_str(chunk.get("topics_text")) or "一般條款"
        label = _chunk_label(chunk)
        contract_type = _safe_str(chunk.get("contract_type")) or "未分類"
        template_role = _safe_str(chunk.get("template_role")) or "歷史基準與規範"
        content = _safe_str(chunk.get("content"))

        parts.append(
            f"### 依據 {i}\n"
            f"來源檔名：{fname}\n"
            f"資料角色：{template_role}\n"
            f"合約 / 規範類型：{contract_type}\n"
            f"對應主題：{topics}\n"
            f"條文定位：{label}\n"
            f"依據原文：{content}"
        )

    return "\n\n".join(parts)


def _has_strong_evidence_for_issue(
    article: Dict[str, Any],
    candidate_chunks: List[Dict[str, Any]],
) -> bool:
    content = _safe_str(article.get("content"))
    if not content:
        return False

    if candidate_chunks:
        return True

    for triggers in list(CRITICAL_RISK_TRIGGERS.values()) + list(HIGH_RISK_TRIGGERS.values()):
        if any(t and t in content for t in triggers):
            return True

    return False


def _build_retrieval_filters(
    draft_type: Optional[str] = None,
    vendor_name: Optional[str] = None,
    system_name: Optional[str] = None,
    service_scope: Optional[str] = None,
    maintenance_type: Optional[str] = None,
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if _safe_str(draft_type):
        filters["contract_type"] = _safe_str(draft_type)
    if _safe_str(vendor_name):
        filters["vendor_name"] = _safe_str(vendor_name)
    if _safe_str(system_name):
        filters["system_name"] = _safe_str(system_name)
    if _safe_str(service_scope):
        filters["service_scope"] = _safe_str(service_scope)
    if _safe_str(maintenance_type):
        filters["maintenance_type"] = _safe_str(maintenance_type)
    return filters



def _normalize_template_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    doc_id = item.get("doc_id")
    if not doc_id:
        return None
    normalized = {
        "doc_id": doc_id,
        "file_name": item.get("file_name", "未知檔案"),
        "contract_type": item.get("contract_type", "其他"),
        "summary": item.get("summary", ""),
        "source_text": item.get("source_text") or item.get("content") or "",
        "keywords": item.get("keywords", []) if isinstance(item.get("keywords"), list) else [],
        "core_topics": _safe_topics(item.get("core_topics", [])),
        "vendor_name": _safe_str(item.get("vendor_name")),
        "system_name": _safe_str(item.get("system_name")),
        "service_scope": _safe_str(item.get("service_scope")),
        "maintenance_type": _safe_str(item.get("maintenance_type")),
    }
    return normalized



def find_related_historical_templates(
    draft_text: str,
    draft_type: Optional[str] = None,
    vendor_name: Optional[str] = None,
    system_name: Optional[str] = None,
    service_scope: Optional[str] = None,
    maintenance_type: Optional[str] = None,
    include_all_related: bool = True,
    max_results: int = 100,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    filters = _build_retrieval_filters(
        draft_type=draft_type,
        vendor_name=vendor_name,
        system_name=system_name,
        service_scope=service_scope,
        maintenance_type=maintenance_type,
    )

    # 1) 結構化檢索：優先抓同廠商 / 同系統 / 同服務範圍 / 同維護類型
    try:
        if filters:
            structured_hits = search_templates_sql(
                query_text=draft_text[:1200],
                filters=filters,
                limit=max_results,
            ) or []
            for item in structured_hits:
                normalized = _normalize_template_record(item)
                if normalized:
                    merged[normalized["doc_id"]] = normalized
    except TypeError:
        try:
            structured_hits = search_templates_sql(draft_text[:1200], filters=filters, limit=max_results) or []
            for item in structured_hits:
                normalized = _normalize_template_record(item)
                if normalized:
                    merged[normalized["doc_id"]] = normalized
        except Exception as e:
            logging.warning(f"結構化模板檢索失敗：{e}")
    except Exception as e:
        logging.warning(f"結構化模板檢索失敗：{e}")

    # 2) 額外以單欄位弱條件補抓，避免資料表沒有完整索引時漏抓
    loose_conditions = [
        ("vendor_name", vendor_name),
        ("system_name", system_name),
        ("service_scope", service_scope),
        ("maintenance_type", maintenance_type),
        ("contract_type", draft_type),
    ]
    for key, value in loose_conditions:
        if not _safe_str(value):
            continue
        try:
            hits = search_templates_sql(
                query_text=_safe_str(value),
                filters={key: _safe_str(value)},
                limit=max_results,
            ) or []
            for item in hits:
                normalized = _normalize_template_record(item)
                if normalized:
                    merged[normalized["doc_id"]] = normalized
        except Exception:
            continue

    # 3) 向量/全文檢索補足語意相關文件
    try:
        semantic_hits = query_templates_fulltext(draft_text, n_results=max_results) or []
        for item in semantic_hits:
            normalized = _normalize_template_record(item)
            if normalized:
                merged.setdefault(normalized["doc_id"], normalized)
    except Exception as e:
        logging.warning(f"語意模板檢索失敗：{e}")

    # 4) include_all_related=False 時仍保留較嚴格的一批；True 則盡量保留全集
    results = list(merged.values())
    if not include_all_related:
        return results[: max_results]
    return results



def select_review_templates(
    draft_text: str,
    articles: List[Dict[str, Any]],
    draft_type: Optional[str] = None,
    vendor_name: Optional[str] = None,
    system_name: Optional[str] = None,
    service_scope: Optional[str] = None,
    maintenance_type: Optional[str] = None,
    include_all_related: bool = True,
    max_candidates: int = 50,
) -> List[Dict[str, Any]]:
    guessed = guess_draft_contract_type(draft_text)

    if isinstance(draft_type, str) and draft_type.strip():
        primary_type = _normalize_contract_type_label(draft_type)
        secondary_types = guessed.get("secondary_types", [])
        mode = guessed.get("mode", detect_contract_mode_from_text(draft_text))
    else:
        primary_type = guessed.get("primary_type", "其他")
        secondary_types = guessed.get("secondary_types", [])
        mode = guessed.get("mode", detect_contract_mode_from_text(draft_text))

    logging.info(
        f"LLM 判斷草稿類型為：primary={primary_type}, secondary={secondary_types}, mode={mode}"
    )

    requested_types = [primary_type] + [x for x in secondary_types if x != primary_type]
    requested_types = [x for x in requested_types if x and x != "其他"]
    if not requested_types:
        requested_types = ["其他"]

    merged_candidates: Dict[str, Dict[str, Any]] = {}

    for ctype in requested_types:
        hits = find_related_historical_templates(
            draft_text=draft_text,
            draft_type=ctype,
            vendor_name=vendor_name,
            system_name=system_name,
            service_scope=service_scope,
            maintenance_type=maintenance_type,
            include_all_related=include_all_related,
            max_results=max_candidates,
        )
        for item in hits:
            doc_id = item.get("doc_id")
            if doc_id:
                merged_candidates[doc_id] = item

    if not merged_candidates:
        logging.info("找不到結構化/語意歷史模板，退回全庫搜尋")
        fallback = query_templates_fulltext(draft_text, n_results=max(12, max_candidates)) or []
        for item in fallback:
            normalized = _normalize_template_record(item)
            if normalized and normalized.get("doc_id"):
                merged_candidates[normalized["doc_id"]] = normalized

    candidates = list(merged_candidates.values())

    draft_topics = detect_topics(draft_text)
    article_topic_union: set = set()
    for a in articles:
        article_topic_union.update(a.get("topics", []))

    ranked = []
    seen_doc_ids: set = set()
    seen_file_names: set = set()

    for c in candidates:
        doc_id = c.get("doc_id")
        file_name = c.get("file_name", "")
        if doc_id in seen_doc_ids:
            continue
        if file_name and file_name in seen_file_names:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        if file_name:
            seen_file_names.add(file_name)

        real_template_topics = c.get("core_topics") or detect_topics(c.get("source_text", ""))
        c["core_topics"] = real_template_topics

        ctype = _normalize_contract_type_label(c.get("contract_type", "其他"))

        text_block = " ".join([
            c.get("file_name", ""),
            ctype,
            c.get("summary", ""),
            " ".join(c.get("keywords", [])),
            " ".join(real_template_topics),
            c.get("vendor_name", ""),
            c.get("system_name", ""),
            c.get("service_scope", ""),
            c.get("maintenance_type", ""),
        ])

        score = lexical_score(text_block, draft_text[:1200])
        score += score_topic_overlap(draft_topics, real_template_topics) * 4
        score += score_topic_overlap(list(article_topic_union), real_template_topics) * 3

        # 類型加權
        if ctype == primary_type:
            score += 18
        elif ctype in secondary_types:
            score += 10

        # 混合型合約時，維護 / 開發 / 保密協定都可加權
        if mode == "混合型" and ctype in ["維護合約", "開發合約", "保密協定"]:
            score += 4

        # 結構化欄位加權
        if vendor_name and c.get("vendor_name") == vendor_name:
            score += 15
        if system_name and c.get("system_name") == system_name:
            score += 12
        if service_scope and c.get("service_scope") == service_scope:
            score += 10
        if maintenance_type and c.get("maintenance_type") == maintenance_type:
            score += 8

        ranked.append((score, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked]

def build_target_queries(
    draft_text: str,
    articles: List[Dict[str, Any]],
    selected_templates: List[Dict[str, Any]],
) -> List[str]:

    queries: List[str] = []
    queries.append(_truncate(draft_text, MAX_QUERY_CHARS, ""))

    for article in articles:
        content = article.get("content", "")
        if not content:
            continue
        article_no = (article.get("article_no") or "").strip()
        title = (article.get("title") or "").strip()
        header = " ".join(x for x in [article_no, title] if x)
        body = _truncate(content, 700, "")
        q = f"{header} {body}".strip() if header else body
        queries.append(q)

    dedup: List[str] = []
    seen: set = set()
    for q in queries:
        q = normalize_text(q)
        if q and q not in seen:
            seen.add(q)
            dedup.append(q)
    return dedup



def search_relevant_templates(
    draft_text: str,
    top_k: int = 5,
    draft_type: Optional[str] = None,
    vendor_name: Optional[str] = None,
    system_name: Optional[str] = None,
    service_scope: Optional[str] = None,
    maintenance_type: Optional[str] = None,
    include_all_related: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    articles = split_draft_into_articles(draft_text)

    selected_templates = select_review_templates(
        draft_text,
        articles,
        draft_type=draft_type,
        vendor_name=vendor_name,
        system_name=system_name,
        service_scope=service_scope,
        maintenance_type=maintenance_type,
        include_all_related=include_all_related,
        max_candidates=max(20, top_k * 8),
    )

    doc_ids = [x["doc_id"] for x in selected_templates if x.get("doc_id")]

    article_topic_fallbacks: List[str] = []
    for article in articles:
        for topic in article.get("topics", []) or []:
            normalized_topic = normalize_topic_name(topic)
            if normalized_topic and normalized_topic not in article_topic_fallbacks:
                article_topic_fallbacks.append(normalized_topic)

    all_chunks: List[Dict[str, Any]] = []
    seen: set = set()

    for q in build_target_queries(draft_text, articles, selected_templates):
        detected_topics = [
            normalize_topic_name(topic)
            for topic in detect_topics(q)
            if str(topic).strip()
        ]

        target_topic = ""
        if detected_topics:
            target_topic = detected_topics[0]
        elif article_topic_fallbacks:
            target_topic = article_topic_fallbacks[0]

        refs = query_template_chunks_by_query(
            q,
            doc_ids,
            n_results=8,
            target_topic=target_topic,
        )

        for r in refs:
            key = (r["doc_id"], r["content"][:150])
            if key not in seen:
                seen.add(key)
                all_chunks.append(r)

    return selected_templates, all_chunks[:40], articles



def search_template_chunks_for_article(
    article: Dict[str, Any],
    selected_templates: List[Dict[str, Any]],
    n_results: int = 4,
) -> List[Dict[str, Any]]:
    doc_ids = [x["doc_id"] for x in selected_templates[:12] if x.get("doc_id")]
    article_text = article.get("content", "")
    article_topics = [
        normalize_topic_name(t)
        for t in article.get("topics", []) or []
        if str(t).strip()
    ]

    article_no = _safe_str(article.get("article_no"))
    article_title = _safe_str(article.get("title"))
    article_title_norm = normalize_text(article_title)
    article_clause_type = _safe_str(article.get("clause_type"))
    header = " ".join(x for x in [article_no, article_title] if x)

    target_topic = article_topics[0] if article_topics else ""
    if not target_topic and article_clause_type:
        target_topic = normalize_topic_name(article_clause_type)

    def _extract_keywords(text: str) -> List[str]:
        text = normalize_text(text)
        raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
        stopwords = {
            "甲方", "乙方", "雙方", "條款", "本條", "本合約", "約定",
            "應", "應於", "以及", "相關", "內容", "方式", "廠商", "草稿",
            "第一條", "第二條", "第三條", "第四條", "第五條",
            "第六條", "第七條", "第八條", "第九條", "第十條",
        }

        out = []
        seen_kw = set()

        for tok in raw_tokens:
            if tok in stopwords:
                continue
            if tok not in seen_kw:
                seen_kw.add(tok)
                out.append(tok)

        legal_terms = [
            "資安檢測", "弱點掃描", "滲透測試", "管轄法院", "準據法", "爭議處置",
            "維護人力", "維護時間", "違約金", "損害賠償", "保密義務", "個資保護",
            "智慧財產權", "交付驗收", "驗收", "付款", "終止", "解除",
            "備份", "災難復原", "事件通報", "異常回覆", "修復時限",

            # 保險代理合約
            "保險代理人", "招攬", "保險商品", "保戶", "要保人", "被保險人",
            "保險費", "佣酬", "佣金", "核保", "理賠", "保險業務員",
            "廣告文宣", "洗錢防制", "打擊資恐", "複委託", "個人資料",
            "利益衝突", "績效考核", "授權範圍", "文件轉送", "招攬義務",
            "廣告", "文宣", "終止事由",
        ]

        for topic in article_topics:
            for term in TOPIC_KEYWORDS.get(topic, [])[:8]:
                n_term = normalize_text(term)
                if n_term and n_term not in seen_kw:
                    seen_kw.add(n_term)
                    out.insert(0, n_term)

        for term in legal_terms:
            n_term = normalize_text(term)
            if n_term in text and n_term not in seen_kw:
                seen_kw.add(n_term)
                out.insert(0, n_term)

        return out[:16]

    query_keywords = _extract_keywords(article_text)

    queries = []

    if header:
        queries.append(f"{header} {article_text[:240]}")

    queries.append(article_text[:500])

    for topic in article_topics[:2]:
        kw_list = TOPIC_KEYWORDS.get(topic, [])
        if kw_list:
            queries.append(f"{topic} {header} " + " ".join(kw_list[:6]))

    queries = [q for q in queries if q][:3]

    seen: set = set()
    out: List[Dict[str, Any]] = []

    for q in queries:
        refs = query_template_chunks_by_query(
            q,
            doc_ids,
            n_results=8,
            target_topic=target_topic,
        )

        for r in refs:
            key = (r["doc_id"], r["content"][:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)

    scored = []

    for r in out:
        r_content = normalize_text(r.get("content", ""))
        chunk_topics = _chunk_topics_list(r)
        r_article_no = normalize_text(str(r.get("article_no", "") or ""))
        r_article_title = normalize_text(str(r.get("article_title", "") or ""))
        r_clause_type = _safe_str(r.get("clause_type"))

        score = 0

        # topic overlap 是最重要的，權重提高
        score += score_topic_overlap(article_topics, chunk_topics) * 12
        score += lexical_score(r_content, article_text[:220]) * 2

        r_label = normalize_text(str(r.get("chunk_label", "") or ""))
        r_topics_text = normalize_text(" ".join(chunk_topics))
        searchable = " ".join([
            r_content,
            r_article_title,
            r_clause_type,
            r_label,
            r_topics_text,
        ])

        keyword_hits = sum(1 for kw in query_keywords if kw in searchable)
        score += keyword_hits * 4

        title_hits = sum(
            1 for kw in query_keywords
            if kw in r_article_title or kw in r_label
        )
        topic_hits = sum(
            1 for kw in query_keywords
            if kw in r_topics_text or kw in r_clause_type
        )

        score += title_hits * 6
        score += topic_hits * 10

        if article_title_norm and article_title_norm in r_article_title:
            score += 20

        if article_clause_type and r_clause_type and article_clause_type == r_clause_type:
            score += 35

        # 本條 topic keywords 命中，額外加權
        for t in article_topics:
            if any(k in r_content for k in TOPIC_KEYWORDS.get(t, [])):
                score += 15

        # 若 target_topic 明確出現在 chunk topic / title / label，強加權
        if target_topic:
            target_terms = [target_topic]
            target_terms.extend(TOPIC_ALIAS.get(target_topic, []))
            target_terms.extend(TOPIC_KEYWORDS.get(target_topic, [])[:6])

            normalized_target_terms = [
                normalize_text(x)
                for x in target_terms
                if str(x).strip()
            ]

            if any(term and term in r_topics_text for term in normalized_target_terms):
                score += 30

            if any(term and (term in r_article_title or term in r_label) for term in normalized_target_terms):
                score += 20

            if any(term and term in r_content for term in normalized_target_terms):
                score += 10

        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:n_results]]
_LAW_REF_RE = re.compile(
    r"[\u4e00-\u9fa5]{1,24}法第\s*[\d一二三四五六七八九十百千]+條(?:之\d+)?(?:第[\d一二三四五六七八九十]+項)?"
)
_MD_EMPHASIS_RE = re.compile(r"\*{1,3}")
_SELF_CHECK_MARKERS = (
    "**【AI 自檢】**",
    "【AI 自檢】",
    "**【免責聲明】**",
    "【免責聲明】",
)


def clean_issue_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "n/a", "na"}:
        return ""
    if text in {"未提供", "無提供", "無", "null", "None"}:
        return ""

    for marker in _SELF_CHECK_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].strip()

    text = _MD_EMPHASIS_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_law_refs(*values: Any) -> List[str]:
    joined = "\n".join(clean_issue_text(v) for v in values if v is not None)
    if not joined:
        return []
    return list(dict.fromkeys(_LAW_REF_RE.findall(joined)))


def normalize_risk_level(value: Any, default: str = "medium") -> str:
    text = str(value or "").strip().lower()
    raw = normalize_text(str(value or ""))

    if text in {"critical", "crit"} or any(k in raw for k in ["重大風險", "嚴重風險", "重大", "嚴重"]):
        return "critical"
    if text == "high" or any(k in raw for k in ["高風險", "高"]):
        return "high"
    if text == "medium" or any(k in raw for k in ["中風險", "一般風險", "中"]):
        return "medium"
    if text == "low" or any(k in raw for k in ["低風險", "低"]):
        return "low"

    return default


def risk_label_from_level(level: Any) -> str:
    level = normalize_risk_level(level)
    return {
        "critical": "重大風險",
        "high": "高風險",
        "medium": "中風險",
        "low": "低風險",
    }.get(level, "中風險")


def normalize_issue_card(
    issue: Dict[str, Any],
    *,
    index: int = 0,
    default_group: str = "general",
) -> Dict[str, Any]:
    item = dict(issue or {})

    raw_risk = item.get("risk_level") or item.get("risk") or item.get("riskLabel") or item.get("risk_label")
    risk_level = normalize_risk_level(
        raw_risk,
        default="high" if default_group == "major" else "medium",
    )
    risk_label = clean_issue_text(item.get("risk_label") or item.get("riskLabel")) or risk_label_from_level(risk_level)

    article_no = clean_issue_text(item.get("draft_article_no") or item.get("article_no") or item.get("article"))
    article_title = clean_issue_text(item.get("draft_article_title") or item.get("article_title") or item.get("title"))
    clause = clean_issue_text(item.get("clause"))

    if not clause:
        if article_no and article_title:
            clause = f"{article_no}：{article_title}"
        elif article_no:
            clause = article_no
        elif article_title:
            clause = article_title
        else:
            clause = "未標示條文"

    issue_topic = normalize_topic_name(
        item.get("issue_topic") or item.get("topic") or item.get("requirement") or "一般風險"
    )
    issue_type = clean_issue_text(item.get("type") or item.get("issue_type")) or "deviation"

    draft_text = clean_issue_text(item.get("draft_text") or item.get("quotedText") or item.get("found_clause"))
    template_snippet = clean_issue_text(item.get("template_snippet") or item.get("template_basis") or item.get("basis"))
    analysis = clean_issue_text(item.get("analysis") or item.get("reasoning") or item.get("gap_description") or item.get("description"))
    suggestion = clean_issue_text(item.get("suggestion") or item.get("修改建議"))
    adjusted_clause = clean_issue_text(item.get("adjusted_clause") or item.get("suggested_addition") or item.get("alternative_clause"))
    negotiation_notes = clean_issue_text(item.get("negotiation_notes") or item.get("cost_bearing_suggestion"))
    source = clean_issue_text(item.get("source") or item.get("file_name")) or "系統審查結果"

    law_refs = item.get("law_refs") or item.get("lawRefs")
    if isinstance(law_refs, list):
        law_refs = [clean_issue_text(x) for x in law_refs if clean_issue_text(x)]
    else:
        law_refs = []

    law_refs = list(dict.fromkeys(
        law_refs + extract_law_refs(template_snippet, analysis, suggestion, adjusted_clause)
    ))

    card_id = clean_issue_text(item.get("id")) or f"{default_group}-risk-{index}"

    normalized = {
        **item,
        "id": card_id,
        "group": default_group,
        "article_key": clean_issue_text(item.get("article_key")) or clean_issue_text(item.get("parent_article_key")),
        "article_no": article_no,
        "article_title": article_title,
        "clause": clause,
        "issue_topic": issue_topic,
        "type": issue_type,
        "risk": risk_level,
        "risk_level": risk_level,
        "risk_label": risk_label,
        "draft_text": draft_text,
        "template_snippet": template_snippet,
        "analysis": analysis,
        "suggestion": suggestion,
        "adjusted_clause": adjusted_clause,
        "negotiation_notes": negotiation_notes,
        "source": source,
        "law_refs": law_refs,
    }

    # 前端相容欄位
    normalized["riskLevel"] = risk_level
    normalized["riskLabel"] = risk_label
    normalized["lawRefs"] = law_refs
    normalized["quotedText"] = draft_text
    normalized["reasoning"] = analysis

    return normalized


def normalize_issue_cards(
    issues: List[Dict[str, Any]],
    *,
    default_group: str = "general",
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()

    for idx, issue in enumerate(issues or []):
        if not isinstance(issue, dict):
            continue

        try:
            card = normalize_issue_card(issue, index=idx, default_group=default_group)
        except Exception as exc:
            logging.warning(f"風險卡正規化失敗，略過單筆 issue：{exc}")
            continue

        if not any(card.get(k) for k in ["analysis", "suggestion", "adjusted_clause", "template_snippet", "draft_text"]):
            continue

        dedup_key = (
            card.get("article_key", ""),
            card.get("clause", ""),
            card.get("issue_topic", ""),
            card.get("analysis", "")[:120],
        )

        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        normalized.append(card)

    return normalized

# 審查邏輯
def _risk_level_rank(risk: str) -> int:
    risk = str(risk or "").lower()
    if risk == "critical":
        return 4
    if risk == "high":
        return 3
    if risk == "medium":
        return 2
    if risk == "low":
        return 1
    return 0


def build_trigger_issues_from_articles(
    articles: List[Dict[str, Any]],
    draft_type: str = "其他",
) -> List[Dict[str, Any]]:

    rules = load_rules()
    if not rules:
        logging.warning("Rule engine 載入 0 條規則 — 紅線保底層失效，請檢查 rules/ 目錄")
        return []

    draft_type = _normalize_contract_type_label(draft_type)

    all_issues: List[Dict[str, Any]] = []
    for idx, article in enumerate(articles, start=1):
        content = str(article.get("content", "") or "").strip()
        if not content:
            continue

        # 規則一次跑完，取得所有命中
        hits = evaluate_all_rules(content, draft_type, rules=rules)
        if not hits:
            continue

        # 每個命中組成一個 issue，後續合併
        for hit in hits:
            issue = render_issue_from_hit(hit, article, idx)
            all_issues.append(issue)

    # 同一條文同 topic 合併（取最高 risk + 合併命中字眼）
    return merge_issues_by_topic(all_issues)

def llm_review_single_article(
    article: Dict[str, Any],
    article_key: str,
    candidate_chunks: List[Dict[str, Any]],
    draft_type: str = "其他",
) -> Dict[str, Any]:
    chunk_block = []
    source_names = []

    for i, c in enumerate(candidate_chunks, start=1):
        fname = c["file_name"]
        source_names.append(fname)
        topics = "、".join(_chunk_topics_list(c)) or "一般條款"
        label = _chunk_label(c)
        # 截斷單個 chunk，避免長依據把 prompt 撐爆
        snippet = _truncate(c["content"], MAX_CHUNK_CHARS)
        chunk_block.append(
            f"【依據 {i}】\n"
            f"對應主題：{topics}\n"
            f"檔名：{fname}\n"
            f"條文定位：{label}\n"
            f"內容：{snippet}"
        )

    allowed_sources = "、".join(sorted(set(source_names))) if source_names else "無"
    evidence_packet = _build_article_evidence_packet(article, candidate_chunks)

    if not _has_strong_evidence_for_issue(article, candidate_chunks):
        return {
            "major_issues": [],
            "general_issues": [],
        }

    draft_content = _truncate(article.get("content", ""), MAX_ARTICLE_CHARS)
    triggered_rules = []

    # 依合約類型啟用對應紅線規則
    # 避免保險代理合約誤套開發 / 維護 / 資安規則。
    normalized_draft_type = _normalize_contract_type_label(draft_type)
    enabled_topics = set(CONTRACT_TYPE_RULESET.get(normalized_draft_type, []))

    for t_category, triggers in CRITICAL_RISK_TRIGGERS.items():
        normalized_topic = normalize_topic_name(t_category)

        if enabled_topics and normalized_topic not in enabled_topics:
            continue

        for trigger in triggers:
            if trigger and trigger in draft_content:
                triggered_rules.append(
                    f"嚴重違規字眼：「{trigger}」(屬於 {normalized_topic})"
                )

    for t_category, triggers in HIGH_RISK_TRIGGERS.items():
        normalized_topic = normalize_topic_name(t_category)

        if enabled_topics and normalized_topic not in enabled_topics:
            continue

        for trigger in triggers:
            if trigger and trigger in draft_content:
                triggered_rules.append(
                    f"高風險字眼：「{trigger}」(屬於 {normalized_topic})"
                )

    triggered_rules = list(dict.fromkeys(triggered_rules))

    rule_injection = ""
    if triggered_rules:
        rule_injection = (
            "\n\n【🚨 系統強制檢核指令】：\n"
            f"本次合約類型判定為「{normalized_draft_type}」，系統僅啟用該類型適用之企業紅線規則。\n"
            "系統已在以下草稿中偵測到觸發企業紅線的字眼：\n"
            + "、".join(triggered_rules) + "\n"
            "你「必須」將這些紅線字眼獨立列為一個 issue，具體分析該字眼在本合約類型下造成的實質業務、法遵或履約風險，"
            "不可含糊帶過，並給出明確協商建議。絕對不可與其他問題混為一談，也不可套用不屬於本合約類型的風險場景。\n"
        )

    prompt = f"""
你是企業合規與法務審查 AI，立場固定站在甲方。
本次採用「調查員 / 判官」兩階段邏輯：
（一）調查員已先將草稿條文與歷史 / 法遵依據整理成【審查證據包】；
（二）你現在是判官，只能依據【審查證據包】與【系統強制檢核指令】判斷是否存在具體風險。
{rule_injection}

嚴格限制：
1. 只能輸出 JSON，不可輸出多餘的文字。
2. 判斷標準必須完全依據【審查證據包】中的「原始條文脈絡」、「知識庫參考依據」或【系統強制檢核指令】。絕不可憑空捏造。
3. 若【審查證據包】明確標示「檢索內容中未含可供比對的歷史基準或法遵片段」，且沒有系統強制檢核指令，major_issues 與 general_issues 必須回傳空陣列。
4. issue_topic 欄位必須填寫風險主題；若知識庫參考依據有標示「對應主題」，優先沿用該主題。
5. clause 欄位必須填寫【審查證據包】中「原始條文脈絡（廠商草稿）」的條文定位，不得填入歷史基準片段的條號、標題或主題名稱。
6. draft_text 必須等於【審查證據包】中「草稿原文」的完整文字，不得改寫、不得引用歷史基準內容。
7. analysis（合規落差分析）：具體說明廠商草稿與企業歷史基準的差異，以及對甲方的潛在合約、法遵或營運風險。
8. suggestion（建議修正與協商方案）：直接依據基準的具體標準給出修改方向。並請針對「若廠商無法配合」，主動提議實務上的替代協商方案。
9. adjusted_clause（建議修改後條文）：根據歷史基準原文，給出可直接貼回合約的具體修改後條文（100字以內）。若完全符合基準則填「符合，無需修改」。
10. negotiation_notes（協商備忘）：若廠商拒絕修改，可接受的最低底線條件，及建議的折衷方案（50字以內）。
11. template_snippet（基準原文）：請從【審查證據包】的「知識庫參考依據」中，精準擷取最相關的「依據原文」字句。
12. source 欄位必須填入對應的真實檔名：{allowed_sources}。絕不可留空。
13. 若廠商草稿條文已實質涵蓋歷史基準要求，且沒有明確偏離、衝突、紅線字眼或責任弱化，major_issues 與 general_issues 必須回傳空陣列。
14. 不得為了產生報告而硬找問題；沒有具體差異證據時，不得輸出 issue。
15. 若只是用語不同但義務內容相同，視為符合，不得列為偏離。
16. 僅在以下情況輸出 issue：
    - 草稿明確排除、減輕或轉嫁乙方義務；
    - 草稿與歷史基準或章則規範有實質衝突；
    - 草稿缺少該條款必要核心義務；
    - 草稿命中系統強制檢核紅線字眼。
17. source 欄位只能使用【審查證據包】中「來源檔名」列出的真實檔名，或在系統強制檢核情境下使用「系統內建企業紅線規則」。
18. 若沒有問題，請輸出：
{{
  "major_issues": [],
  "general_issues": []
}}

輸出格式：
{{
"major_issues": [
  {{
    "article_key": "{article_key}",
    "clause": "條款名稱",
    "issue_topic": "對應的主題",
    "type": "deviation 或 conflict",
    "risk": "Critical/High/Medium/Low",
    "template_snippet": "(動態生成) 乙方應依歷史基準或章則規範履行必要義務...",
    "analysis": "(動態生成) 草稿與歷史基準或章則規範存在實質落差，可能增加甲方法遵、履約或營運風險。",
    "suggestion": "(動態生成) 建議依歷史基準補明乙方義務、責任分配與違約效果。",
    "adjusted_clause": "(動態生成) 乙方應依本合約及相關法令規範履行必要義務，並配合甲方查核與改善。",
    "negotiation_notes": "(動態生成) 最低底線：不得降低甲方法遵、保戶權益、個資保護、保密、爭議處理或違約救濟保障。",
    "source": "來源檔名.docx"
  }}
],
"general_issues": []
}}

【審查證據包】
{evidence_packet}
"""
    logging.info(f"llm_review_single_article 使用模型：{REVIEW_MODEL}")
    return ollama_json(prompt, model=REVIEW_MODEL, num_ctx=LLM_NUM_CTX)


def infer_missing_topics_from_templates(
    selected_templates: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    found_topics_by_llm: set,
    all_candidate_chunks: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """
    歷史模板缺漏條款推論｜保守上線版

    設計目的：
    1. 不再因為「歷史模板有某 topic」就硬判定草稿缺漏。
    2. 只有在 topic 明顯屬於本次合約必要核心條款，且草稿完全沒有相關文字時，才列為 missing。
    3. 避免正常合約被硬抓出交付驗收、付款價金、智慧財產權等缺漏。
    4. 對保險代理合約加入專用必要 topic，避免誤套開發 / 維護合約缺漏邏輯。
    """
    draft_full_text = normalize_text(
        "\n".join(a.get("content", "") for a in articles)
    )

    draft_topics: set = set()
    for article in articles:
        for topic in article.get("topics", []) or []:
            draft_topics.add(normalize_topic_name(topic))

    all_found = draft_topics.union({
        normalize_topic_name(t) for t in found_topics_by_llm if str(t).strip()
    })

    has_development = any(k in draft_full_text for k in [
        "建置", "開發", "設計", "交付", "驗收", "測試版",
        "原始碼", "程式碼", "系統設計", "開發成果"
    ])

    has_maintenance = any(k in draft_full_text for k in [
        "維護", "維運", "技術支援", "修補", "故障排除",
        "服務時間", "回覆時限", "維護標的"
    ])

    has_confidential = any(k in draft_full_text for k in [
        "保密", "機密", "不得揭露", "第三人", "個人資料", "個資"
    ])

    has_insurance_agency = any(k in draft_full_text for k in [
        "保險代理人",
        "保險商品",
        "招攬",
        "保戶",
        "要保人",
        "被保險人",
        "保險費",
        "佣酬",
        "佣金",
        "核保",
        "理賠",
        "保險業務員",
        "保險代理人管理規則",
        "保險業招攬及核保理賠辦法",
        "金融消費者保護法",
    ])

    development_required_topics = {
        "交付驗收",
        "智慧財產權",
    }

    maintenance_required_topics = {
        "維護標的",
        "維護時間",
        "維護人力",
        "異常回覆時限",
        "修復時限",
    }

    confidential_required_topics = {
        "保密措施",
        "第三人揭露禁止",
        "保密期間",
    }

    insurance_required_topics = {
        "代理人資格",
        "授權範圍",
        "文件轉送期限",
        "招攬義務",
        "廣告文宣控管",
        "佣酬返還",
        "理賠協助",
        "法規遵循與合規", 
        "個人資料保護",
        "複委託監督",
        "洗錢防制與打擊資恐",
        "終止事由",
        "績效考核",
        "利益衝突",
        "保密措施",
    }

    enabled_topics = set()

    if has_development:
        enabled_topics.update(development_required_topics)

    if has_maintenance:
        enabled_topics.update(maintenance_required_topics)

    if has_confidential:
        enabled_topics.update(confidential_required_topics)

    if has_insurance_agency:
        enabled_topics.update(insurance_required_topics)

    # 商務條款只在草稿明顯已有付款、價金、違約責任框架時才檢查
    if any(k in draft_full_text for k in ["價金", "費用", "付款", "報價", "金額", "合約總額"]):
        enabled_topics.add("付款價金")

    if any(k in draft_full_text for k in ["佣酬", "佣金", "服務費", "委辦費用"]):
        enabled_topics.add("佣酬返還")

    if any(k in draft_full_text for k in ["違約", "遲延", "賠償", "損害", "罰則", "裁罰"]):
        enabled_topics.update({"違約金", "損害賠償"})

    if any(k in draft_full_text for k in ["法院", "管轄", "準據法", "法律", "涉訟"]):
        enabled_topics.update({"管轄法院", "準據法"})

    # 短合約時只刪除開發 / 維護 / 商務類容易誤判的 topic；
    # 不刪保險代理合約核心 topic，避免測試草稿被過度放寬。
    if len(draft_full_text) < 800:
        enabled_topics = enabled_topics - {
            "交付驗收",
            "智慧財產權",
            "付款價金",
            "違約金",
            "損害賠償",
            "維護時間",
            "異常回覆時限",
            "修復時限",
        }

    # 若完全沒有判斷出本次合約類型所需 topic，保守回傳空缺漏，避免硬找問題。
    if not enabled_topics:
        return []

    chunk_evidenced: Dict[str, Tuple[str, str]] = {}

    for chunk in all_candidate_chunks:
        chunk_content = normalize_text(chunk.get("content", ""))
        fname = chunk.get("file_name", "")

        if not chunk_content:
            continue

        pattern = r"(第[一二三四五六七八九十百0-9]+條[：:\s].*?)(?=(?:\n?第[一二三四五六七八九十百0-9]+條[：:\s])|$)"
        sub_articles = re.findall(pattern, chunk_content, flags=re.S)

        if not sub_articles:
            sub_articles = [chunk_content]

        for topic, keywords in TOPIC_KEYWORDS.items():
            topic = normalize_topic_name(topic)

            if topic in chunk_evidenced:
                continue

            # 不在 enabled_topics 裡，就不要列為缺漏
            if topic not in enabled_topics:
                continue

            min_hits = TOPIC_MIN_MATCHES.get(topic, 1)
            best_snippet = ""

            for sub_art in sub_articles:
                hits = sum(1 for k in keywords if k and k in sub_art)
                if hits >= min_hits:
                    best_snippet = sub_art.strip()
                    break

            if best_snippet:
                snippet = best_snippet[:400] + ("..." if len(best_snippet) > 400 else "")
                chunk_evidenced[topic] = (fname, snippet)

    def _topic_appears_in_draft(topic: str) -> bool:
        topic = normalize_topic_name(topic)

        aliases = set()
        aliases.add(topic)
        aliases.update(TOPIC_ALIAS.get(topic, []) or [])
        aliases.update(TOPIC_KEYWORDS.get(topic, []) or [])

        normalized_aliases = {
            normalize_text(a)
            for a in aliases
            if str(a).strip()
        }

        return any(alias and alias in draft_full_text for alias in normalized_aliases)

    missing: List[Tuple[str, str, str]] = []

    for topic, (fname, snippet) in chunk_evidenced.items():
        topic = normalize_topic_name(topic)

        if topic in all_found:
            continue

        if _topic_appears_in_draft(topic):
            continue

        missing.append((topic, fname, snippet))

    return missing

# 批次起草所有缺漏條文
def _draft_all_missing_clauses(
    missing_topic_tuples: List[Tuple[str, str, str]]
) -> Dict[str, str]:
    if not missing_topic_tuples:
        return {}

    items_text = "\n".join(
        f"{i+1}. 主題：{topic}\n"
        f"   歷史基準片段（節錄）：{snippet[:240]}"
        for i, (topic, _, snippet) in enumerate(missing_topic_tuples)
    )

    prompt = f"""
你是企業法務條款整理助理。你的任務不是自由發明新條文，而是依照「歷史基準片段」做最小必要改寫。

嚴格規則：
1. 每個主題都要輸出一條補充條文，長度 40~100 字。
2. 廠商單方義務條款可用「乙方應」開頭；但「準據法」、「管轄法院」、「付款價金」、「違約金」、「損害賠償」等雙方約定或法律效果條款，不得硬套「乙方應」句型，應改用「雙方同意」、「本合約」或其他自然法務用語。
3. 優先沿用歷史基準片段中的句型、義務、期間、責任分配。
4. 若歷史片段沒有明確期間、金額、比例、次數，不得自行新增。
5. 不得加入歷史片段中未出現的新流程、新法規名稱、新技術標準。
6. 若歷史基準本身很短，就維持簡短，只做最小改寫讓它更像可直接貼回合約的條文。
7. 只能輸出 JSON，不要任何說明文字。

輸出格式：
{{
  "drafts": [
    {{"topic": "主題名稱", "clause": "乙方應..."}},
    ...
  ]
}}

缺漏主題清單：
{items_text}
"""
    result = ollama_json(prompt)
    drafts = result.get("drafts", [])
    if not isinstance(drafts, list):
        return {}

    out: Dict[str, str] = {}
    for d in drafts:
        if not isinstance(d, dict):
            continue
        topic = str(d.get("topic", "")).strip()
        clause = str(d.get("clause", "")).strip()
        if not topic or not clause:
            continue
        bilateral_topics = {"準據法", "管轄法院", "付款價金", "違約金", "損害賠償"}
        if topic in bilateral_topics:
            clause = clause.lstrip("乙方應").strip()
            if topic == "準據法" and not clause.startswith(("本合約", "雙方同意")):
                clause = f"本合約未盡事宜，{clause}"
            if topic == "管轄法院" and not clause.startswith(("本合約", "雙方同意")):
                clause = f"雙方同意{clause}"
        elif not clause.startswith("乙方應"):
            clause = f"乙方應{clause.lstrip('乙方應')}".strip()
        out[topic] = clause

    return out


def _build_requirement_evidence(
    requirement: str,
    draft_text: str,
    selected_templates: List[Dict[str, Any]],
    all_candidate_chunks: List[Dict[str, Any]],
    dynamic_rules: Dict[str, List[str]],
) -> Dict[str, Any]:
    aliases = set([requirement])
    aliases.update(TOPIC_ALIAS.get(requirement, []))
    aliases.update(dynamic_rules.get(requirement, []) or [])
    aliases.update(TOPIC_KEYWORDS.get(requirement, []) or [])
    aliases = {normalize_text(x) for x in aliases if str(x).strip()}

    draft_hits = []
    for line in re.split(r"[\n。；;]", draft_text):
        ln = line.strip()
        if not ln:
            continue
        nln = normalize_text(ln)
        if any(a and a in nln for a in aliases):
            draft_hits.append(ln)

    history_hits = []
    for chunk in all_candidate_chunks:
        content = chunk.get("content", "")
        ncontent = normalize_text(content)
        if any(a and a in ncontent for a in aliases):
            history_hits.append({
                "file_name": chunk.get("file_name", "未知檔案"),
                "topics": "、".join(_chunk_topics_list(chunk)) or chunk.get("topics", ""),
                "content": content[:300]
            })

    if not history_hits:
        for t in selected_templates:
            source_text = t.get("source_text", "")
            nsource = normalize_text(source_text)
            if any(a and a in nsource for a in aliases):
                history_hits.append({
                    "file_name": t.get("file_name", "未知檔案"),
                    "topics": "、".join(t.get("core_topics", [])),
                    "content": source_text[:300]
                })

    return {
        "requirement": requirement,
        "draft_hits": draft_hits[:5],
        "history_hits": history_hits[:5],
        "examples": dynamic_rules.get(requirement, []) or [],
    }

def llm_compliance_obligation_scan(
    draft_text: str,
    articles: List[Dict[str, Any]],
    selected_templates: Optional[List[Dict[str, Any]]] = None,
    all_candidate_chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    章則規範義務稽核｜上線穩定版

    核心原則：
    1. LLM 只判斷是否涵蓋，不讓 LLM 自由寫 suggested_addition。
    2. suggested_addition 由程式依 requirement 套固定保守條款。
    3. 若草稿已實質涵蓋，程式強制改為 is_covered=true，避免硬找錯。
    4. 若草稿把該義務寫成另行報價、甲方負擔、得視情況，強制視為未完整涵蓋。
    5. gap_description 不得出現「本次檢索片段未找到明確對應依據」這種對使用者沒有幫助的文字。
    """

    selected_templates = selected_templates or []
    all_candidate_chunks = all_candidate_chunks or []
    full_text = "\n".join(a.get("content", "") for a in articles)

    dynamic_rules = get_compliance_rules() or DEFAULT_COMPLIANCE_RULES

    DEFAULT_SUGGESTED_CLAUSES = {
        "事件通報與應變": (
            "乙方應於發現資安事件後依甲方要求通報甲方，"
            "並配合提供必要之事件說明、應變支援與後續處理資料。"
        ),
        "個資保護與保密": (
            "乙方應遵守個人資料保護及保密相關規範，"
            "並對其因履約所知悉或持有之甲方資料負保密及妥善保管責任。"
        ),
        "備份與災難復原": (
            "乙方應依甲方要求配合資料備份、復原及災難復原相關作業，"
            "並提供必要之技術協助與佐證資料。"
        ),
        "弱點修補與維護": (
            "乙方應就其履約範圍內所發現或經甲方通知之系統弱點，"
            "配合進行修補、維護與處理進度回報。"
        ),
        "資安檢測與掃描": (
            "乙方應配合甲方執行必要之資安檢測與弱點掃描，"
            "並提供檢測相關資料、說明及必要協助。"
        ),
    }

    def _clean_optional(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        if text.lower() in {"none", "null", "n/a", "na"}:
            return None
        if text in {"未提供", "無提供", "無", "null", "None"}:
            return None
        return text

    def _normalize_requirement_name(name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""

        for topic in dynamic_rules.keys():
            if raw == topic:
                return topic

        normalized_raw = normalize_text(raw)

        for topic in dynamic_rules.keys():
            aliases = set()
            aliases.add(topic)
            aliases.update(TOPIC_ALIAS.get(topic, []) or [])
            aliases.update(TOPIC_KEYWORDS.get(topic, []) or [])
            aliases.update(dynamic_rules.get(topic, []) or [])

            normalized_aliases = {
                normalize_text(a)
                for a in aliases
                if str(a).strip()
            }

            if normalized_raw in normalized_aliases:
                return topic

        return raw

    def _default_clause_for(requirement: str) -> str:
        requirement = str(requirement or "").strip()

        if requirement in DEFAULT_SUGGESTED_CLAUSES:
            return DEFAULT_SUGGESTED_CLAUSES[requirement]

        return (
            f"乙方應依甲方要求配合辦理「{requirement}」相關作業，"
            "並提供履約所必要之協助、說明與佐證資料。"
        )

    def _is_weak_coverage(text: str) -> bool:
        found_norm = normalize_text(text)

        weak_coverage_triggers = [
            "另行報價",
            "另行收費",
            "額外計費",
            "另簽",
            "另付費",
            "由甲方自行負擔",
            "甲方另行負擔",
            "由甲方負擔",
            "得視情況",
            "得另行",
            "不包含",
            "不負責",
            "無須配合",
            "概不負責",
        ]

        return any(trigger in found_norm for trigger in weak_coverage_triggers)

    def _sanitize_gap_description(value: Any) -> Optional[str]:
        text = _clean_optional(value)
        if not text:
            return None

        retrieval_only_full_patterns = [
            r"^本次檢索片段未找到明確對應依據[。\.]?$",
            r"^未於本次檢索片段中找到明確歷史作法[。\.]?$",
            r"^無相關歷史片段支持[。\.]?$",
            r"^也無相關歷史片段支持[。\.]?$",
            r"^無明確歷史片段支持[。\.]?$",
            r"^也無明確歷史片段支持[。\.]?$",
            r"^未找到明確歷史片段[。\.]?$",
            r"^無明確歷史片段[。\.]?$",
        ]

        for pattern in retrieval_only_full_patterns:
            if re.search(pattern, text):
                return None

        remove_tail_patterns = [
            r"[，,；;]?\s*也?無相關歷史片段支持[。\.]?",
            r"[，,；;]?\s*也?沒有相關歷史片段支持[。\.]?",
            r"[，,；;]?\s*也?無明確歷史片段支持[。\.]?",
            r"[，,；;]?\s*也?沒有明確歷史片段支持[。\.]?",
            r"[，,；;]?\s*也?未找到明確歷史片段[。\.]?",
            r"[，,；;]?\s*也?無明確歷史片段[。\.]?",
            r"[，,；;]?\s*也?本次檢索片段未找到明確對應依據[。\.]?",
        ]

        for pattern in remove_tail_patterns:
            text = re.sub(pattern, "", text).strip()

        bad_endings = [
            "，也", ",也", "；也", ";也",
            "，且", ",且", "；且", ";且",
            "，", ",", "；", ";", "、", "也", "且",
        ]

        changed = True
        while changed:
            changed = False
            for ending in bad_endings:
                if text.endswith(ending):
                    text = text[: -len(ending)].strip()
                    changed = True

        text = text.strip(" ，,。；;、")

        if not text:
            return None

        if not text.endswith(("。", "！", "？")):
            text += "。"

        return text

    def _default_gap_description_for(
        requirement: str,
        found_clause: Optional[str] = None,
    ) -> str:
        requirement = _normalize_requirement_name(requirement)
        found_clause = _clean_optional(found_clause)

        if found_clause:
            return (
                f"草稿已有相關文字：「{found_clause}」，"
                f"但未能確認已完整約定「{requirement}」之廠商協助義務。"
            )

        default_gaps = {
            "事件通報與應變": (
                "草稿未明確約定乙方於發現資安事件後之通報、說明、"
                "應變支援與後續處理協助義務。"
            ),
            "個資保護與保密": (
                "草稿未明確約定乙方就履約過程中接觸、知悉或持有之甲方資料，"
                "負個資保護、保密及妥善保管義務。"
            ),
            "備份與災難復原": (
                "草稿未明確約定乙方應配合資料備份、復原或災難復原相關作業之協助義務。"
            ),
            "弱點修補與維護": (
                "草稿未明確約定乙方就系統弱點或漏洞進行修補、維護及處理進度回報之義務。"
            ),
            "資安檢測與掃描": (
                "草稿未明確約定乙方應配合甲方執行資安檢測、弱點掃描並提供相關資料與必要協助。"
            ),
        }

        return default_gaps.get(
            requirement,
            f"草稿未明確約定乙方應配合辦理「{requirement}」相關作業之廠商協助義務。",
        )

    def _coverage_keywords_for(requirement: str) -> List[str]:
        requirement = _normalize_requirement_name(requirement)

        coverage_keywords = {
            "事件通報與應變": [
                "資安事件",
                "安全事件",
                "事件通報",
                "通報甲方",
                "應變支援",
                "後續處理",
                "事件說明",
            ],
            "個資保護與保密": [
                "個人資料",
                "個資",
                "保密",
                "機密",
                "妥善保管",
                "不得洩漏",
                "資料保護",
            ],
            "備份與災難復原": [
                "備份",
                "資料備份",
                "復原",
                "災難復原",
                "災復",
                "復原作業",
            ],
            "弱點修補與維護": [
                "弱點修補",
                "漏洞修補",
                "系統弱點",
                "安全漏洞",
                "修補",
                "維護",
                "處理進度回報",
            ],
            "資安檢測與掃描": [
                "資安檢測",
                "弱點掃描",
                "漏洞掃描",
                "安全掃描",
                "資安掃描",
                "檢測相關資料",
                "必要協助",
            ],
        }

        keywords = coverage_keywords.get(requirement, [])

        if not keywords:
            aliases = set()
            aliases.add(requirement)
            aliases.update(TOPIC_ALIAS.get(requirement, []) or [])
            aliases.update(TOPIC_KEYWORDS.get(requirement, []) or [])
            aliases.update(dynamic_rules.get(requirement, []) or [])
            keywords = list(aliases)

        return [
            normalize_text(k)
            for k in keywords
            if str(k).strip()
        ]

    def _sentences_related_to_requirement(requirement: str, text: str) -> List[str]:
        """
        只抓真正與該 requirement 有關的句子。
        避免「雲端、程式碼、公開」被誤當成備份、弱點修補、資安檢測。
        """
        normalized_keywords = _coverage_keywords_for(requirement)

        # 每個 requirement 必須命中比較核心的詞，不能只靠泛用詞
        strict_keywords = {
            "事件通報與應變": ["資安事件", "安全事件", "事件通報", "通報", "應變", "事故", "系統中止", "運作阻礙"],
            "個資保護與保密": ["個人資料", "個資", "保密", "機密", "不得洩漏", "資料保護"],
            "備份與災難復原": ["備份", "資料備份", "災難復原", "災復", "復原演練", "備援"],
            "弱點修補與維護": ["弱點修補", "漏洞修補", "安全漏洞", "系統弱點", "修補", "修復", "維護"],
            "資安檢測與掃描": ["資安檢測", "弱點掃描", "漏洞掃描", "安全掃描", "資安掃描", "滲透測試"],
        }

        required_hits = [
            normalize_text(k)
            for k in strict_keywords.get(requirement, normalized_keywords)
            if str(k).strip()
        ]

        unrelated_negative_keywords = {
            "備份與災難復原": ["雲端代碼儲存空間", "公眾可無償存取", "查閱", "複製", "優化", "開源", "上傳"],
            "弱點修補與維護": ["雲端代碼儲存空間", "公眾可無償存取", "查閱", "複製", "優化", "開源", "上傳"],
            "資安檢測與掃描": ["雲端代碼儲存空間", "公眾可無償存取", "查閱", "複製", "優化", "開源", "上傳"],
        }

        raw_parts = re.split(r"[\n。；;]", str(text or ""))
        related = []

        for part in raw_parts:
            part = str(part or "").strip()
            if not part:
                continue

            n_part = normalize_text(part)
            negative_hits = [normalize_text(k) for k in unrelated_negative_keywords.get(requirement, [])]

            if any(neg and neg in n_part for neg in negative_hits) and not any(k and k in n_part for k in required_hits):
                continue

            if any(k and k in n_part for k in required_hits):
                related.append(part)

        return related

    def _has_substantial_coverage(requirement: str, text: str) -> bool:
        """
        實質涵蓋判斷。

        只要草稿中與該 requirement 相關的句子已經明確出現：
        1. 乙方/廠商義務主體
        2. 對應 requirement 的核心義務詞
        3. 該相關句子沒有弱化責任文字

        就視為已涵蓋，避免系統硬找錯。
        """
        requirement = _normalize_requirement_name(requirement)
        related_sentences = _sentences_related_to_requirement(requirement, text)

        if not related_sentences:
            return False

        subject_words = [
            "乙方應",
            "乙方須",
            "乙方需",
            "乙方應配合",
            "乙方應提供",
            "乙方負責",
            "乙方應遵守",
            "乙方不得",
            "廠商應",
            "廠商須",
            "廠商需",
        ]

        normalized_keywords = _coverage_keywords_for(requirement)

        for sentence in related_sentences:
            n_sentence = normalize_text(sentence)

            if _is_weak_coverage(sentence):
                continue

            subject_hit = any(word in n_sentence for word in subject_words)
            keyword_hit = any(k and k in n_sentence for k in normalized_keywords)

            if subject_hit and keyword_hit:
                return True

        return False

    def _requirement_has_weak_coverage(requirement: str, text: str) -> bool:
        """
        只看與該 requirement 相關的句子是否有弱化責任。
        """
        related_sentences = _sentences_related_to_requirement(requirement, text)
        return any(_is_weak_coverage(sentence) for sentence in related_sentences)

    def _find_related_draft_clause(requirement: str, text: str) -> Optional[str]:
        related_sentences = _sentences_related_to_requirement(requirement, text)
        if related_sentences:
            return related_sentences[0]
        return None

    evidence_packets = []

    for topic in dynamic_rules.keys():
        evidence_packets.append(
            _build_requirement_evidence(
                requirement=topic,
                draft_text=full_text,
                selected_templates=selected_templates,
                all_candidate_chunks=all_candidate_chunks,
                dynamic_rules=dynamic_rules,
            )
        )

    obligation_list = "\n".join(
        f"- 【{topic}】：需檢查是否明確涵蓋此項廠商協助義務"
        for topic in dynamic_rules.keys()
    )

    evidence_text_parts = []

    for packet in evidence_packets:
        history_text = "\n".join(
            f"  - 歷史依據【{h['file_name']}】({h.get('topics', '')})：{h['content']}"
            for h in packet["history_hits"]
        ) or "  - 無明確歷史片段"

        draft_texts = "\n".join(
            f"  - 草稿文字：{d}" for d in packet["draft_hits"]
        ) or "  - 草稿未明確提及"

        examples = "、".join(packet.get("examples", [])[:5])

        evidence_text_parts.append(
            f"【{packet['requirement']}】\n"
            f"規範示例：{examples or '無'}\n"
            f"{draft_texts}\n"
            f"{history_text}"
        )

    prompt = f"""
你是企業法遵稽核員，立場站在甲方。
請依據「章則規範要求清單」逐項檢查廠商合約是否已包含對應的廠商協助義務。

【章則規範要求清單】
{obligation_list}

【廠商合約全文（節錄）】
{full_text[:6000]}

【歷史合約與規範證據】
{chr(10).join(evidence_text_parts)[:12000]}

嚴格規則：
1. 只能判斷是否涵蓋與說明缺口。
2. 採「實質涵蓋」標準，不要求文字完全相同。
3. 若草稿已明確約定乙方負有該項協助、配合、處理、通報、保密、備份、修補或檢測義務，應判定 is_covered=true。
4. 不得因為草稿沒有使用章則示例的完全相同文字，就判定為缺漏。
5. 不得為了產生缺漏而硬找問題。
6. 若草稿把廠商義務改成另行報價、另簽、另付費、甲方自理，應判定為未完整涵蓋。
7. 若草稿有提到相關主題，但只是可選服務、另行收費或責任不清，也應判定 is_covered=false。
8. 若 is_covered=true，gap_description 必須為 null。
9. 不得自行撰寫補充條款。
10. 不得自行新增時限、頻率、費用、比例、標準或流程。
11. requirement 必須從章則規範要求清單中選，不得新增名稱。

請只輸出 JSON，不要有其他文字：
{{
  "compliance_scan_results": [
    {{
      "requirement": "章則規範要求的義務名稱",
      "is_covered": true,
      "found_clause": "若有涵蓋或有相關但不足的文字，直接引用合約原文；若完全沒有則填 null",
      "gap_description": "若未涵蓋或涵蓋不足，說明具體缺口；若完全已涵蓋則填 null"
    }}
  ]
}}
"""

    result = ollama_json(prompt, model=JSON_MODEL)
    scan_results = result.get("compliance_scan_results", [])

    normalized: List[Dict[str, Any]] = []
    seen_requirements = set()

    for r in scan_results:
        if not isinstance(r, dict):
            continue

        requirement = _normalize_requirement_name(r.get("requirement", ""))

        if not requirement:
            continue

        if requirement not in dynamic_rules:
            continue

        found_clause = _clean_optional(r.get("found_clause"))
        related_clause = _find_related_draft_clause(requirement, full_text)
        if related_clause:
            found_clause = related_clause
        else:
            found_clause = None

        gap_description = _sanitize_gap_description(r.get("gap_description"))
        is_covered = bool(r.get("is_covered", False))

        # 程式保護 1：該 requirement 已實質涵蓋，強制 covered
        if _has_substantial_coverage(requirement, full_text):
            is_covered = True
            gap_description = None

        # 程式保護 2：該 requirement 的相關文字有弱化責任，強制 uncovered
        elif _requirement_has_weak_coverage(requirement, full_text) or (
            found_clause and _is_weak_coverage(found_clause)
        ):
            is_covered = False
            gap_description = (
                "草稿雖有提及相關事項，但以另行報價、另行收費、"
                "甲方負擔或非明確義務方式呈現，未完整涵蓋廠商協助義務。"
            )

        # 程式保護 3：LLM 有指出實質 gap，才 uncovered
        elif gap_description:
            is_covered = False

        normalized.append({
            "requirement": requirement,
            "is_covered": is_covered,
            "found_clause": found_clause,
            "gap_description": None if is_covered else (
                gap_description or _default_gap_description_for(requirement, found_clause)
            ),
            "suggested_addition": None if is_covered else _default_clause_for(requirement),
        })

        seen_requirements.add(requirement)

    # 模型沒吐全時，補齊所有 requirement
    for topic in dynamic_rules.keys():
        if topic in seen_requirements:
            continue

        evidence = _build_requirement_evidence(
            requirement=topic,
            draft_text=full_text,
            selected_templates=selected_templates,
            all_candidate_chunks=all_candidate_chunks,
            dynamic_rules=dynamic_rules,
        )

        found_clause = evidence["draft_hits"][0] if evidence["draft_hits"] else None
        if not found_clause:
            found_clause = _find_related_draft_clause(topic, full_text)

        if _has_substantial_coverage(topic, full_text):
            is_covered = True
            gap = None
        elif _requirement_has_weak_coverage(topic, full_text) or (
            found_clause and _is_weak_coverage(found_clause)
        ):
            is_covered = False
            gap = (
                "草稿雖有提及相關事項，但以另行報價、另行收費、"
                "甲方負擔或非明確義務方式呈現，未完整涵蓋廠商協助義務。"
            )
        else:
            is_covered = False
            gap = _default_gap_description_for(topic, found_clause)

        normalized.append({
            "requirement": topic,
            "is_covered": is_covered,
            "found_clause": found_clause,
            "gap_description": gap,
            "suggested_addition": None if is_covered else _default_clause_for(topic),
        })

    # 最後去重與防呆
    final_results = []
    final_seen = set()

    for item in normalized:
        requirement = _normalize_requirement_name(item.get("requirement", ""))

        if not requirement or requirement in final_seen:
            continue

        final_seen.add(requirement)

        found_clause = item.get("found_clause")
        gap_description = _sanitize_gap_description(item.get("gap_description"))
        is_covered = bool(item.get("is_covered", False))

        if _has_substantial_coverage(requirement, full_text):
            is_covered = True
            gap_description = None

        elif _requirement_has_weak_coverage(requirement, full_text) or (
            found_clause and _is_weak_coverage(found_clause)
        ):
            is_covered = False
            gap_description = (
                "草稿雖有提及相關事項，但以另行報價、另行收費、"
                "甲方負擔或非明確義務方式呈現，未完整涵蓋廠商協助義務。"
            )

        final_gap = None

        if not is_covered:
            final_gap = _sanitize_gap_description(gap_description)

            if not final_gap:
                final_gap = _default_gap_description_for(requirement, found_clause)

        final_results.append({
            "requirement": requirement,
            "is_covered": is_covered,
            "found_clause": found_clause,
            "gap_description": final_gap,
            "suggested_addition": None if is_covered else _default_clause_for(requirement),
        })

    return final_results

def cross_contract_gap_analysis(
    current_draft_text: str,
    selected_templates: List[Dict[str, Any]],
    compliance_scan: List[Dict[str, Any]],
    all_candidate_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    跨合約差距分析｜穩定上線版

    設計原則：
    1. 不使用 LLM 自由生成，避免幻想歷史合約作法。
    2. 僅根據 compliance_scan、實際檢索片段與 suggested_addition 輸出。
    3. 若找不到歷史片段，明確標示「未於本次檢索片段中找到明確歷史作法」。
    4. 不自行新增頻率、時限、費用分攤、標準、流程。
    """

    all_candidate_chunks = all_candidate_chunks or []
    compliance_scan = compliance_scan or []

    uncovered = [
        r for r in compliance_scan
        if isinstance(r, dict) and not r.get("is_covered", True)
    ]

    if not uncovered:
        return {
            "gap_summary": "本次合約已涵蓋所有章則規範要求的廠商義務，無明顯差距。",
            "gaps": [],
        }

    def _clean(value: Any) -> str:
        text = str(value or "").strip()
        if text in ["", "None", "null", "未提供", "無提供"]:
            return ""
        return text

    def _build_aliases(topic: str) -> set:
        aliases = set()
        topic = _clean(topic)
        if topic:
            aliases.add(topic)

        try:
            aliases.update(TOPIC_ALIAS.get(topic, []) or [])
        except Exception:
            pass

        try:
            aliases.update(TOPIC_KEYWORDS.get(topic, []) or [])
        except Exception:
            pass

        return {normalize_text(x) for x in aliases if str(x).strip()}

    def _find_related_evidence(topic: str) -> List[Dict[str, str]]:
        aliases = _build_aliases(topic)
        if not aliases:
            return []

        related = []

        # 1. 先從 all_candidate_chunks 找真正命中的歷史片段
        for chunk in all_candidate_chunks:
            if not isinstance(chunk, dict):
                continue

            content = _clean(chunk.get("content"))
            if not content:
                continue

            ncontent = normalize_text(content)
            topic_text = normalize_text(
                " ".join(_chunk_topics_list(chunk))
                or str(chunk.get("topics", "") or "")
            )

            if any(alias and (alias in ncontent or alias in topic_text) for alias in aliases):
                related.append({
                    "file_name": _clean(chunk.get("file_name")) or "未知檔案",
                    "label": _chunk_label(chunk),
                    "content": content[:220],
                })

        # 2. 若 chunks 沒有，再從 selected_templates 的 source_text 補找
        if not related:
            for template in selected_templates[:10]:
                if not isinstance(template, dict):
                    continue

                source_text = _clean(template.get("source_text"))
                if not source_text:
                    continue

                nsource = normalize_text(source_text)
                if any(alias and alias in nsource for alias in aliases):
                    related.append({
                        "file_name": _clean(template.get("file_name")) or "未知檔案",
                        "label": "全文摘要",
                        "content": source_text[:220],
                    })

        return related[:5]

    def _format_history_coverage(evidence_list: List[Dict[str, str]]) -> str:
        if not evidence_list:
            return "未於本次檢索片段中找到明確歷史作法；本項主要依章則規範義務稽核結果判定。"

        parts = []
        for evidence in evidence_list:
            parts.append(
                f"歷史文件【{evidence['file_name']}｜{evidence['label']}】記載：{evidence['content']}"
            )

        return "；".join(parts)

    def _build_current_status(item: Dict[str, Any]) -> str:
        found_clause = _clean(item.get("found_clause"))
        gap_description = _clean(item.get("gap_description"))

        if found_clause and gap_description:
            return f"草稿已有相關文字：「{found_clause}」；但問題為：{gap_description}"

        if found_clause:
            return f"草稿已有相關文字：「{found_clause}」，但仍未被判定為完整涵蓋。"

        if gap_description:
            return gap_description

        return "本次草稿未明確約定此項廠商協助義務。"

    def _build_risk_description(item: Dict[str, Any]) -> str:
        gap_description = _clean(item.get("gap_description"))
        topic = _clean(item.get("requirement"))

        if gap_description:
            return f"若廠商拒絕補強，將延續此缺口：{gap_description}"

        if topic:
            return f"若廠商拒絕補強「{topic}」，甲方可能需自行承擔相關合規、資安或營運風險。"

        return "若廠商拒絕補強，甲方可能需自行承擔相關合規、資安或營運風險。"

    def _build_cost_suggestion(item: Dict[str, Any]) -> str:
        suggested = _clean(item.get("suggested_addition"))

        # 只給保守原則，不自行編費用比例或共同分擔
        if "免費" in suggested or "無償" in suggested:
            return "建議依建議補充條文處理，明確約定乙方應負擔其義務範圍內之配合作業。"

        if any(word in suggested for word in ["費用", "價金", "另計費", "報價"]):
            return "建議於合約中明訂費用負擔、適用範圍、上限與啟動條件，避免日後爭議。"

        return "建議於合約中明確約定本項義務是否已包含於合約價金；如需另計費，應另定範圍、上限與書面同意程序。"

    normalized_gaps = []

    for item in uncovered:
        topic = _clean(item.get("requirement"))
        if not topic:
            continue

        evidence_list = _find_related_evidence(topic)
        suggested_addition = _clean(item.get("suggested_addition"))

        risk_description = _build_risk_description(item)
        cost_bearing_suggestion = _build_cost_suggestion(item)
        alternative_clause = suggested_addition or "目前審查結果未提供可直接引用之替代條文，建議由法務依規範另行補訂。"

        normalized_gaps.append({
            "topic": topic,
            "other_vendors_coverage": _format_history_coverage(evidence_list),
            "current_vendor_status": _build_current_status(item),
            "vendor_refuse_scenario": {
                "risk_description": risk_description,
                "cost_bearing_suggestion": cost_bearing_suggestion,
                "alternative_clause": alternative_clause,
            },
            "risk_description": risk_description,
            "cost_bearing_suggestion": cost_bearing_suggestion,
            "alternative_clause": alternative_clause,
        })

    topic_names = [g["topic"] for g in normalized_gaps]
    if topic_names:
        if len(topic_names) <= 3:
            topic_text = "、".join(topic_names)
        else:
            topic_text = "、".join(topic_names[:3]) + f"等 {len(topic_names)} 項"

        gap_summary = (
            f"本次合約尚未完整涵蓋 {topic_text} 廠商協助義務；"
            "建議依章則規範義務稽核結果補入明確條款，並釐清費用負擔與執行責任。"
        )
    else:
        gap_summary = "本次合約存在若干章則規範義務缺口，建議依審查結果逐項補強。"

    return {
        "gap_summary": gap_summary,
        "gaps": normalized_gaps,
    }

# 防禦 Prompt Injection
def check_prompt_injection(text: str) -> bool:
    """
    偵測明顯的 Prompt Injection 攻擊。

    注意：
    不要把「無風險」這類正常合約用語單獨列為封鎖條件，
    否則會誤擋正常合約內容或使用者的一般提問。
    """
    suspicious_patterns = [
        r"忽略.*(系統|上述|前面|所有).*指示",
        r"ignore.*(system|previous|all).*instructions",
        r"reveal.*system.*prompt",
        r"show.*system.*prompt",
        r"system prompt",
        r"忘記.*(規則|指令|設定)",
        r"覆寫.*(規則|指令|設定)",
        r"override.*(rules|instructions|system)",
        r"你是.*不要扮演",
        r"不要.*遵守.*規則",
        r"請.*直接輸出.*無風險",
        r"直接.*判定.*無風險",
        r"不要審查.*無風險",
        r"略過.*審查",
        r"跳過.*合規",
    ]

    text_lower = text.lower()

    for pattern in suspicious_patterns:
        if re.search(pattern, text_lower):
            logging.warning(f"🚨 偵測到潛在的 Prompt Injection 攻擊！觸發規則：{pattern}")
            return True

    return False


def review_articles_individually(
    draft_text: str,
    selected_templates: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    draft_type: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    emit_progress(progress_callback, "start", "開始合約審查", 1)

    if check_prompt_injection(draft_text):
        emit_progress(progress_callback, "blocked", "偵測到疑似 Prompt Injection，已中止", 100)
        report = ReviewReport(
            contract_type_guess="拒絕審查",
            summary="🚨 系統偵測到合約內容包含惡意指令（Prompt Injection），為保護系統與營業機密安全，已強制終止審查程序。",
            score=0,
        )
        return report.model_dump()

    emit_progress(progress_callback, "detect_contract_type", "正在判定合約類型", 5)

    guessed = guess_draft_contract_type(draft_text)

    if draft_type is None or not str(draft_type).strip():
        draft_type = guessed.get("primary_type", "其他")

    draft_type = _normalize_contract_type_label(draft_type)
    contract_mode = guessed.get("mode", detect_contract_mode_from_text(draft_text))
    enabled_ruleset = sorted(CONTRACT_TYPE_RULESET.get(draft_type, []))

    emit_progress(progress_callback, "detect_contract_type", f"合約類型判定完成：{draft_type}", 10)
    emit_progress(progress_callback, "prepare_articles", "正在分析條款類型", 15)

    for article in articles:
        article["clause_type"] = detect_clause_type(
            article.get("content", ""),
            article.get("title", ""),
        )

    all_major: List[Dict[str, Any]] = []
    all_general: List[Dict[str, Any]] = []
    found_topics_by_llm: set = set()
    all_article_chunks: List[Dict[str, Any]] = []

    emit_progress(progress_callback, "trigger_scan", "正在執行企業紅線規則掃描", 18)

    trigger_issues = build_trigger_issues_from_articles(
        articles,
        draft_type=draft_type,
    )

    if trigger_issues:
        for issue in trigger_issues:
            issue["contract_type"] = draft_type
            issue["rule_topic"] = normalize_topic_name(issue.get("issue_topic", ""))
            issue["clause_topic"] = normalize_topic_name(issue.get("issue_topic", ""))
            issue["evidence_topic"] = normalize_topic_name(issue.get("issue_topic", ""))

        all_major.extend(trigger_issues)

        for issue in trigger_issues:
            topic = issue.get("issue_topic")
            if topic:
                found_topics_by_llm.add(normalize_topic_name(topic))

        logging.info(f"企業紅線規則命中 {len(trigger_issues)} 項風險")

    article_chunk_pairs = []
    total_articles = max(len(articles), 1)

    emit_progress(progress_callback, "retrieve_chunks", f"開始檢索 {len(articles)} 條條文的歷史片段", 20)

    for idx, article in enumerate(articles, start=1):
        article_key = article_to_key(article, idx)

        candidate_chunks = search_template_chunks_for_article(
            article,
            selected_templates,
            n_results=4,
        )

        all_article_chunks.extend(candidate_chunks)
        article_chunk_pairs.append((article, article_key, candidate_chunks))

        percent = 20 + int((idx / total_articles) * 20)
        emit_progress(
            progress_callback,
            "retrieve_chunks",
            f"已完成第 {idx}/{total_articles} 條的歷史片段檢索",
            percent,
        )

    def _review_one(args):
        article, article_key, candidate_chunks = args
        result = llm_review_single_article(
            article,
            article_key,
            candidate_chunks,
            draft_type=draft_type,
        )
        return article, article_key, result

    emit_progress(progress_callback, "review_articles", "開始逐條審查", 45)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_review_one, args): args for args in article_chunk_pairs}
        total_reviews = len(futures)
        completed_reviews = 0

        for future in as_completed(futures):
            try:
                article, article_key, article_result = future.result()
            except Exception as e:
                logging.error(f"條文並行審查失敗：{e}")
                completed_reviews += 1
                percent = 45 + int((completed_reviews / max(total_reviews, 1)) * 20)
                emit_progress(
                    progress_callback,
                    "review_articles",
                    f"第 {completed_reviews}/{total_reviews} 條審查失敗，持續處理其餘條文",
                    percent,
                )
                continue

            majors = _sanitize_issues_for_article(
                article_result.get("major_issues", []),
                article,
                article_key,
            )
            generals = _sanitize_issues_for_article(
                article_result.get("general_issues", []),
                article,
                article_key,
            )

            article_topics = [
                normalize_topic_name(x)
                for x in article.get("topics", []) or []
                if str(x).strip()
            ]
            clause_topic = article_topics[0] if article_topics else normalize_topic_name(article.get("clause_type", ""))

            for issue in majors if isinstance(majors, list) else []:
                issue["contract_type"] = draft_type
                issue["clause_topic"] = clause_topic
                issue["rule_topic"] = normalize_topic_name(issue.get("issue_topic", ""))
                issue["evidence_topic"] = normalize_topic_name(issue.get("issue_topic", ""))

            for issue in generals if isinstance(generals, list) else []:
                issue["contract_type"] = draft_type
                issue["clause_topic"] = clause_topic
                issue["rule_topic"] = normalize_topic_name(issue.get("issue_topic", ""))
                issue["evidence_topic"] = normalize_topic_name(issue.get("issue_topic", ""))

            if isinstance(majors, list):
                all_major.extend(majors)
            if isinstance(generals, list):
                all_general.extend(generals)

            for issue in (majors if isinstance(majors, list) else []) + (generals if isinstance(generals, list) else []):
                topic = issue.get("issue_topic")
                if topic:
                    found_topics_by_llm.add(normalize_topic_name(topic))

            completed_reviews += 1
            percent = 45 + int((completed_reviews / max(total_reviews, 1)) * 20)
            emit_progress(
                progress_callback,
                "review_articles",
                f"已完成第 {completed_reviews}/{total_reviews} 條審查",
                percent,
            )

    emit_progress(progress_callback, "infer_missing", "正在比對缺漏條款", 68)

    missing_topic_tuples = infer_missing_topics_from_templates(
        selected_templates,
        articles,
        found_topics_by_llm,
        all_article_chunks,
    )

    def _run_batch_draft():
        return _draft_all_missing_clauses(missing_topic_tuples)

    def _run_compliance_scan():
        return llm_compliance_obligation_scan(
            draft_text,
            articles,
            selected_templates=selected_templates,
            all_candidate_chunks=all_article_chunks,
        )

    emit_progress(progress_callback, "compliance_scan", "開始執行缺漏條文起草與合規義務掃描", 72)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_drafts = executor.submit(_run_batch_draft)
        future_compliance = executor.submit(_run_compliance_scan)

        all_clause_drafts = future_drafts.result()
        emit_progress(progress_callback, "draft_missing_clauses", "缺漏條文草稿產生完成", 80)

        compliance_scan = future_compliance.result()
        emit_progress(progress_callback, "compliance_scan", "合規義務掃描完成", 88)

    missing_clauses = []
    for topic, source, snippet in missing_topic_tuples:
        normalized_topic = normalize_topic_name(topic)

        suggested_draft = all_clause_drafts.get(
            topic,
            f"乙方應依企業規範履行「{topic}」相關義務，具體標準參照甲方歷史合約基準。",
        )

        missing_clauses.append({
            "clause": topic,
            "issue_topic": normalized_topic,
            "contract_type": draft_type,
            "evidence_topic": normalized_topic,
            "why_missing": f"企業歷史基準或法規文檔【{source}】中有明確相關規範，但目前草稿中未找到對應保障。",
            "suggestion": f"建議參考歷史基準中關於「{topic}」的條款，補入具體規定以確保合規。",
            "suggested_draft": suggested_draft,
            "source": source,
            "template_snippet": snippet,
        })

    normalized_compliance_scan = []
    for item in compliance_scan or []:
        if not isinstance(item, dict):
            continue
        requirement = normalize_topic_name(item.get("requirement", ""))
        item["topic"] = normalize_topic_name(item.get("topic") or requirement)
        item["contract_type"] = item.get("contract_type") or draft_type
        normalized_compliance_scan.append(item)

    emit_progress(progress_callback, "finalize", "審查資料整理完成", 95)

    all_major = normalize_issue_cards(all_major, default_group="major")
    all_general = normalize_issue_cards(all_general, default_group="general")
    risk_cards = all_major + all_general

    result = {
        "contract_type_guess": draft_type,
        "contract_mode": contract_mode,
        "enabled_ruleset": enabled_ruleset,
        "summary": "系統已根據動態檢索出的歷史合約全集、法遵規範與企業紅線規則，逐條進行合規落差掃描，並完成章則義務稽核。",
        "major_issues": all_major,
        "general_issues": all_general,
        "risk_cards": risk_cards,
        "missing_clauses": missing_clauses,
        "compliance_scan": normalized_compliance_scan,
        "all_candidate_chunks": all_article_chunks,
    }

    emit_progress(progress_callback, "done", "services 審查流程完成", 100)
    return result

def normalize_review_json(
    raw: Dict[str, Any],
    used_templates: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    top_chunks: List[Dict[str, Any]],
    original_draft_text: str,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    article_map = build_article_map(articles)

    def _article_display_name_for_report(article: Dict[str, Any], fallback_clause: str) -> str:
        raw_no = str(article.get("article_no", "") or "").strip()
        raw_title = str(article.get("title", "") or "").strip()

        if raw_no and raw_title:
            return f"{raw_no}：{raw_title}"
        if raw_no:
            return raw_no
        if raw_title:
            return raw_title
        return fallback_clause or "未命名項目"

    def _find_article_for_issue(article_key: str, clause: str, draft_text: str) -> Dict[str, Any]:
        article = article_map.get(article_key)
        if article and str(article.get("content", "") or "").strip():
            return article

        clause_norm = normalize_text(clause)
        draft_norm = normalize_text(draft_text)

        for candidate in articles:
            candidate_content = str(candidate.get("content", "") or "")
            candidate_title = str(candidate.get("title", "") or "")
            candidate_no = str(candidate.get("article_no", "") or "")
            candidate_header_norm = normalize_text(f"{candidate_no} {candidate_title}")
            candidate_content_norm = normalize_text(candidate_content)

            if draft_norm and candidate_content_norm and (
                draft_norm in candidate_content_norm or candidate_content_norm in draft_norm
            ):
                return candidate
            if clause_norm and clause_norm in candidate_header_norm:
                return candidate
            if clause_norm and clause_norm in candidate_content_norm[:160]:
                return candidate

        return article or {"content": ""}

    def _analysis_has_substance(text: str) -> bool:
        t = normalize_text(text)
        if len(t) < 18:
            return False

        generic_phrases = [
            "未明確規定",
            "不符合歷史基準",
            "建議補入",
            "建議修改",
            "可能導致風險",
        ]

        return not (len(t) < 30 and any(p in t for p in generic_phrases))

    def _is_system_trigger_issue(item: Dict[str, Any]) -> bool:
        source = str(item.get("source", "") or "")
        basis = str(item.get("template_basis", "") or item.get("template_snippet", "") or "")
        return "系統內建企業紅線規則" in source or "CRITICAL_RISK_TRIGGERS" in basis or "HIGH_RISK_TRIGGERS" in basis

    def norm_issue(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        clause = normalize_text(item.get("clause", "未命名條款"))
        topic = normalize_topic_name(item.get("issue_topic", "一般風險"))
        source = str(item.get("source", "") or "").strip()
        article_key = str(item.get("article_key", "") or "").strip()

        raw_draft_text = normalize_text(item.get("draft_text") or "")
        article = _find_article_for_issue(article_key, clause, raw_draft_text)
        base_clause = _article_display_name_for_report(article, clause)
        display_clause = f"{base_clause}｜{topic}" if topic and topic not in base_clause else base_clause
        draft_text = normalize_text(
            raw_draft_text
            or article.get("content", "")
            or ""
        )

        extracted_snippet = str(
            item.get("template_basis")
            or item.get("template_snippet")
            or "詳見基準內容"
        ).strip()

        analysis = normalize_text(str(item.get("analysis", "")))
        suggestion = normalize_text(str(item.get("suggestion", "")))

        # 一般 LLM issue 如果沒有草稿內容，丟掉
        # 但系統紅線 issue 允許保留，因為它本身就是從條文掃描產生
        if len(draft_text.strip()) < 15 and not _is_system_trigger_issue(item):
            return None

        # 一般 LLM issue 若分析與建議都很空，丟掉
        # 系統紅線 issue 不丟，避免 high risk 被 LLM 品質影響
        if (
            not _is_system_trigger_issue(item)
            and not _analysis_has_substance(analysis)
            and not _analysis_has_substance(suggestion)
        ):
            return None

        issue = {
            "clause": display_clause,
            "risk": str(item.get("risk", "Medium") or "Medium").strip(),
            "draft_text": draft_text,
            "template_basis": extracted_snippet,
            "template_snippet": str(item.get("template_snippet", extracted_snippet) or extracted_snippet).strip(),
            "analysis": analysis,
            "suggestion": suggestion,
            "adjusted_clause": str(item.get("adjusted_clause", "") or "").strip(),
            "negotiation_notes": str(item.get("negotiation_notes", "") or "").strip(),
            "source": source,
            "type": str(item.get("type", "deviation") or "deviation").strip(),
            "issue_topic": topic,
            "article_key": article_key,
        }

        dt = draft_text
        is_critical = False

        for triggers in CRITICAL_RISK_TRIGGERS.values():
            if any(t in dt for t in triggers):
                issue["risk"] = "Critical"
                issue["type"] = "conflict"
                is_critical = True
                break

        if not is_critical:
            for triggers in HIGH_RISK_TRIGGERS.values():
                if any(t in dt for t in triggers):
                    if _risk_level_rank(issue.get("risk")) < _risk_level_rank("High"):
                        issue["risk"] = "High"
                    break

        return issue

    raw_major = [x for x in (norm_issue(i) for i in raw.get("major_issues", [])) if x]
    raw_general = [x for x in (norm_issue(i) for i in raw.get("general_issues", [])) if x]

    def dedup_issues(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for it in items:
            key = (
                it.get("article_key", ""),
                it.get("issue_topic", ""),
                it.get("type", ""),
            )

            if key not in merged:
                merged[key] = it.copy()
                continue

            current = merged[key]

            # 風險等級取高者
            if _risk_level_rank(it.get("risk")) > _risk_level_rank(current.get("risk")):
                current["risk"] = it.get("risk", current.get("risk"))

            if str(it.get("risk", "")).lower() == "critical":
                current["type"] = "conflict"

            # 合併分析
            if it.get("analysis") and it.get("analysis") not in current.get("analysis", ""):
                if current.get("analysis"):
                    current["analysis"] += f"\n\n🔸 **其他關聯風險**：{it.get('analysis')}"
                else:
                    current["analysis"] = it.get("analysis", "")

            # 合併建議
            if it.get("suggestion") and it.get("suggestion") not in current.get("suggestion", ""):
                if current.get("suggestion"):
                    current["suggestion"] += f"\n\n🔧 **補充建議**：{it.get('suggestion')}"
                else:
                    current["suggestion"] = it.get("suggestion", "")

            # 補齊 adjusted_clause / negotiation_notes / source
            for field in ["adjusted_clause", "negotiation_notes", "source", "template_basis", "template_snippet"]:
                if not current.get(field) and it.get(field):
                    current[field] = it.get(field)

        return list(merged.values())

    all_issues = dedup_issues(raw_major + raw_general)

    # High / Critical 放 major，其餘放 general
    major_issues = [
        issue for issue in all_issues
        if str(issue.get("risk", "")).lower() in ["critical", "high"]
    ]
    general_issues = [
        issue for issue in all_issues
        if str(issue.get("risk", "")).lower() not in ["critical", "high"]
    ]

    uniq_templates = []
    seen_template_names: set = set()

    for t in used_templates:
        fname = t.get("file_name", "未知檔案")
        if fname not in seen_template_names:
            seen_template_names.add(fname)
            uniq_templates.append({
                "file_name": fname,
                "contract_type": t.get("contract_type", "其他"),
                "summary": t.get("summary", ""),
                "core_topics": t.get("core_topics", []),
            })

    cleaned_missing_clauses = []
    for m in raw.get("missing_clauses", []):
        if not isinstance(m, dict):
            continue

        snippet = normalize_text(str(m.get("template_snippet", "")))
        suggested_draft = normalize_text(str(m.get("suggested_draft", "")))

        # 有基準片段或有建議草稿即可保留
        if len(snippet) < 15 and len(suggested_draft) < 10:
            continue

        cleaned_missing_clauses.append(m)

    compliance_scan = raw.get("compliance_scan", [])
    if not isinstance(compliance_scan, list):
        compliance_scan = []

    current_score = 100

    for issue in major_issues + general_issues:
        risk_level = str(issue.get("risk", "")).lower()
        if risk_level == "critical":
            current_score -= 20
        elif risk_level == "high":
            current_score -= 10
        elif risk_level == "medium":
            current_score -= 5
        else:
            current_score -= 2

    current_score -= len(cleaned_missing_clauses) * 3

    uncovered_count = sum(
        1 for r in compliance_scan
        if isinstance(r, dict) and not r.get("is_covered", True)
    )
    current_score -= uncovered_count * 5

    final_score = max(0, current_score)

    try:
        report = ReviewReport(
            contract_type_guess=raw.get("contract_type_guess", "未判定"),
            summary="系統已根據歷史合約、法遵規範與企業紅線規則，完成草稿比對，並結合章則規範義務稽核與跨合約差距分析，整理出完整合規落差報告。",
            used_templates=uniq_templates,
            major_issues=major_issues,
            general_issues=general_issues,
            missing_clauses=cleaned_missing_clauses,
            score=final_score,
        )

        report_dict = report.model_dump()
        report_dict["compliance_scan"] = compliance_scan

        gap_analysis = cross_contract_gap_analysis(
            current_draft_text=original_draft_text,
            selected_templates=used_templates,
            compliance_scan=compliance_scan,
            all_candidate_chunks=raw.get("all_candidate_chunks") or top_chunks,
        )

        report_dict["gap_analysis"] = gap_analysis

        return report_dict

    except Exception as e:
        logging.error(f"審查報告格式化失敗: {e}")
        fallback = ReviewReport(
            summary=f"⚠️ 報告生成發生錯誤，請重新審查或手動檢查。錯誤訊息: {e}",
            score=0,
        )
        return fallback.model_dump()

def assess_price_risk(user_input: str) -> str:
    prompt = f"""
你是一位專業的金融合約分析官。請分析內容並嚴格輸出 JSON 格式。
{{"vendor_name": "廠商名稱", "amount": 1000000}}
輸入內容：{user_input[:2000]}
"""
    data = ollama_json(prompt)

    vendor_clean = (data.get("vendor_name", "") or data.get("vendor", "")).replace("万", "萬")
    try:
        current_amount = int(re.sub(r"[^\d]", "", str(data.get("amount", 0))))
    except ValueError:
        current_amount = 0

    if not vendor_clean or current_amount == 0:
        return f"⚠️ 萃取資訊不足。識別結果：廠商 `{vendor_clean}`，金額 `{current_amount}`。"

    past_records = find_history_by_vendor_keyword(vendor_clean)
    if not past_records and len(vendor_clean) >= 2:
        past_records = find_history_by_vendor_keyword(vendor_clean[:4])

    if past_records:
        avg_amount = sum(r["amount"] for r in past_records) / len(past_records)
        report = (
            "### 💰 歷史報價風險評估\n\n"
            f"- **識別廠商**：{vendor_clean}\n"
            f"- **本次報價**：新台幣 {current_amount:,.0f} 元\n"
            f"- **歷史均價**：新台幣 {avg_amount:,.0f} 元\n\n"
        )

        if current_amount > avg_amount * 1.5:
            report += "🚨 **【高風險警示】** 報價超過歷史均價 1.5 倍，建議啟動議價程序。"
        else:
            report += "✅ **【報價合理】** 報價落於該廠商之歷史合理區間內。"
        return report

    return f"⚠️ 查無與 `{vendor_clean}` 相關的歷史報價紀錄。"



def generate_contract_from_template(
    template_path: str, output_path: str, fields: Dict[str, Any]
) -> bool:
    try:
        doc = DocxTemplate(template_path)
        context = fields.copy()
        context["today"] = datetime.date.today().strftime("%Y年%m月%d日")

        if "amount" in context and context["amount"]:
            try:
                clean_amount = re.sub(r"[^\d]", "", str(context["amount"]))
                context["amount_formatted"] = f"{int(clean_amount):,}" if clean_amount else ""
            except ValueError:
                context["amount_formatted"] = str(context["amount"])

        if "term" in context and context["term"]:
            context["term"] = normalize_term(context["term"])

        doc.render(context)
        doc.save(output_path)
        return True
    except Exception as e:
        logging.error(f"合約生成失敗: {e}")
        return False



def llm_chat(messages: List[Dict[str, str]], draft_text: str = "", review_context: dict = None) -> str:
    """
    法務助理自由對話。

    核心設計：
    1. 「根據審查結果補條款」交給 clause_followup_service 處理。
    2. 條款補寫結果由程式從 review_context 抽取，不讓 LLM 自由生成。
    3. 只有一般聊天、合約草稿討論、審查結果說明才交給 LLM。
    """

    latest_user_input = ""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_input = str(msg.get("content", "") or "").strip()
            break

    has_draft = bool(draft_text and draft_text.strip())
    has_review_context = bool(review_context and isinstance(review_context, dict))
    if has_review_context:
        clause_answer = answer_clause_followup(
            user_input=latest_user_input,
            review_context=review_context,
        )

        if clause_answer:
            return clause_answer
    if not has_draft and not has_review_context:
        system_prompt = (
            "你是一個專業的法務合約助理。請全程使用正體中文（繁體字）回答，禁止使用簡體字。\n"
            "請回答使用者的問題，可以解釋法律、合約、資安、採購或企業合規相關概念。\n"
            "若使用者只是詢問概念或定義，請直接解釋，不要自行假設正在審查某份合約。\n"
            "不得把使用者的一般提問判定為合約缺漏、風險或不合規事項。\n"
            "若問題需要正式法律意見，請提醒仍需由法務或律師複核。\n"
        )
    elif has_draft and not has_review_context:
        system_prompt = (
            "你是專業的企業法務與合規 AI 助理。\n"
            "目前使用者提供了一份合約草稿，你可以協助解釋、整理、摘要或提出初步修改方向。\n"
            "請注意：若使用者只是問一般概念，不要把該概念直接判定為合約缺漏。\n"
            "若要指出合約問題，必須明確引用合約草稿中實際存在的文字。\n"
            "不得憑空新增合約中沒有出現的風險事實。\n"
            "若資料不足，請明確說明「目前草稿未提供足夠資訊」。\n"
        )

        system_prompt += (
            f"\n\n【目前正在處理的合約草稿內容（部分擷取）】：\n"
            f"{draft_text[:2000]}"
        )
    else:
        system_prompt = (
            "你是專業的企業合規與法務 AI 助理，正在協助使用者理解前一次合約審查結果。\n"
            "請嚴格區分以下三種資訊：\n"
            "1. 合約原文明確存在的內容\n"
            "2. 系統審查報告指出的風險或缺漏\n"
            "3. 使用者的一般提問或補充問題\n\n"
            "重要規則：\n"
            "若使用者只是詢問概念、定義或背景知識，不得把該提問當成新的合約缺漏。\n"
            "若使用者詢問審查報告內容，請根據下方審查報告回答。\n"
            "若資料不足，請明確說明資料不足。\n"
            "不得自行新增審查報告中沒有的義務、時限、頻率、金額、比例、標準、流程或報告格式。\n"
        )

        if has_draft:
            system_prompt += (
                f"\n\n【目前正在處理的合約草稿內容（部分擷取）】：\n"
                f"{draft_text[:2000]}"
            )

        system_prompt += "\n\n【系統先前的合規檢核報告資訊】：\n"

        issues = (
            review_context.get("major_issues", []) or []
        ) + (
            review_context.get("general_issues", []) or []
        )

        if issues:
            system_prompt += "\n🚨 發現的風險與建議：\n"
            for issue in issues[:6]:
                system_prompt += (
                    f"- 條款名稱：{issue.get('clause')}\n"
                    f"  風險分析：{issue.get('analysis')}\n"
                    f"  建議修正與協商方案：{issue.get('suggestion')}\n"
                    f"  建議調整後條文 adjusted_clause：{issue.get('adjusted_clause', '未提供')}\n"
                    f"  基準原文 template_basis：{issue.get('template_basis', '無提供')}\n"
                )

        missing = review_context.get("missing_clauses", []) or []
        if missing:
            system_prompt += "\n🧩 缺漏的合規條款：\n"
            for item in missing[:8]:
                system_prompt += (
                    f"- 應補入條款：{item.get('clause')}\n"
                    f"  缺漏原因：{item.get('why_missing')}\n"
                    f"  建議補充條文 suggested_draft：{item.get('suggested_draft', '無提供')}\n"
                    f"  基準原文 template_snippet：{item.get('template_snippet', '無提供')}\n"
                )

        compliance_scan = review_context.get("compliance_scan", []) or []
        uncovered = [
            item for item in compliance_scan
            if not item.get("is_covered", True)
        ]

        if uncovered:
            system_prompt += "\n⚠️ 章則規範要求但合約尚未涵蓋的廠商義務：\n"
            for item in uncovered[:8]:
                system_prompt += (
                    f"- 義務項目：{item.get('requirement')}\n"
                    f"  缺漏風險：{item.get('gap_description', '未說明')}\n"
                    f"  建議補充條文 suggested_addition：{item.get('suggested_addition', '無提供')}\n"
                )

        gap_analysis = review_context.get("gap_analysis", {}) or {}
        gaps = gap_analysis.get("gaps", []) or []

        if gaps:
            system_prompt += "\n📊 跨合約差距分析重點：\n"
            for gap in gaps[:5]:
                scenario = gap.get("vendor_refuse_scenario", {}) or {}
                system_prompt += (
                    f"- 差距主題：{gap.get('topic')}\n"
                    f"  歷史合約作法：{gap.get('other_vendors_coverage', '無資料')}\n"
                    f"  本次廠商現況：{gap.get('current_vendor_status', '未知')}\n"
                    f"  廠商拒絕風險：{scenario.get('risk_description', '未說明')}\n"
                    f"  替代條文建議：{scenario.get('alternative_clause', '無提供')}\n"
                )

    ollama_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role not in ["system", "user", "assistant"]:
            role = "user"

        # 避免外部傳入 system 覆蓋系統規則
        if role == "system":
            continue

        ollama_messages.append({
            "role": role,
            "content": content,
        })

    try:
        logging.info(f"llm_chat 使用模型：{CHAT_MODEL}")

        res = ollama.chat(
            model=CHAT_MODEL,
            messages=ollama_messages,
            options={
                "temperature": 0.1,
                "top_p": 0.4,
            },
        )

        return ensure_traditional(
            res.get("message", {}).get("content", "系統無法產生回覆。")
        )

    except Exception as e:
        logging.error(f"Ollama 對話失敗: {e}")
        return "對話服務暫時無法使用，請確認 Ollama 模型已啟動。"


def emit_progress(progress_callback, stage: str, message: str, percent: int):
    if callable(progress_callback):
        progress_callback({
            "stage": stage,
            "message": message,
            "percent": percent,
            "ts": datetime.datetime.now().isoformat(),
        })
