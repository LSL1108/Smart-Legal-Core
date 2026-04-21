import os
import re
import time
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
from models import UserRequestIntent, ReviewReport

import ollama
from docxtpl import DocxTemplate

from config import (
    MODEL, ALL_TOPICS_FOR_PROMPT, TOPIC_KEYWORDS, TOPIC_ALIAS,
    TOPIC_MIN_MATCHES, CRITICAL_RISK_TRIGGERS, HIGH_RISK_TRIGGERS
)

from utils import (
    safe_json_load, normalize_text, detect_topics, lexical_score, score_topic_overlap,
    detect_contract_mode_from_text, build_article_map, normalize_topic_name,
    article_to_key, short_text, split_draft_into_articles, extract_text_from_pdf,
    extract_text_from_docx, make_output_path, normalize_term, save_upload_file,
    parse_template_selector, chunk_text
)

from database import (
    query_templates_fulltext, query_template_chunks_by_query, find_history_by_vendor_keyword,
    template_exists_by_sha256, insert_template_doc, upsert_template_vectors,
    get_template_by_selector, search_templates_sql,
    template_collection, get_compliance_rules, DEFAULT_COMPLIANCE_RULES
)


# Ollama

def ollama_json(
    prompt: str,
    model: str = MODEL,
    temperature: float = 0.0,
    top_p: float = 0.1,
    retries: int = 2,
) -> Dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            res = ollama.generate(
                model=model,
                prompt=prompt.strip(),
                format="json",
                options={"temperature": temperature, "top_p": top_p},
            )
            return safe_json_load((res or {}).get("response", "{}"))
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                logging.warning(f"Ollama JSON 失敗（第 {attempt + 1} 次），{wait}s 後重試：{e}")
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
{text[:7000]}

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


# RAG 檢索與關聯模板選擇

def guess_draft_contract_type(draft_text: str) -> str:
    prompt = f"""
請判斷以下合約草稿的類型。只輸出合約類型名稱（如：維護合約、保密協定、採購合約、租賃合約），不要輸出任何其他文字。
草稿內容：
{draft_text[:1500]}
"""
    try:
        res = ollama.generate(model=MODEL, prompt=prompt.strip())
        return str(res.get("response", "其他")).strip()
    except Exception:
        return "其他"



def _safe_str(value: Any) -> str:
    return str(value or "").strip()



def _safe_topics(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_topic_name(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [normalize_topic_name(x) for x in value.split(",") if x.strip()]
    return []



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
    if draft_type is None:
        draft_type = guess_draft_contract_type(draft_text)
    logging.info(f"LLM 判斷草稿類型為：{draft_type}")

    candidates = find_related_historical_templates(
        draft_text=draft_text,
        draft_type=draft_type,
        vendor_name=vendor_name,
        system_name=system_name,
        service_scope=service_scope,
        maintenance_type=maintenance_type,
        include_all_related=include_all_related,
        max_results=max_candidates,
    )

    if not candidates:
        logging.info("找不到結構化/語意歷史模板，退回全庫搜尋")
        candidates = query_templates_fulltext(draft_text, n_results=max(12, max_candidates)) or []
        candidates = [x for x in (_normalize_template_record(c) for c in candidates) if x]

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

        text_block = " ".join([
            c.get("file_name", ""),
            c.get("contract_type", ""),
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

        if vendor_name and c.get("vendor_name") == vendor_name:
            score += 15
        if system_name and c.get("system_name") == system_name:
            score += 12
        if service_scope and c.get("service_scope") == service_scope:
            score += 10
        if maintenance_type and c.get("maintenance_type") == maintenance_type:
            score += 8

        if (
            detect_contract_mode_from_text(draft_text) == "混合型"
            and c.get("contract_type") in ["維護合約", "保密協定", "開發合約"]
        ):
            score += 2

        ranked.append((score, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked]



def build_target_queries(
    draft_text: str,
    articles: List[Dict[str, Any]],
    selected_templates: List[Dict[str, Any]],
) -> List[str]:
    queries = [draft_text[:1500]]
    for article in articles:
        content = article.get("content", "")
        queries.append(content[:700])
        for topic in article.get("topics", []):
            kw = " ".join(TOPIC_KEYWORDS.get(topic, [])[:5])
            queries.append(f"{topic} {kw} {content[:240]}")

    template_topics: set = set()
    for t in selected_templates:
        template_topics.update(t.get("core_topics", []))
    for topic in template_topics:
        kw = " ".join(TOPIC_KEYWORDS.get(topic, [])[:5])
        queries.append(f"{topic} {kw}")

    dedup = []
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

    all_chunks: List[Dict[str, Any]] = []
    seen: set = set()
    for q in build_target_queries(draft_text, articles, selected_templates):
        refs = query_template_chunks_by_query(q, doc_ids, n_results=8)
        for r in refs:
            key = (r["doc_id"], r["content"][:150])
            if key not in seen:
                seen.add(key)
                all_chunks.append(r)

    # selected_templates 保留全集供比對，但 LLM 僅餵前 top_k 個高相關模板摘要
    return selected_templates, all_chunks[:40], articles



def search_template_chunks_for_article(
    article: Dict[str, Any],
    selected_templates: List[Dict[str, Any]],
    n_results: int = 10,
) -> List[Dict[str, Any]]:
    doc_ids = [x["doc_id"] for x in selected_templates if x.get("doc_id")]
    article_text = article.get("content", "")
    article_topics = article.get("topics", [])

    queries = [article_text[:900]]
    for topic in article_topics[:2]:
        kw_list = TOPIC_KEYWORDS.get(topic, [])
        if kw_list:
            queries.append(f"{topic} " + " ".join(kw_list[:6]))
        queries.append(f"{topic} {article_text[:120]}")
    queries = queries[:5]

    seen: set = set()
    out: List[Dict[str, Any]] = []

    for q in queries:
        refs = query_template_chunks_by_query(q, doc_ids, n_results=15)
        for r in refs:
            key = (r["doc_id"], r["content"][:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)

    scored = []
    for r in out:
        chunk_topics = [t for t in (r.get("topics") or "").split(",") if t.strip()]
        score = score_topic_overlap(article_topics, chunk_topics)
        score += lexical_score(r.get("content", ""), article_text[:300]) * 2

        r_content = normalize_text(r.get("content", ""))
        for t in article_topics:
            if any(k in r_content for k in TOPIC_KEYWORDS.get(t, [])):
                score += 15

        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:n_results]]


# 審查邏輯

def llm_review_single_article(
    article: Dict[str, Any],
    article_key: str,
    candidate_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    chunk_block = []
    source_names = []

    for i, c in enumerate(candidate_chunks, start=1):
        fname = c["file_name"]
        source_names.append(fname)
        topics = c.get("topics", "一般條款")
        chunk_block.append(f"【依據 {i}】\n對應主題：{topics}\n檔名：{fname}\n內容：{c['content']}")

    allowed_sources = "、".join(sorted(set(source_names))) if source_names else "無"

    draft_content = article.get("content", "")
    triggered_rules = []

    for t_category, triggers in CRITICAL_RISK_TRIGGERS.items():
        for trigger in triggers:
            if trigger in draft_content:
                triggered_rules.append(f"嚴重違規字眼：「{trigger}」(屬於 {t_category})")

    for t_category, triggers in HIGH_RISK_TRIGGERS.items():
        for trigger in triggers:
            if trigger in draft_content:
                triggered_rules.append(f"高風險字眼：「{trigger}」(屬於 {t_category})")

    rule_injection = ""
    if triggered_rules:
        rule_injection = (
            "\n\n【🚨 系統強制檢核指令】：\n"
            "系統已在以下草稿中偵測到觸發企業紅線的字眼：\n"
            + "、".join(triggered_rules) + "\n"
            "你「必須」將這些紅線字眼獨立列為一個 issue，具體分析該字眼造成的實質業務風險（不可含糊帶過），並給出強烈的協商建議！絕對不可與其他問題混為一談！\n"
        )

    prompt = f"""
你是企業合規與法務審查 AI，立場固定站在甲方。
請執行【雙軌審查】：
（一）與歷史基準對齊：找出廠商草稿和企業歷史慣例的差距，給出調整建議；
（二）風險偵測：找出合規偏離、條文衝突或遺漏。
{rule_injection}

嚴格限制：
1. 只能輸出 JSON，不可輸出多餘的文字。
2. 判斷標準【必須完全依據歷史基準片段】或【系統強制檢核指令】。絕不可憑空捏造。
3. 【嚴格強制】：issue_topic 欄位必須「完全照抄」【歷史基準與法遵片段】中標示的「對應主題」。
4. 若單一草稿條款包含多個不同層面的風險，請務必窮盡列出，並拆分成多筆獨立的 JSON 物件。
5. analysis（合規落差分析）：具體說明廠商草稿與企業歷史基準的差異，以及對甲方的潛在資安或法遵風險。
6. suggestion（建議修正與協商方案）：直接依據基準的具體標準給出修改方向。並請針對「若廠商無法配合」，主動提議實務上的替代協商方案。
7. adjusted_clause（建議修改後條文）：根據歷史基準原文，給出可直接貼回合約的具體修改後條文（100字以內）。若完全符合基準則填「符合，無需修改」。
8. negotiation_notes（協商備忘）：若廠商拒絕修改，可接受的最低底線條件，及建議的折衷方案（50字以內）。
9. template_snippet（基準原文）：請從【歷史基準與法遵片段】中，精準擷取最相關的「原文字句」。
10. source 欄位必須填入對應的真實檔名：{allowed_sources}。絕不可留空。

輸出格式：
{{
"major_issues": [
  {{
    "article_key": "{article_key}",
    "clause": "條款名稱",
    "issue_topic": "對應的主題",
    "type": "deviation 或 conflict",
    "risk": "Critical/High/Medium/Low",
    "template_snippet": "(動態生成) 乙方應於系統上線前完成弱點掃描並無償修補...",
    "analysis": "(動態生成) 草稿約定掃描修補需另行收費，不符合我方過往免費修補之合規基準。",
    "suggestion": "(動態生成) 建議要求廠商依過往合約無償修補。若廠商拒絕，建議釐清修補計費標準。",
    "adjusted_clause": "(動態生成) 乙方應於每次版本更新後30日內完成弱點掃描，並無償提供修補服務，修補結果應以書面通知甲方。",
    "negotiation_notes": "(動態生成) 最低底線：掃描費用甲乙雙方各半；修補逾期按日罰款。",
    "source": "資安規範.docx"
  }}
],
"general_issues": []
}}

【廠商草稿條文】
{article.get("content", "")}

【歷史基準與法遵片段】
{chr(10).join(chunk_block)}
"""
    return ollama_json(prompt)



def infer_missing_topics_from_templates(
    selected_templates: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    found_topics_by_llm: set,
    all_candidate_chunks: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    chunk_evidenced: Dict[str, Tuple[str, str]] = {}

    for chunk in all_candidate_chunks:
        chunk_content = normalize_text(chunk.get("content", ""))
        fname = chunk.get("file_name", "")
        pattern = r"(第[一二三四五六七八九十百0-9]+條[：:\s].*?)(?=(?:\n?第[一二三四五六七八九十百0-9]+條[：:\s])|$)"
        sub_articles = re.findall(pattern, chunk_content, flags=re.S)

        if not sub_articles:
            sub_articles = [chunk_content]

        for topic, keywords in TOPIC_KEYWORDS.items():
            if topic in chunk_evidenced:
                continue
            min_hits = TOPIC_MIN_MATCHES.get(topic, 1)
            best_snippet = ""
            for sub_art in sub_articles:
                hits = sum(1 for k in keywords if k in sub_art)
                if hits >= min_hits:
                    best_snippet = sub_art.strip()
                    break

            if best_snippet:
                snippet = best_snippet[:400] + ("..." if len(best_snippet) > 400 else "")
                chunk_evidenced[topic] = (fname, snippet)

    draft_topics: set = set()
    for a in articles:
        draft_topics.update(a.get("topics", []))
    all_found = draft_topics.union(found_topics_by_llm)

    template_declared: set = set()
    for t in selected_templates:
        template_declared.update(t.get("core_topics", []))

    missing: List[Tuple[str, str, str]] = []
    for topic in sorted(template_declared):
        if topic in all_found:
            continue
        if topic not in chunk_evidenced:
            continue
        fname, snippet = chunk_evidenced[topic]
        missing.append((topic, fname, snippet))

    return missing


# 批次起草所有缺漏條文

def _draft_all_missing_clauses(
    missing_topic_tuples: List[Tuple[str, str, str]]
) -> Dict[str, str]:
    if not missing_topic_tuples:
        return {}

    items_text = "\n".join(
        f"{i+1}. 主題：{topic}\n   歷史基準片段（節錄）：{snippet[:200]}"
        for i, (topic, _, snippet) in enumerate(missing_topic_tuples)
    )

    prompt = f"""
請為以下每個缺漏主題，各起草一條「廠商應負責事項」的補充條文（每條80字以內，以「乙方應」開頭）。
只輸出 JSON，不要任何說明文字：
{{
  "drafts": [
    {{"topic": "主題名稱", "clause": "乙方應...條文內容"}},
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
    return {
        d["topic"]: d["clause"]
        for d in drafts
        if isinstance(d, dict) and "topic" in d and "clause" in d
    }



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
                "topics": chunk.get("topics", ""),
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
    selected_templates = selected_templates or []
    all_candidate_chunks = all_candidate_chunks or []
    full_text = "\n".join(a.get("content", "") for a in articles)

    dynamic_rules = get_compliance_rules() or DEFAULT_COMPLIANCE_RULES

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
        f"- 【{topic}】：需明訂（例如：{examples[0] if examples else ''}）"
        for topic, examples in dynamic_rules.items()
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
你是企業法遵稽核員，立場站在甲方。依據下列「章則規範要求清單」，逐項檢查廠商合約中是否已包含對應的「廠商協助義務條款」。

【章則規範要求清單】（公司規範要求合約必須涵蓋）
{obligation_list}

【廠商合約全文（節錄）】
{full_text[:6000]}

【歷史合約與規範證據】
{chr(10).join(evidence_text_parts)[:12000]}

請注意：
1. 不只判斷草稿有沒有寫，還要判斷是否符合最新規範要求的責任深度。
2. 若草稿把應由廠商協助處理的事項改成另行報價、另簽、另付費、甲方自理，也應視為可能未充分涵蓋。
3. 若歷史合約多數有寫，現在草稿沒寫，要明確指出差距。
4. 若廠商不免費處理，suggested_addition 可補充費用、分工、修補時限或通知方式。

請針對每個規範項目輸出 JSON，只輸出 JSON 不要有其他文字：
{{
  "compliance_scan_results": [
    {{
      "requirement": "章則規範要求的義務名稱",
      "is_covered": true,
      "found_clause": "若有涵蓋，合約中找到的對應文字（直接引用原文）；若無則填 null",
      "gap_description": "若未涵蓋，說明缺漏的具體風險；若已涵蓋但責任不足，也要說明不足之處；若完全已涵蓋則填 null",
      "suggested_addition": "若未涵蓋或涵蓋不足，建議補入的合約條文草稿（可直接貼入合約，80字以內）；若已涵蓋則填 null"
    }}
  ]
}}
"""
    result = ollama_json(prompt)
    scan_results = result.get("compliance_scan_results", [])

    normalized = []
    seen_requirements = set()
    for r in scan_results:
        if not isinstance(r, dict):
            continue
        requirement = str(r.get("requirement", "")).strip()
        if not requirement:
            continue
        seen_requirements.add(requirement)
        found_clause = r.get("found_clause") or None
        gap_description = r.get("gap_description") or None
        suggested_addition = r.get("suggested_addition") or None

        is_covered = bool(r.get("is_covered", False))
        if gap_description and is_covered:
            # 有明確 gap 說明時，將其視為未完整覆蓋
            is_covered = False

        normalized.append({
            "requirement": requirement,
            "is_covered": is_covered,
            "found_clause": found_clause,
            "gap_description": gap_description,
            "suggested_addition": suggested_addition,
        })

    # 模型沒吐全時，補齊缺漏 requirement，避免報告不完整
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
        normalized.append({
            "requirement": topic,
            "is_covered": bool(evidence["draft_hits"]),
            "found_clause": evidence["draft_hits"][0] if evidence["draft_hits"] else None,
            "gap_description": None if evidence["draft_hits"] else "草稿未明確約定此項廠商協助義務，且需依最新章則及歷史合約補強。",
            "suggested_addition": None if evidence["draft_hits"] else f"乙方應配合辦理{topic}相關作業，並依甲方要求提供必要協助、修補、回報與文件。",
        })

    return normalized



def cross_contract_gap_analysis(
    current_draft_text: str,
    selected_templates: List[Dict[str, Any]],
    compliance_scan: List[Dict[str, Any]],
    all_candidate_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    all_candidate_chunks = all_candidate_chunks or []

    uncovered = [r for r in compliance_scan if not r.get("is_covered", True)]
    if not uncovered:
        return {
            "gap_summary": "本次合約已涵蓋所有章則規範要求的廠商義務，無明顯差距。",
            "gaps": []
        }

    gap_evidence_blocks = []
    for r in uncovered:
        topic = r.get("requirement", "")
        aliases = set([topic])
        aliases.update(TOPIC_ALIAS.get(topic, []))
        aliases.update(TOPIC_KEYWORDS.get(topic, []))
        aliases = {normalize_text(x) for x in aliases if str(x).strip()}

        related_chunks = []
        for c in all_candidate_chunks:
            content = c.get("content", "")
            ncontent = normalize_text(content)
            chunk_topics = normalize_text(c.get("topics", ""))
            if any(a and (a in ncontent or a in chunk_topics) for a in aliases):
                related_chunks.append(
                    f"- 【{c.get('file_name', '未知檔案')}】主題:{c.get('topics', '')} 內容:{content[:220]}"
                )

        if not related_chunks:
            for t in selected_templates[:10]:
                src = t.get("source_text", "")
                nsrc = normalize_text(src)
                if any(a and a in nsrc for a in aliases):
                    related_chunks.append(
                        f"- 【{t.get('file_name', '未知檔案')}】摘要:{t.get('summary', '')[:120]} 內容:{src[:220]}"
                    )

        gap_evidence_blocks.append(
            f"【{topic}】\n"
            f"本次草稿缺口：{r.get('gap_description', '未提供義務條款')}\n"
            f"草稿現有文字：{r.get('found_clause') or '未找到'}\n"
            f"歷史合約證據：\n{chr(10).join(related_chunks[:6]) if related_chunks else '- 無明確歷史片段'}"
        )

    prompt = f"""
你是企業採購法務顧問，立場站在甲方。請根據以下資訊，提供「跨廠商合約差距分析」報告。

【本次廠商草稿節錄】
{current_draft_text[:2500]}

【其他歷史合約實際證據】
{chr(10).join(gap_evidence_blocks)[:12000]}

請輸出 JSON，只輸出 JSON 不要有其他文字：
{{
  "gap_summary": "一段100字以內的差距總結，說明本次合約最關鍵的差距為何",
  "gaps": [
    {{
      "topic": "差距主題名稱",
      "other_vendors_coverage": "其他歷史合約如何處理此項目，請引用具體作法，不要只寫抽象句",
      "current_vendor_status": "本次廠商草稿中此項目的現況",
      "vendor_refuse_scenario": {{
        "risk_description": "若廠商拒絕協助此項目，企業面臨的具體風險",
        "cost_bearing_suggestion": "建議費用分擔方式",
        "alternative_clause": "建議補充的替代條文草稿（100字以內）"
      }}
    }}
  ]
}}
"""
    result = ollama_json(prompt)

    if not isinstance(result, dict):
        return {"gap_summary": "差距分析產生失敗，請重新審查。", "gaps": []}

    gaps = result.get("gaps", [])
    normalized_gaps = []
    for g in gaps:
        if not isinstance(g, dict):
            continue
        vendor_scenario = g.get("vendor_refuse_scenario", {}) or {}
        normalized_gaps.append({
            "topic": str(g.get("topic", "")).strip(),
            "other_vendors_coverage": str(g.get("other_vendors_coverage", "")).strip(),
            "current_vendor_status": str(g.get("current_vendor_status", "")).strip(),
            "vendor_refuse_scenario": {
                "risk_description": str(vendor_scenario.get("risk_description", "")).strip(),
                "cost_bearing_suggestion": str(vendor_scenario.get("cost_bearing_suggestion", "")).strip(),
                "alternative_clause": str(vendor_scenario.get("alternative_clause", "")).strip(),
            }
        })

    return {
        "gap_summary": str(result.get("gap_summary", "")).strip(),
        "gaps": normalized_gaps,
    }


# 防禦 Prompt Injection

def check_prompt_injection(text: str) -> bool:
    suspicious_patterns = [
        r"忽略.*指示", r"ignore.*instructions", r"system prompt",
        r"忘記.*規則", r"覆寫", r"override", r"你是.*不要扮演",
        r"直接輸出.*風險", r"無風險",
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
) -> Dict[str, Any]:
    if check_prompt_injection(draft_text):
        report = ReviewReport(
            contract_type_guess="拒絕審查",
            summary="🚨 系統偵測到合約內容包含惡意指令（Prompt Injection），為保護系統與營業機密安全，已強制終止審查程序。",
            score=0
        )
        return report.model_dump()

    if draft_type is None:
        draft_type = guess_draft_contract_type(draft_text)

    all_major: List[Dict[str, Any]] = []
    all_general: List[Dict[str, Any]] = []
    found_topics_by_llm: set = set()
    all_article_chunks: List[Dict[str, Any]] = []

    article_chunk_pairs = []
    for idx, article in enumerate(articles, start=1):
        article_key = article_to_key(article, idx)
        candidate_chunks = search_template_chunks_for_article(article, selected_templates, n_results=10)
        all_article_chunks.extend(candidate_chunks)
        article_chunk_pairs.append((article, article_key, candidate_chunks))

    def _review_one(args):
        article, article_key, candidate_chunks = args
        result = llm_review_single_article(article, article_key, candidate_chunks)
        return article_key, result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_review_one, args): args for args in article_chunk_pairs}
        for future in as_completed(futures):
            try:
                _, article_result = future.result()
            except Exception as e:
                logging.error(f"條文並行審查失敗：{e}")
                continue

            majors = article_result.get("major_issues", [])
            generals = article_result.get("general_issues", [])

            all_major.extend(majors)
            all_general.extend(generals)

            for issue in majors + generals:
                topic = issue.get("issue_topic")
                if topic:
                    found_topics_by_llm.add(normalize_topic_name(topic))

    missing_topic_tuples = infer_missing_topics_from_templates(
        selected_templates, articles, found_topics_by_llm, all_article_chunks
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

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_drafts = executor.submit(_run_batch_draft)
        future_compliance = executor.submit(_run_compliance_scan)
        all_clause_drafts = future_drafts.result()
        compliance_scan = future_compliance.result()

    missing_clauses = []
    for topic, source, snippet in missing_topic_tuples:
        suggested_draft = all_clause_drafts.get(
            topic,
            f"乙方應依企業規範履行「{topic}」相關義務，具體標準參照甲方歷史合約基準。"
        )
        missing_clauses.append({
            "clause": topic,
            "issue_topic": topic,
            "why_missing": f"企業歷史基準或法規文檔【{source}】中有明確相關規範，但目前草稿中未找到對應保障。",
            "suggestion": f"建議參考歷史基準中關於「{topic}」的條款，補入具體規定以確保合規。",
            "suggested_draft": suggested_draft,
            "source": source,
            "template_snippet": snippet,
        })

    return {
        "contract_type_guess": draft_type,
        "summary": "系統已根據動態檢索出的歷史合約全集與法遵規範，逐條進行合規落差掃描，並完成章則義務稽核。",
        "major_issues": all_major,
        "general_issues": all_general,
        "missing_clauses": missing_clauses,
        "compliance_scan": compliance_scan,
        "all_candidate_chunks": all_article_chunks,
    }



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

    def norm_issue(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        article_key = str(item.get("article_key", "") or "").strip()
        clause = str(item.get("clause", "未命名項目")).strip()
        topic = normalize_topic_name(str(item.get("issue_topic", clause) or clause))
        source = str(item.get("source", "") or "").strip()

        article = article_map.get(article_key, {"content": ""})
        draft_text = normalize_text(article.get("content", "") or "")
        extracted_snippet = str(item.get("template_snippet", "詳見基準內容")).strip()

        issue = {
            "clause": clause,
            "risk": str(item.get("risk", "Medium")).strip(),
            "draft_text": draft_text,
            "template_basis": extracted_snippet,
            "analysis": normalize_text(str(item.get("analysis", ""))),
            "suggestion": normalize_text(str(item.get("suggestion", ""))),
            "adjusted_clause": str(item.get("adjusted_clause", "")).strip(),
            "negotiation_notes": str(item.get("negotiation_notes", "")).strip(),
            "source": source,
            "type": str(item.get("type", "deviation")).strip(),
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
                    issue["risk"] = "High"
                    break

        return issue

    raw_major = [x for x in (norm_issue(i) for i in raw.get("major_issues", [])) if x]
    raw_general = [x for x in (norm_issue(i) for i in raw.get("general_issues", [])) if x]

    def dedup_issues(items):
        merged = {}
        for it in items:
            key = (
                it.get("article_key", ""),
                it.get("issue_topic", ""),
                it.get("type", ""),
            )
            if key not in merged:
                merged[key] = it.copy()
            else:
                if it.get("analysis", "") and it.get("analysis", "") not in merged[key]["analysis"]:
                    merged[key]["analysis"] += f"\n\n🔸 **其他關聯風險**：{it.get('analysis', '')}"
                if it.get("suggestion", "") and it.get("suggestion", "") not in merged[key]["suggestion"]:
                    merged[key]["suggestion"] += f"\n\n🔧 **補充建議**：{it.get('suggestion', '')}"

                new_risk = str(it.get("risk", "")).lower()
                curr_risk = str(merged[key].get("risk", "")).lower()

                if new_risk == "critical":
                    merged[key]["risk"] = "Critical"
                    merged[key]["type"] = "conflict"
                elif new_risk == "high" and curr_risk not in ["critical", "high"]:
                    merged[key]["risk"] = "High"

        return list(merged.values())

    major_issues = dedup_issues(raw_major)
    general_issues = dedup_issues(raw_general)

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

    missing_count = len(raw.get("missing_clauses", []))
    current_score -= missing_count * 3

    compliance_scan = raw.get("compliance_scan", [])
    uncovered_count = sum(1 for r in compliance_scan if not r.get("is_covered", True))
    current_score -= uncovered_count * 5

    final_score = max(0, current_score)

    try:
        report = ReviewReport(
            contract_type_guess=raw.get("contract_type_guess", "未判定"),
            summary="系統已根據歷史合約與法遵規範，完成草稿比對，並結合章則規範義務稽核與跨合約差距分析，整理出完整合規落差報告。",
            used_templates=uniq_templates,
            major_issues=major_issues,
            general_issues=general_issues,
            missing_clauses=raw.get("missing_clauses", []),
            score=final_score
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
            score=0
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
    system_prompt = (
        "你是專業的企業合規與法務 AI 助理。請協助使用者審閱、修改合約，或解答法務問題。\n"
        "【最高防幻覺指令】：當使用者要求撰寫或補齊條款時，你「必須 100% 照抄」【企業歷史基準與法遵參考字句】中的原文進行微調即可。\n"
        "🚫 絕對禁止自行擴寫內容。\n"
        "🚫 絕對禁止補充行業慣例。\n"
        "🚫 絕對禁止發明規範中沒有的流程（如：測試報告、未提及的天數、資安標準等）。\n"
        "💡 若基準原文很簡短，你的回答就必須一樣簡短，維持企業既有原貌。\n"
    )

    if draft_text:
        system_prompt += f"\n\n【目前正在處理的合約草稿內容（部分擷取）】：\n{draft_text[:2000]}"

    if review_context:
        system_prompt += "\n\n【系統先前的合規檢核報告資訊】：\n"

        issues = review_context.get("major_issues", []) + review_context.get("general_issues", [])
        if issues:
            system_prompt += "\n🚨 發現的風險與建議：\n"
            for i in issues[:5]:
                system_prompt += (
                    f"- 條款名稱：{i.get('clause')}\n"
                    f"  風險分析：{i.get('analysis')}\n"
                    f"  建議修正與協商方案：{i.get('suggestion')}\n"
                    f"  建議調整後條文：{i.get('adjusted_clause', '未提供')}\n"
                    f"  基準原文：{i.get('template_basis', '無提供')}\n"
                )

        missing = review_context.get("missing_clauses", [])
        if missing:
            system_prompt += "\n🧩 缺漏的合規條款：\n"
            for m in missing[:6]:
                system_prompt += (
                    f"- 應補入條款：{m.get('clause')}\n"
                    f"  缺漏原因：{m.get('why_missing')}\n"
                    f"  建議補充條文草稿：{m.get('suggested_draft', '無提供')}\n"
                    f"  基準原文：{m.get('template_snippet', '無提供')}\n"
                )

        compliance_scan = review_context.get("compliance_scan", [])
        uncovered = [r for r in compliance_scan if not r.get("is_covered", True)]
        if uncovered:
            system_prompt += "\n⚠️ 章則規範要求但合約尚未涵蓋的廠商義務：\n"
            for r in uncovered[:5]:
                system_prompt += (
                    f"- 義務項目：{r.get('requirement')}\n"
                    f"  缺漏風險：{r.get('gap_description', '未說明')}\n"
                    f"  建議補充條文：{r.get('suggested_addition', '無提供')}\n"
                )

        gap_analysis = review_context.get("gap_analysis", {})
        gaps = gap_analysis.get("gaps", [])
        if gaps:
            system_prompt += "\n📊 跨合約差距分析重點：\n"
            for g in gaps[:3]:
                scenario = g.get("vendor_refuse_scenario", {})
                system_prompt += (
                    f"- 差距主題：{g.get('topic')}\n"
                    f"  歷史合約作法：{g.get('other_vendors_coverage', '無資料')}\n"
                    f"  本次廠商現況：{g.get('current_vendor_status', '未知')}\n"
                    f"  廠商拒絕風險：{scenario.get('risk_description', '未說明')}\n"
                    f"  替代條文建議：{scenario.get('alternative_clause', '無提供')}\n"
                )

    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        ollama_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        res = ollama.chat(
            model=MODEL,
            messages=ollama_messages,
            options={"temperature": 0.3, "top_p": 0.8}
        )
        return res.get("message", {}).get("content", "系統無法產生回覆。")
    except Exception as e:
        logging.error(f"Ollama 對話失敗: {e}")
        return "對話服務暫時無法使用，請確認 Ollama 模型已啟動。"
