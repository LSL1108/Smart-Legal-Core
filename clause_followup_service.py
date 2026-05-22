import re
import logging
from typing import Any, Dict, List, Tuple, Optional

from config import TOPIC_ALIAS, TOPIC_KEYWORDS


logger = logging.getLogger(__name__)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    bad_values = {
        "",
        "未提供",
        "無提供",
        "none",
        "null",
        "None",
        "Null",
        "N/A",
        "na",
    }
    return "" if text in bad_values else text


def normalize_for_match(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", "", text)

    replacements = {
        "資訊安全": "資安",
        "資通安全": "資安",
        "弱點掃描": "弱掃",
        "弱點檢測": "弱掃",
        "弱點修復": "弱點修補",
        "漏洞修復": "弱點修補",
        "安全漏洞修復": "弱點修補",
        "漏洞掃描": "弱掃",
        "安全掃描": "弱掃",
        "資安掃描": "弱掃",
        "安全性測試": "資安檢測",
        "安全測試": "資安檢測",
        "安全事件": "資安事件",
        "事故通報": "事件通報",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_clause_followup_request(user_input: str, has_review_context: bool = True) -> bool:
    text = normalize_for_match(user_input)

    clause_words = [
        "條款",
        "條文",
        "建議條文",
        "補充條文",
        "修改後條文",
        "怎麼補",
        "如何補",
        "補入",
        "加入",
        "新增",
        "修改",
        "改成",
        "怎麼寫",
        "如何寫",
        "怎麼撰寫",
        "如何撰寫",
        "撰寫",
        "草擬",
        "補充",
        "可直接貼入",
        "整理成條文",
        "寫進合約",
        "列出來",
        "列出",
    ]

    context_words = [
        "根據剛剛",
        "根據前面",
        "根據上述",
        "審查結果",
        "合規檢核",
        "剛剛的審查",
        "這份合約",
        "上述合約",
        "前次審查",
    ]

    return any(w in text for w in clause_words) and (
        has_review_context or any(w in text for w in context_words)
    )


def is_all_clause_request(user_input: str) -> bool:
    text = normalize_for_match(user_input)

    all_words = [
        "全部",
        "所有",
        "每一個",
        "完整",
        "一次列",
        "全部列",
        "都列",
        "全部缺漏",
        "所有缺漏",
        "所有條款",
        "完整條款",
    ]

    clause_words = [
        "條款",
        "缺漏",
        "補入",
        "補充",
        "建議條文",
        "怎麼補",
        "如何補",
        "列出",
        "列出來",
    ]

    return any(w in text for w in all_words) and any(w in text for w in clause_words)


def is_all_missing_obligation_request(user_input: str) -> bool:
    text = normalize_for_match(user_input)

    return (
        is_all_clause_request(user_input)
        and any(w in text for w in ["缺漏", "義務", "尚未涵蓋", "未涵蓋"])
        and not any(w in text for w in ["所有風險", "全部風險", "建議修改", "修正建議", "重大風險"])
    )


def is_all_revision_request(user_input: str) -> bool:

    text = normalize_for_match(user_input)

    return (
        is_all_clause_request(user_input)
        and any(w in text for w in ["所有風險", "全部風險", "建議修改", "修正建議", "重大風險", "所有建議"])
    )


def build_requirement_aliases(requirement: str) -> List[str]:
    req = str(requirement or "").strip()
    aliases = {req}

    static_aliases = {
        "資安檢測與掃描": [
            "資安檢測",
            "資安掃描",
            "弱點掃描",
            "弱掃",
            "漏洞掃描",
            "安全掃描",
            "安全性評估",
            "資安評估",
        ],
        "弱點修補與維護": [
            "弱點修補",
            "弱點修復",
            "漏洞修補",
            "安全漏洞修復",
            "漏洞修復",
            "修補期限",
            "弱點維護",
            "漏洞維護",
            "弱點修復與維護",
            "弱點修補與維護",
            "弱點修復條款",
            "弱點修補條款",
        ],
        "事件通報與應變": [
            "事件通報",
            "資安事件通報",
            "通報",
            "應變",
            "資安事件",
            "事故通報",
            "安全事件",
        ],
        "個資保護與保密": [
            "個資保護",
            "個人資料",
            "個資",
            "保密",
            "資料保護",
            "機密資訊",
            "個資法",
        ],
        "備份與災難復原": [
            "備份",
            "災難復原",
            "備份復原",
            "資料備份",
            "災復",
            "復原演練",
            "災難復原演練",
        ],
    }

    if req in static_aliases:
        aliases.update(static_aliases[req])

    try:
        aliases.update(TOPIC_ALIAS.get(req, []))
    except Exception:
        pass

    try:
        aliases.update(TOPIC_KEYWORDS.get(req, []))
    except Exception:
        pass

    normalized_aliases = []
    seen = set()

    for alias in aliases:
        normalized = normalize_for_match(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_aliases.append(normalized)

    return normalized_aliases


def score_requirement_match(user_input: str, requirement: str) -> int:
    user_norm = normalize_for_match(user_input)
    req_norm = normalize_for_match(requirement)
    aliases = build_requirement_aliases(requirement)

    score = 0

    if req_norm and req_norm in user_norm:
        score += 120

    for alias in aliases:
        if alias and alias in user_norm:
            score += 45 + min(len(alias), 25)

    if any(k in user_norm for k in ["資安檢測", "資安掃描", "弱掃", "漏洞掃描", "安全掃描"]):
        if req_norm == normalize_for_match("資安檢測與掃描"):
            score += 100
        if req_norm == normalize_for_match("弱點修補與維護"):
            score -= 80

    if any(k in user_norm for k in ["弱點修補", "弱點修復", "修補", "修復", "漏洞修復", "漏洞修補", "修補期限", "弱點維護"]):
        if req_norm == normalize_for_match("弱點修補與維護"):
            score += 140
        if req_norm == normalize_for_match("資安檢測與掃描"):
            score -= 80

    if any(k in user_norm for k in ["事件通報", "資安事件", "通報", "應變"]):
        if req_norm == normalize_for_match("事件通報與應變"):
            score += 100
        if req_norm in [
            normalize_for_match("資安檢測與掃描"),
            normalize_for_match("弱點修補與維護"),
        ]:
            score -= 40

    if any(k in user_norm for k in ["個資", "個人資料", "保密", "資料保護"]):
        if req_norm == normalize_for_match("個資保護與保密"):
            score += 100

    if any(k in user_norm for k in ["備份", "災難復原", "災復", "復原演練"]):
        if req_norm == normalize_for_match("備份與災難復原"):
            score += 100

    return score


def collect_clause_candidates(
    review_context: Dict[str, Any],
    include_issues: bool = True,
    include_gap_analysis: bool = True,
) -> List[Dict[str, Any]]:
    """
    從 review_context 蒐集可直接引用的條款建議。
    """
    candidates: List[Dict[str, Any]] = []

    if not isinstance(review_context, dict):
        return candidates

    for r in review_context.get("compliance_scan", []) or []:
        if not isinstance(r, dict):
            continue

        requirement = clean_text(r.get("requirement"))
        clause = clean_text(r.get("suggested_addition"))
        gap = clean_text(r.get("gap_description"))
        found = clean_text(r.get("found_clause"))

        if requirement and clause:
            candidates.append({
                "requirement": requirement,
                "clause_text": clause,
                "basis": gap or found or "依據前次審查報告之章則規範義務稽核結果。",
                "source_type": "章則規範義務稽核",
                "priority": 100,
                "origin": "compliance_scan",
            })

    for m in review_context.get("missing_clauses", []) or []:
        if not isinstance(m, dict):
            continue

        requirement = clean_text(m.get("clause") or m.get("issue_topic"))
        clause = clean_text(m.get("suggested_draft"))
        basis = clean_text(m.get("why_missing") or m.get("template_snippet"))
        source = clean_text(m.get("source"))

        if requirement and clause:
            candidates.append({
                "requirement": requirement,
                "clause_text": clause,
                "basis": basis or "依據前次審查報告之歷史規範缺漏條款。",
                "source_type": f"歷史規範缺漏條款{f'｜{source}' if source else ''}",
                "priority": 80,
                "origin": "missing_clauses",
            })

    if include_issues:
        issues = (
            review_context.get("major_issues", []) or []
        ) + (
            review_context.get("general_issues", []) or []
        ) + (
            review_context.get("risk_cards", []) or []
        )

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            requirement = clean_text(issue.get("issue_topic") or issue.get("clause"))
            clause = clean_text(issue.get("adjusted_clause"))
            basis = clean_text(issue.get("analysis") or issue.get("template_basis"))
            source = clean_text(issue.get("source"))

            if requirement and clause and clause != "符合，無需修改":
                candidates.append({
                    "requirement": requirement,
                    "clause_text": clause,
                    "basis": basis or "依據前次審查報告之重大風險與建議。",
                    "source_type": f"風險條款修正建議{f'｜{source}' if source else ''}",
                    "priority": 60,
                    "origin": "issues",
                })

    if include_gap_analysis:
        gap_analysis = review_context.get("gap_analysis", {}) or {}

        for gap in gap_analysis.get("gaps", []) or []:
            if not isinstance(gap, dict):
                continue

            requirement = clean_text(gap.get("topic"))
            scenario = gap.get("vendor_refuse_scenario", {}) or {}
            clause = clean_text(scenario.get("alternative_clause"))
            basis = clean_text(
                gap.get("current_vendor_status") or gap.get("other_vendors_coverage")
            )

            if requirement and clause:
                candidates.append({
                    "requirement": requirement,
                    "clause_text": clause,
                    "basis": basis or "依據前次審查報告之跨合約差距分析。",
                    "source_type": "跨合約差距分析",
                    "priority": 40,
                    "origin": "gap_analysis",
                })

    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for candidate in candidates:
        key = (
            normalize_for_match(candidate.get("requirement")),
            normalize_for_match(candidate.get("clause_text")),
        )

        if key not in dedup or candidate.get("priority", 0) > dedup[key].get("priority", 0):
            dedup[key] = candidate

    return list(dedup.values())


def select_clause_candidates(
    user_input: str,
    review_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    include_issues = not is_all_missing_obligation_request(user_input)
    include_gap_analysis = not is_all_missing_obligation_request(user_input)

    candidates = collect_clause_candidates(
        review_context,
        include_issues=include_issues,
        include_gap_analysis=include_gap_analysis,
    )

    if not candidates:
        return []

    if is_all_clause_request(user_input):
        return sorted(
            candidates,
            key=lambda c: int(c.get("priority", 0)),
            reverse=True,
        )[:12]

    scored = []

    for candidate in candidates:
        requirement = candidate.get("requirement", "")
        score = score_requirement_match(user_input, requirement) + int(candidate.get("priority", 0))
        scored.append((score, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)

    high_confidence = [(score, candidate) for score, candidate in scored if score >= 130]

    if high_confidence:
        top_score = high_confidence[0][0]
        selected = []

        for score, candidate in high_confidence:
            if score >= top_score - 25:
                selected.append(candidate)

        if len(selected) > 1:
            user_norm = normalize_for_match(user_input)
            multi_markers = ["和", "與", "及", "以及", "還有", "全部", "所有", "分別"]

            if not any(marker in user_norm for marker in multi_markers):
                selected = [selected[0]]

        return selected[:5]

    return []


def build_clause_reminder(requirement: str) -> str:
    req_norm = normalize_for_match(requirement)

    if req_norm == normalize_for_match("資安檢測與掃描"):
        return (
            "\n\n補充提醒：\n"
            "「弱點修補期限」屬於另一個「弱點修補與維護」條款，"
            "建議另列條款處理，不要直接混入本條。"
        )

    if req_norm == normalize_for_match("弱點修補與維護"):
        return (
            "\n\n補充提醒：\n"
            "「弱點掃描與檢測報告」屬於另一個「資安檢測與掃描」條款，"
            "建議另列條款處理，不要直接混入本條。"
        )

    if req_norm == normalize_for_match("事件通報與應變"):
        return (
            "\n\n補充提醒：\n"
            "本條重點是資安事件通報與應變支援；弱點掃描或漏洞修補期限，"
            "建議另列於資安檢測或弱點修補條款。"
        )

    return ""


def sanitize_clause_text(requirement: str, clause_text: str) -> str:
    req_norm = normalize_for_match(requirement)
    text = clean_text(clause_text)

    # 弱點修補與維護條款需要比 compliance_scan 的一句話更完整，且不得沿用草稿中的高風險時限或人員描述。
    if req_norm == normalize_for_match("弱點修補與維護"):
        return (
            "乙方應就其履約範圍內之系統、程式、模組、介面及相關交付成果，配合甲方執行弱點修補與維護作業。"
            "乙方於發現或經甲方通知任何系統弱點、資安漏洞、異常狀況或維運缺失時，應即時啟動處理程序，"
            "並於甲方指定或雙方書面約定之期限內完成初步回覆、修補計畫、修補作業及處理進度回報。"
            "乙方應提供必要之技術協助、修補紀錄、測試結果、改善報告及其他甲方合理要求之佐證資料。"
            "除經甲方事前書面同意或另有明確約定外，乙方不得以另行收費、非正式人員處理、資料不足或第三方責任為由，"
            "拒絕、延遲或減輕其弱點修補與維護義務。"
            "如弱點或異常係可歸責於乙方交付成果、系統設計、程式瑕疵、設定錯誤或維護不當所致，乙方應負責無償修補，"
            "並持續追蹤至甲方確認改善完成為止。"
        )

    risky_replacements = {
        "建教合作培訓專員": "具相關經驗與資格之正式授權人員",
        "專案見習員": "具相關經驗與資格之正式授權人員",
        "見習員": "正式授權人員",
        "見習": "正式授權",
        "培訓專員": "正式授權人員",
        "兩個營業週期": "甲方指定或雙方書面約定之期限",
        "三個太陽日": "甲方指定或雙方書面約定之期限",
        "一個太陽日": "甲方指定或雙方書面約定之期限",
        "營業週期": "明確約定期限",
        "太陽日": "日曆日或雙方書面約定期限",
    }
    for old, new in risky_replacements.items():
        text = text.replace(old, new)

    return text


def format_single_clause_answer(candidate: Dict[str, Any], include_reminder: bool = True) -> str:
    requirement = clean_text(candidate.get("requirement")) or "建議補充條款"
    clause_text = sanitize_clause_text(requirement, candidate.get("clause_text"))
    basis = clean_text(candidate.get("basis"))
    source_type = clean_text(candidate.get("source_type"))

    if not clause_text:
        return "目前審查報告中沒有足夠的建議條文依據，無法直接補寫。"

    reminder = build_clause_reminder(requirement) if include_reminder else ""

    return (
        f"建議補充條款名稱：{requirement}\n\n"
        f"可直接貼入合約的條文：\n"
        f"{clause_text}\n\n"
        f"依據來源：\n"
        f"{source_type or '前次審查報告'}"
        f"{f'；{basis}' if basis else ''}"
        f"{reminder}"
    )


def format_multi_clause_answer(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "目前審查報告中沒有足夠的建議條文依據，無法直接補寫。"

    parts = ["以下依據前次審查結果整理可直接補入合約的條款："]

    for idx, candidate in enumerate(candidates, start=1):
        requirement = clean_text(candidate.get("requirement")) or f"建議補充條款 {idx}"
        clause_text = sanitize_clause_text(requirement, candidate.get("clause_text"))
        basis = clean_text(candidate.get("basis"))
        source_type = clean_text(candidate.get("source_type"))

        if not clause_text:
            continue

        parts.append(
            f"\n{idx}. {requirement}\n"
            f"條文：{clause_text}\n"
            f"依據：{source_type or '前次審查報告'}"
            f"{f'；{basis}' if basis else ''}"
        )

    return "\n".join(parts)


def build_clause_extra(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "matched_requirements": [
            clean_text(item.get("requirement"))
            for item in selected
            if clean_text(item.get("requirement"))
        ],
        "candidate_count": len(selected),
        "origins": list({
            clean_text(item.get("origin"))
            for item in selected
            if clean_text(item.get("origin"))
        }),
        "source_types": list({
            clean_text(item.get("source_type"))
            for item in selected
            if clean_text(item.get("source_type"))
        }),
    }


def answer_clause_followup_with_meta(
    user_input: str,
    review_context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    if not is_clause_followup_request(user_input, has_review_context=bool(review_context)):
        return None

    selected = select_clause_candidates(user_input, review_context)

    if not selected:
        return {
            "reply": (
                "目前審查報告中沒有找到足夠明確的對應條款建議，無法直接補寫。\n\n"
                "你可以改問更明確的義務名稱，例如：\n"
                "「資安檢測與掃描條款要怎麼補？」\n"
                "「弱點修補與維護條款要怎麼補？」\n"
                "「事件通報與應變條款要怎麼補？」"
            ),
            "tool_name": "clause_followup_no_match",
            "extra": {
                "matched_requirements": [],
                "candidate_count": 0,
                "origins": [],
                "source_types": [],
            },
        }

    if len(selected) == 1:
        reply = format_single_clause_answer(selected[0], include_reminder=True)
    else:
        reply = format_multi_clause_answer(selected)

    return {
        "reply": reply,
        "tool_name": "clause_followup",
        "extra": build_clause_extra(selected),
    }


def answer_clause_followup(
    user_input: str,
    review_context: Dict[str, Any],
) -> Optional[str]:

    result = answer_clause_followup_with_meta(user_input, review_context)

    if result is None:
        return None

    return result.get("reply", "")