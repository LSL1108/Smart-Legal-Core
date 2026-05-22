import re
from typing import Literal, List


IntentType = Literal[
    "contract_review",
    "contract_generate",
    "price_risk",
    "law_check",
    "security_explanation",
    "historical_compare",
    "chat",
]


_GENERATE_KEYWORDS = [
    "生成合約", "產生合約", "草擬合約", "寫一份合約",
    "幫我寫合約", "套版", "自動生成", "產出合約",
    "製作合約", "起草合約",
]


_GENERATE_PATTERNS = [
    r"(生成|產生|草擬|起草|製作|產出|寫).{0,8}合約",
]

_REVIEW_KEYWORDS = [
    "審查", "檢查合約", "合約風險", "風險條款", "缺漏",
    "不合規", "高風險", "中風險", "低風險", "幫我看這份合約",
    "有問題嗎", "看一下合約", "合約有沒有問題", "幫我看合約",
    "看看這份", "這份合約有",
]

_HISTORICAL_KEYWORDS = [
    "歷史合約", "舊合約", "舊約", "過往合約",
    "前一版合約", "上一版合約", "跟以前比", "差在哪", "差異",
    "新舊比較", "版本比較", "比舊版", "和以前",
]

_LAW_KEYWORDS = [
    "法條", "法規", "章則", "金管會", "保險業",
    "個資法", "資通安全法", "委外", "內控", "是否合規",
    "合規要求", "法遵", "監理", "主管機關", "資安法",
    "金融監督", "個人資料保護法",
]


_PRICE_STRONG_KEYWORDS = [
    "報價", "太貴", "偏高", "偏低",
    "歷史價格", "採購金額", "這個價錢", "報價合理嗎",
    "報價風險",
]


_PRICE_WEAK_KEYWORDS = ["價格", "金額", "費用", "合理嗎"]

_CONTRACT_CONTEXT_WORDS = [
    "條款", "合約", "契約", "草稿", "這份", "此份", "本合約",
    "審查", "缺漏", "約定", "補入", "加入", "那份",
    "原文", "內容", "甲方", "乙方", "丙方", "廠商",
]

_DOCUMENT_POINTER_WORDS = [
    "這份", "此份", "本合約", "這個合約", "這份合約",
    "草稿", "廠商草稿", "上傳", "剛剛那份", "剛才那份",
    "第", "條", "條款", "原文", "內容",
    "甲方", "乙方", "丙方", "這條", "該條", "這一條",
]

_GENERAL_EXPLANATION_PATTERNS = [
    r"(什麼是|是什麼|介紹|解釋|說明|定義).{0,20}(合約|契約|法遵|合規|資安|資訊安全|委外|個資|弱點掃描|弱掃|滲透測試)",
    r"(合約審查|法遵審查|合規檢查|資安檢測|弱點掃描|滲透測試).{0,12}(怎麼做|如何做|流程|架構|設計|原理|意思)",
    r"(請問|想問|我想知道).{0,20}(是什麼|什麼意思|怎麼定義|如何理解)",
]


_SECURITY_EXPLAIN_PATTERNS = [
    r"(什麼是|是什麼|介紹|解釋|說明).{0,10}(資安|資訊安全|資通安全)",
    r"(資安|弱掃|弱點掃描|滲透測試|iso\s*27001|soc\s*2).{0,8}(是什麼|什麼意思|怎麼定義|如何定義|指的是)",
    r"(什麼是|是什麼).{0,15}(弱掃|弱點掃描|滲透測試|資安檢測|資安掃描)",
    r"(幫我|請|可以).{0,5}(解釋|說明|介紹).{0,10}(資安|弱掃|弱點掃描|iso27001|滲透測試|資安掃描|資安檢測)",
    r"(說明|解釋|介紹).{0,10}(弱掃|弱點掃描|滲透測試|資安檢測|資安掃描|iso27001)",
]


# 工具函數

def _normalize(text: str) -> str:
    """小寫 + 移除空白，讓比對不受輸入格式影響。"""
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def _has_contract_context(text: str) -> bool:
    return _has_any(text, _CONTRACT_CONTEXT_WORDS)


def _has_document_pointer(text: str) -> bool:
    return _has_any(text, _DOCUMENT_POINTER_WORDS)


def _is_general_explanation(text: str) -> bool:
    return any(re.search(p, text) for p in _GENERAL_EXPLANATION_PATTERNS)


def _is_generate_intent(text: str) -> bool:
    return (
        _has_any(text, _GENERATE_KEYWORDS)
        or any(re.search(p, text) for p in _GENERATE_PATTERNS)
    )


def _is_security_explanation(text: str) -> bool:
    return any(re.search(p, text) for p in _SECURITY_EXPLAIN_PATTERNS)


# 主函數
def detect_intent(user_input: str) -> IntentType:
    text = _normalize(user_input)

    if not text:
        return "chat"

    if _is_security_explanation(text):
        return "security_explanation"

    if _is_general_explanation(text):
        return "chat"

    # ① 合約生成
    if _is_generate_intent(text):
        return "contract_generate"

    # ② 歷史比對
    if _has_any(text, _HISTORICAL_KEYWORDS):
        return "historical_compare"

    # ③ 合約審查：必須有審查詞，且要有文件／條款指向或明確合約語境。
    if _has_any(text, _REVIEW_KEYWORDS):
        if _has_document_pointer(text) or _has_contract_context(text):
            return "contract_review"

    # ④ 報價風險：強訊號直接命中
    if _has_any(text, _PRICE_STRONG_KEYWORDS):
        return "price_risk"

    # ④b 弱訊號：有合約脈絡 → contract_review；無 → price_risk
    if _has_any(text, _PRICE_WEAK_KEYWORDS):
        return "contract_review" if (_has_document_pointer(text) or _has_contract_context(text)) else "price_risk"

    # ⑤ 法規 / 法遵查詢：有文件指向或合約語境才進 law_check；純概念問題留給一般聊天。
    if _has_any(text, _LAW_KEYWORDS):
        if _has_document_pointer(text) or _has_contract_context(text):
            return "law_check"
        return "chat"

    return "chat"


def detect_intents(user_input: str) -> List[IntentType]:

    text = _normalize(user_input)

    if not text:
        return ["chat"]


    if _is_security_explanation(text):
        return ["security_explanation"]

    if _is_general_explanation(text):
        return ["chat"]

    results: List[IntentType] = []

    if _is_generate_intent(text):
        results.append("contract_generate")

    if _has_any(text, _HISTORICAL_KEYWORDS):
        results.append("historical_compare")

    has_doc_context = _has_document_pointer(text) or _has_contract_context(text)

    if _has_any(text, _REVIEW_KEYWORDS) and has_doc_context:
        results.append("contract_review")
    elif _has_any(text, _PRICE_WEAK_KEYWORDS) and has_doc_context:

        results.append("contract_review")

    if _has_any(text, _PRICE_STRONG_KEYWORDS) or (
        _has_any(text, _PRICE_WEAK_KEYWORDS) and not has_doc_context
    ):
        results.append("price_risk")

    if _has_any(text, _LAW_KEYWORDS) and has_doc_context:
        results.append("law_check")

    return results if results else ["chat"]