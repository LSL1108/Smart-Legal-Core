"""
Rule Engine — 合約紅線檢核的資料驅動規則層。

取代 services.py 內 `build_trigger_issues_from_articles` 那個 333 行 if/elif 字串清單。

核心設計：
1. 規則是資料（YAML），不是程式碼
2. 合約類型是規則的標籤，不是 dispatch 邏輯
3. 通用層 / 領域層分檔但同 schema
4. 三層比對策略：exact (含 alias + 正規化) → regex → semantic
5. 否定句 / 語境保護避免誤觸

規則 YAML schema（單一 rule）：

    - id: foreign_jurisdiction              # 必填，全域唯一
      topic: 管轄法院                         # 必填，對應 TOPIC_KEYWORDS
      risk: High                              # Critical | High | Medium
      type: deviation                         # conflict | deviation | missing
      applicable_contract_types:              # ["*"] = 通用；列舉 = 限定類型
        - "維護合約"
        - "資安合約"

      match:                                  # 武器 1：字串 alias（必有 any_of 或 all_of）
        any_of: ["美國加州", "California", ...]
        # all_of: [...]                       # 須全部出現才命中

      regex_match:                            # 武器 2：可選，正規表達式
        pattern: "違約金.*?上限.*?(?P<amount>\\d+)\\s*萬"
        threshold:                            # 可選：對 named capture 做數值比對
          capture: amount
          op: "<"
          value: 100

      semantic_match:                         # 武器 3：可選，語意相似
        examples:
          - "本合約以美國加州法院為管轄"
          - "適用合約簽訂地之當地法律"
        threshold: 0.62

      context_required:                       # 命中後還需驗證的語境（避免無關語境誤觸）
        any_of: ["維護", "故障", "技術支援"]

      exclude:                                # 否定句保護（命中字眼前後 N 字內出現排除詞→跳過）
        near_terms: ["不得", "禁止", "排除"]
        window: 8

      analysis_template: |                    # 文案，可用 {matched} 變數
        本條出現「{matched}」字眼，...
      suggestion: 建議...
      adjusted_clause: 應...
"""

from __future__ import annotations

import glob
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from text_normalize import (
    normalize_for_match,
    expand_aliases_for_match,
    find_any_occurrence,
)

logger = logging.getLogger(__name__)


# ---- Optional dependency: PyYAML ----
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception as exc:
    logger.error("PyYAML 未安裝，rule engine 將無法載入規則: %s", exc)
    _YAML_AVAILABLE = False


# ---- Rule schema ----

@dataclass
class TriggerRule:
    id: str
    topic: str
    risk: str                                  # Critical / High / Medium
    type: str                                  # conflict / deviation / missing
    applicable_contract_types: List[str]       # ["*"] = 全部適用
    match: Dict[str, Any]                      # any_of / all_of
    regex_match: Optional[Dict[str, Any]]
    semantic_match: Optional[Dict[str, Any]]
    context_required: Optional[Dict[str, Any]]
    exclude: Optional[Dict[str, Any]]
    analysis_template: str
    suggestion: str
    adjusted_clause: str

    # 載入時預先計算的 alias 比對表 [(原字串, 正規化字串)]
    _match_alias_pairs: List[Tuple[str, str]] = field(default_factory=list)
    _all_of_pairs: List[Tuple[str, str]] = field(default_factory=list)
    _exclude_norm: List[str] = field(default_factory=list)
    _context_norm: List[str] = field(default_factory=list)
    _compiled_regex: Optional[re.Pattern] = None


# ---- Loader ----

_rules_cache: Optional[List[TriggerRule]] = None
_cache_lock = threading.Lock()
_rule_id_seen: Set[str] = set()


def _compile_rule(raw: Dict[str, Any]) -> Optional[TriggerRule]:
    """單一規則 dict → TriggerRule，預先做 alias 展開、regex 編譯。"""
    try:
        match_block = raw.get("match", {}) or {}
        regex_block = raw.get("regex_match")
        semantic_block = raw.get("semantic_match")
        context_block = raw.get("context_required")
        exclude_block = raw.get("exclude")

        rule = TriggerRule(
            id=raw["id"],
            topic=raw["topic"],
            risk=raw.get("risk", "High"),
            type=raw.get("type", "deviation"),
            applicable_contract_types=raw.get("applicable_contract_types", ["*"]) or ["*"],
            match=match_block,
            regex_match=regex_block,
            semantic_match=semantic_block,
            context_required=context_block,
            exclude=exclude_block,
            analysis_template=raw.get("analysis_template", "本條疑似命中「{matched}」紅線。"),
            suggestion=raw.get("suggestion", "建議重新審視本條規範。"),
            adjusted_clause=raw.get("adjusted_clause", ""),
        )

        # 預先展開 any_of alias（含正規化、簡繁變體）
        any_of = match_block.get("any_of") or []
        if any_of:
            rule._match_alias_pairs = expand_aliases_for_match(any_of)

        # 預先展開 all_of alias
        all_of = match_block.get("all_of") or []
        if all_of:
            rule._all_of_pairs = expand_aliases_for_match(all_of)

        # 預先正規化 exclude / context terms
        if exclude_block:
            rule._exclude_norm = [
                normalize_for_match(t) for t in (exclude_block.get("near_terms") or [])
                if t
            ]
        if context_block:
            rule._context_norm = [
                normalize_for_match(t) for t in (context_block.get("any_of") or [])
                if t
            ]

        # 預編譯 regex
        if regex_block and regex_block.get("pattern"):
            try:
                rule._compiled_regex = re.compile(regex_block["pattern"])
            except re.error as e:
                logger.error("Rule %s 的 regex 無法編譯：%s", rule.id, e)
                rule._compiled_regex = None

        return rule
    except KeyError as e:
        logger.error("規則缺少必填欄位 %s：%s", e, raw)
        return None
    except Exception as e:
        logger.error("規則編譯失敗：%s — raw=%s", e, raw)
        return None


def _load_yaml_file(path: str) -> List[Dict[str, Any]]:
    """單一 YAML 檔讀進 dict 清單。容忍空檔、錯誤檔。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "rules" in data:
            return data["rules"] or []
        logger.warning("規則檔 %s 格式不正確（須為 list 或 {rules: [...]}）", path)
        return []
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.error("規則檔 %s 載入失敗：%s", path, e)
        return []


def load_rules(rules_dir: Optional[str] = None, force_reload: bool = False) -> List[TriggerRule]:
    """
    從 rules/*.yaml 載入所有規則，合併後回傳。

    結果快取在記憶體。force_reload=True 可繞過快取（測試 / 開發用）。
    """
    global _rules_cache, _rule_id_seen

    if not force_reload and _rules_cache is not None:
        return _rules_cache

    if not _YAML_AVAILABLE:
        logger.error("PyYAML 不可用，rule engine 載入空規則")
        with _cache_lock:
            _rules_cache = []
        return []

    if rules_dir is None:
        from config import BASE_DIR
        rules_dir = os.path.join(BASE_DIR, "rules")

    with _cache_lock:
        if not force_reload and _rules_cache is not None:
            return _rules_cache

        rules: List[TriggerRule] = []
        seen_ids: Set[str] = set()

        if not os.path.isdir(rules_dir):
            logger.warning("規則目錄不存在：%s", rules_dir)
            _rules_cache = []
            return []

        for path in sorted(glob.glob(os.path.join(rules_dir, "*.yaml")) +
                           glob.glob(os.path.join(rules_dir, "*.yml"))):
            raw_list = _load_yaml_file(path)
            for raw in raw_list:
                rule = _compile_rule(raw)
                if rule is None:
                    continue
                if rule.id in seen_ids:
                    logger.warning("重複的 rule id「%s」（位於 %s）— 已略過後者", rule.id, path)
                    continue
                seen_ids.add(rule.id)
                rules.append(rule)

        _rules_cache = rules
        _rule_id_seen = seen_ids

        logger.info("Rule engine 載入完成：%d 條規則", len(rules))
        return rules


def reload_rules() -> List[TriggerRule]:
    """強制重新載入（測試 / 開發用）。"""
    return load_rules(force_reload=True)


# ---- 適用性判定 ----

def rule_applies_to_contract_type(rule: TriggerRule, contract_type: str) -> bool:
    """
    判斷規則是否適用於此合約類型。
    - applicable_contract_types 含 "*" → 全部適用
    - 否則需嚴格列舉相符
    """
    types = rule.applicable_contract_types
    if not types:
        return True
    if "*" in types:
        return True
    return contract_type in types


# ---- 三層 evaluator ----

def _check_exclude(content_norm: str, matched_pairs: List[Tuple[str, str]],
                   exclude_norm: List[str], window: int) -> bool:
    """
    否定句保護：對每個命中字眼，檢查其前 window 字內是否出現排除詞。
    回傳 True 表示「應該排除」（即該命中視為無效）。
    任一命中字眼前後都沒有排除詞，才算真正命中。
    """
    if not exclude_norm:
        return False

    # 對每個命中字眼，找出其在 content_norm 的位置，看前 window 字是否有排除詞
    for raw, norm in matched_pairs:
        pos = find_any_occurrence(content_norm, norm)
        if pos is None:
            continue
        start = pos[0]
        # 前 window 字的範圍
        prefix_start = max(0, start - window)
        prefix_text = content_norm[prefix_start:start]
        if any(neg in prefix_text for neg in exclude_norm):
            return True
    return False


def _check_context(content_norm: str, context_norm: List[str]) -> bool:
    """語境檢查：條文需含至少一個語境詞。沒設定 = 永遠通過。"""
    if not context_norm:
        return True
    return any(t in content_norm for t in context_norm)


def evaluate_exact(rule: TriggerRule, content: str, content_norm: str) -> Optional[List[str]]:
    """
    武器 1：字串 alias 比對（含正規化、否定句保護、語境檢查）。
    回傳命中的「原字串」清單；沒命中回傳 None。
    """
    matched_pairs: List[Tuple[str, str]] = []

    # any_of 模式
    if rule._match_alias_pairs:
        for raw, norm in rule._match_alias_pairs:
            if norm and norm in content_norm:
                matched_pairs.append((raw, norm))

    # all_of 模式（必須全部出現）
    if rule._all_of_pairs:
        all_hit = all(norm in content_norm for _, norm in rule._all_of_pairs)
        if all_hit:
            matched_pairs.extend(rule._all_of_pairs)

    if not matched_pairs:
        return None

    # 語境檢查
    if not _check_context(content_norm, rule._context_norm):
        return None

    # 否定句保護
    window = (rule.exclude or {}).get("window", 8) if rule.exclude else 8
    if _check_exclude(content_norm, matched_pairs, rule._exclude_norm, window):
        return None

    # 回傳唯一原字串清單
    seen: Set[str] = set()
    out: List[str] = []
    for raw, _ in matched_pairs:
        if raw and raw not in seen:
            seen.add(raw)
            out.append(raw)
    return out or None


def evaluate_regex(rule: TriggerRule, content: str, content_norm: str) -> Optional[List[str]]:
    """
    武器 2：regex 比對 + 可選的數值門檻檢查。
    回傳命中的「字串片段」清單；沒命中回傳 None。
    """
    if rule._compiled_regex is None:
        return None

    matches = list(rule._compiled_regex.finditer(content))
    if not matches:
        # 用正規化版再試一次（regex 不一定能對全形 / 簡繁）
        matches = list(rule._compiled_regex.finditer(content_norm))
    if not matches:
        return None

    # 語境檢查
    if not _check_context(content_norm, rule._context_norm):
        return None

    # 否定句保護（regex 也適用 — 命中片段前 window 字檢查）
    window = (rule.exclude or {}).get("window", 8) if rule.exclude else 8
    if rule._exclude_norm:
        # 把 match 結果轉成 (raw, norm) pairs，重用 _check_exclude
        pairs = [(m.group(0), normalize_for_match(m.group(0))) for m in matches]
        if _check_exclude(content_norm, pairs, rule._exclude_norm, window):
            return None

    # 數值門檻：對 named capture 做比較
    threshold = (rule.regex_match or {}).get("threshold")
    if threshold:
        cap_name = threshold.get("capture")
        op = threshold.get("op", "<")
        value = threshold.get("value", 0)

        passing = []
        for m in matches:
            try:
                # 先試指定的 capture name；若該 group 為 None（regex 有多個 alternative
                # named groups 時常見），再從 m.groupdict() 找第一個非 None 的數值
                cap_value = None
                if cap_name:
                    cap_value = m.groupdict().get(cap_name)
                if cap_value is None:
                    # fallback：取第一個有值的 group
                    for k, v in m.groupdict().items():
                        if v is not None:
                            cap_value = v
                            break
                if cap_value is None:
                    continue
                # 把可能的千分位、單位殘留拿掉
                num_str = re.sub(r"[^\d.\-]", "", str(cap_value))
                num = float(num_str) if num_str else None
                if num is None:
                    continue
                if op == "<" and num < value:
                    passing.append(m.group(0))
                elif op == "<=" and num <= value:
                    passing.append(m.group(0))
                elif op == ">" and num > value:
                    passing.append(m.group(0))
                elif op == ">=" and num >= value:
                    passing.append(m.group(0))
                elif op == "==" and num == value:
                    passing.append(m.group(0))
            except (ValueError, TypeError):
                continue
        if not passing:
            return None
        return list(dict.fromkeys(passing))

    # 沒有 threshold → 任何命中都算
    return list(dict.fromkeys(m.group(0) for m in matches))


# ---- 武器 3：semantic match ----
# 用既有的 BGE-M3 透過 Chroma 的 embedding function 算向量；不另裝模型。

_semantic_centroids: Dict[str, Dict[str, Any]] = {}
_semantic_init_lock = threading.Lock()
_semantic_init_done = False


def _ensure_semantic_centroids():
    """為所有有 semantic_match 的規則預先計算 centroid embedding。一次性。"""
    global _semantic_init_done
    if _semantic_init_done:
        return

    from config import ENABLE_SEMANTIC_TRIGGERS
    if not ENABLE_SEMANTIC_TRIGGERS:
        _semantic_init_done = True
        return

    with _semantic_init_lock:
        if _semantic_init_done:
            return

        rules = load_rules()
        rules_with_semantic = [r for r in rules if r.semantic_match and r.semantic_match.get("examples")]
        if not rules_with_semantic:
            _semantic_init_done = True
            return

        try:
            embed_fn = _get_embedding_function()
        except Exception as exc:
            logger.warning("semantic trigger 初始化失敗（embedding function 不可用）：%s", exc)
            _semantic_init_done = True
            return

        for rule in rules_with_semantic:
            try:
                examples = rule.semantic_match["examples"]
                vectors = embed_fn(examples)
                if vectors is None:
                    continue
                # 統一轉 list，避免後續 cosine 收到 ndarray 觸發真值歧義
                try:
                    import numpy as np
                    arr = np.asarray(vectors, dtype=float)
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    centroid = arr.mean(axis=0).tolist()
                except Exception:
                    centroid = _mean_vector(vectors)
                if not centroid:
                    continue
                _semantic_centroids[rule.id] = {
                    "centroid": centroid,
                    "threshold": float(rule.semantic_match.get("threshold", 0.6)),
                }
            except Exception as exc:
                logger.warning("計算規則 %s 的 semantic centroid 失敗：%s", rule.id, exc)

        logger.info("Semantic trigger 已初始化 %d 條規則的 centroid", len(_semantic_centroids))
        _semantic_init_done = True


def _get_embedding_function():
    """取得 Chroma 用的同一個 BGE-M3 embedding function（避免重複載入）。"""
    from chromadb.utils import embedding_functions
    from config import OLLAMA_EMBED_URL, EMBED_MODEL
    return embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_EMBED_URL,
        model_name=EMBED_MODEL,
    )


def _mean_vector(vectors) -> List[float]:
    """
    平均多個向量。接受 list[list[float]] 或 list[ndarray] 或 ndarray。
    回傳 list[float]（純 Python 型別，方便後續序列化 / cache）。
    """
    # 統一轉成 list of list
    if vectors is None:
        return []
    try:
        # 試著轉成 numpy 計算，省力且通用
        import numpy as np  # 局部 import，避免上游沒裝 numpy 時整個檔載入失敗
        arr = np.asarray(vectors, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return []
        return arr.mean(axis=0).tolist()
    except Exception:
        # 退路：純 Python 算
        if len(vectors) == 0:
            return []
        try:
            dim = len(vectors[0])
        except Exception:
            return []
        out = [0.0] * dim
        n = 0
        for v in vectors:
            if v is None:
                continue
            for i in range(min(dim, len(v))):
                out[i] += float(v[i])
            n += 1
        if n == 0:
            return []
        return [x / n for x in out]


def _cosine(a, b) -> float:
    """
    兩向量 cosine similarity。接受 list[float] 或 numpy array。

    重要：用 `len(...) == 0` 與顯式 `is None` 判斷，
    不能用 `if not a or not b` — 對 numpy array 會丟
    「The truth value of an array with more than one element is ambiguous」。
    """
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
        a_arr = np.asarray(a, dtype=float).ravel()
        b_arr = np.asarray(b, dtype=float).ravel()
        if a_arr.size == 0 or b_arr.size == 0 or a_arr.size != b_arr.size:
            return 0.0
        na = float(np.linalg.norm(a_arr))
        nb = float(np.linalg.norm(b_arr))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (na * nb))
    except Exception:
        # 純 Python 退路
        try:
            la, lb = len(a), len(b)
        except Exception:
            return 0.0
        if la == 0 or lb == 0 or la != lb:
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = sum(float(x) * float(x) for x in a) ** 0.5
        nb = sum(float(y) * float(y) for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


# 條文 embedding 快取（key = sha256，避免同條文重審重算）
_content_embedding_cache: Dict[str, List[float]] = {}
_content_cache_lock = threading.Lock()
_CONTENT_CACHE_MAX = 512  # LRU 上限，避免無上界成長


def _get_content_embedding(content_norm: str) -> Optional[List[float]]:
    import hashlib
    if not content_norm:
        return None
    key = hashlib.sha256(content_norm.encode("utf-8")).hexdigest()[:24]

    with _content_cache_lock:
        if key in _content_embedding_cache:
            return _content_embedding_cache[key]

    try:
        embed_fn = _get_embedding_function()
        vectors = embed_fn([content_norm[:2000]])  # 條文太長截斷
        if vectors is None:
            return None
        # Chroma 的 embedding function 可能回傳 list 或 ndarray，
        # 統一轉成純 list 存 cache（避免下游算 cosine 時還要再判 numpy 型別）
        try:
            import numpy as np
            arr = np.asarray(vectors)
            if arr.size == 0:
                return None
            # 第一個向量
            emb = arr[0].tolist() if arr.ndim >= 2 else arr.tolist()
        except Exception:
            if not vectors:
                return None
            v0 = vectors[0]
            emb = list(v0) if hasattr(v0, "__iter__") else None
            if emb is None:
                return None
    except Exception as exc:
        logger.debug("條文 embedding 失敗（semantic trigger 跳過）：%s", exc)
        return None

    with _content_cache_lock:
        if len(_content_embedding_cache) >= _CONTENT_CACHE_MAX:
            # 簡單 FIFO：刪最早一筆
            try:
                _content_embedding_cache.pop(next(iter(_content_embedding_cache)))
            except StopIteration:
                pass
        _content_embedding_cache[key] = emb
    return emb


def evaluate_semantic(rule: TriggerRule, content: str, content_norm: str) -> Optional[List[str]]:
    """
    武器 3：用代表句的 centroid 與條文做 cosine 相似度。
    超過 threshold → 視為命中。回傳特殊標記 ["（語意命中）"]。
    """
    if not rule.semantic_match:
        return None

    from config import ENABLE_SEMANTIC_TRIGGERS, SEMANTIC_MIN_CONTENT_CHARS
    if not ENABLE_SEMANTIC_TRIGGERS:
        return None

    # 條文太短不適合走 semantic（embedding 不穩）
    if len(content) < SEMANTIC_MIN_CONTENT_CHARS:
        return None

    _ensure_semantic_centroids()

    entry = _semantic_centroids.get(rule.id)
    if not entry:
        return None

    content_emb = _get_content_embedding(content_norm)
    if content_emb is None:
        return None

    score = _cosine(content_emb, entry["centroid"])
    if score < entry["threshold"]:
        return None

    # 語境 / 排除句保護仍然套用
    if not _check_context(content_norm, rule._context_norm):
        return None
    # semantic 命中沒有具體 anchor 字眼，無法做 near-window exclude；
    # 改成檢查整條文是否含排除詞
    if rule._exclude_norm:
        if any(neg in content_norm for neg in rule._exclude_norm):
            # 整條文有排除詞 → 偏保守，跳過
            logger.debug("Rule %s semantic 命中但條文含排除詞，已跳過", rule.id)
            return None

    return [f"（語意相似 {score:.2f}）"]


# ---- 主入口 ----

@dataclass
class RuleHit:
    rule: TriggerRule
    matched: List[str]
    match_type: str  # "exact" / "regex" / "semantic"


def evaluate_rule(
    rule: TriggerRule,
    content: str,
    content_norm: Optional[str] = None,
) -> Optional[RuleHit]:
    """
    對單一規則跑全部 evaluator，按優先序：exact → regex → semantic。
    第一個命中就回傳，不重複跑。
    """
    if content_norm is None:
        content_norm = normalize_for_match(content)

    matched = evaluate_exact(rule, content, content_norm)
    if matched:
        return RuleHit(rule=rule, matched=matched, match_type="exact")

    matched = evaluate_regex(rule, content, content_norm)
    if matched:
        return RuleHit(rule=rule, matched=matched, match_type="regex")

    matched = evaluate_semantic(rule, content, content_norm)
    if matched:
        return RuleHit(rule=rule, matched=matched, match_type="semantic")

    return None


def evaluate_all_rules(
    content: str,
    contract_type: str,
    rules: Optional[List[TriggerRule]] = None,
) -> List[RuleHit]:
    """
    對單一條文跑所有適用規則，回傳全部命中。
    呼叫端負責後續去重（相同 topic 取最高 risk）。
    """
    if rules is None:
        rules = load_rules()
    if not content or not content.strip():
        return []

    content_norm = normalize_for_match(content)
    out: List[RuleHit] = []
    for rule in rules:
        if not rule_applies_to_contract_type(rule, contract_type):
            continue
        hit = evaluate_rule(rule, content, content_norm)
        if hit:
            out.append(hit)
    return out


# ---- 文案組裝 ----

def render_issue_from_hit(hit: RuleHit, article: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    把 RuleHit 轉成 services.py 預期的 issue dict。
    保持 schema 跟舊版一致，盡量不影響下游。
    """
    matched_text = "、".join(hit.matched) if hit.matched else "（未提供）"

    article_no = str(article.get("article_no", "") or "").strip()
    article_title = str(article.get("title", "") or "").strip()
    if article_no and article_title:
        clause_name = f"{article_no}：{article_title}"
    elif article_no:
        clause_name = article_no
    elif article_title:
        clause_name = article_title
    else:
        clause_name = f"第 {idx} 條"

    # 延遲 import 避免循環依賴
    from utils import normalize_topic_name, article_to_key

    analysis = hit.rule.analysis_template.format(matched=matched_text)

    return {
        "article_key": article_to_key(article, idx),
        "clause": clause_name,
        "issue_topic": normalize_topic_name(hit.rule.topic),
        "type": hit.rule.type,
        "risk": hit.rule.risk,
        "draft_text": str(article.get("content", "") or "").strip(),
        "template_basis": f"系統紅線規則：{hit.rule.id}",
        "template_snippet": f"命中（{hit.match_type}）：{matched_text}",
        "analysis": analysis,
        "suggestion": hit.rule.suggestion,
        "adjusted_clause": hit.rule.adjusted_clause,
        "negotiation_notes": "最低底線：不得降低甲方法遵、保戶權益、個資保護、保密、爭議處理或違約救濟保障。",
        "source": "系統內建企業紅線規則",
        "_rule_id": hit.rule.id,
        "_match_type": hit.match_type,
    }


def merge_issues_by_topic(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同一條文同 topic 多個 rule 命中時合併：取最高 risk，matched 字眼合併。
    跨條文則保留（不同條文的同 topic 是獨立 issue）。
    """
    if not issues:
        return []

    # group by (article_key, topic)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for issue in issues:
        key = (str(issue.get("article_key", "")), str(issue.get("issue_topic", "")))
        grouped.setdefault(key, []).append(issue)

    risk_rank = {"Critical": 3, "High": 2, "Medium": 1}
    out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        # 取最高 risk 那筆作主，其他規則的 matched 併入 template_snippet
        main = max(group, key=lambda x: risk_rank.get(x.get("risk", "Medium"), 0))
        all_snippets = []
        all_rule_ids = []
        for g in group:
            snip = str(g.get("template_snippet", "") or "")
            if snip:
                all_snippets.append(snip)
            rid = g.get("_rule_id")
            if rid:
                all_rule_ids.append(rid)
        if all_snippets:
            main = dict(main)
            main["template_snippet"] = " ｜ ".join(dict.fromkeys(all_snippets))
        if all_rule_ids:
            main["_rule_ids"] = list(dict.fromkeys(all_rule_ids))
        out.append(main)

    return out


# ---- 規則健康度檢查 ----

def lint_rules(rules: Optional[List[TriggerRule]] = None) -> List[str]:
    """
    回傳規則檢查結果（warnings 清單）。CI / 啟動時可呼叫。
    """
    if rules is None:
        rules = load_rules()
    warnings: List[str] = []

    seen_ids: Set[str] = set()
    for r in rules:
        if r.id in seen_ids:
            warnings.append(f"重複 rule id: {r.id}")
        seen_ids.add(r.id)
        if r.risk not in ("Critical", "High", "Medium"):
            warnings.append(f"{r.id}: risk='{r.risk}' 不是預期值")
        if r.type not in ("conflict", "deviation", "missing"):
            warnings.append(f"{r.id}: type='{r.type}' 不是預期值")
        if not r._match_alias_pairs and not r._all_of_pairs and not r._compiled_regex and not r.semantic_match:
            warnings.append(f"{r.id}: 沒有任何 match 條件（exact/regex/semantic 全空）")
        if not r.applicable_contract_types:
            warnings.append(f"{r.id}: applicable_contract_types 為空（建議至少寫 ['*']）")
    return warnings
