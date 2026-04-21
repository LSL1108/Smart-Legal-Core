import streamlit as st
import os
import datetime
import logging
import time
import requests

from database import (
    count_history_records, insert_history_records, get_all_templates,
    get_compliance_rules, upsert_compliance_rule, delete_compliance_rule,  
)
from utils import extract_text_from_pdf, extract_text_from_docx

API_BASE_URL = "http://127.0.0.1:8000/api"


@st.cache_data(ttl=30)
def cached_get_all_templates():
    return get_all_templates()


@st.cache_data(ttl=10)  
def cached_get_compliance_rules():
    return get_compliance_rules()


# UI 渲染輔助函式
def risk_label(risk: str) -> str:
    r = (risk or "").lower()
    if r == "critical":
        return "🔴 嚴重合規風險"
    if r == "high":
        return "🟠 高度合規風險"
    if r == "medium":
        return "🟡 條件偏離"
    return "🟢 符合基準"


def render_issue_block(item: dict):
    issue_type = item.get("type", "")
    issue_type_label = ""
    if issue_type == "conflict":
        issue_type_label = "（條文衝突）"
    elif issue_type == "deviation":
        issue_type_label = "（標準偏離）"

    st.markdown(f"#### {risk_label(item.get('risk'))}｜{item.get('clause', '未命名條款')} {issue_type_label}")
    with st.container(border=True):
        if item.get("draft_text"):
            with st.expander("點擊展開：廠商草稿內容"):
                st.write(item["draft_text"])

        if item.get("template_basis"):
            with st.expander(" 點擊展開：企業法遵與歷史基準依據"):
                st.write(item["template_basis"])

        if item.get("analysis"):
            st.markdown(" **合規落差分析**")
            st.write(item["analysis"])

        if item.get("suggestion"):
            st.markdown(" **建議修正與協商方案**")
            st.write(item["suggestion"])

        if item.get("adjusted_clause") and item["adjusted_clause"] != "符合，無需修改":
            st.markdown(" **建議修改後條文**")
            st.info(item["adjusted_clause"])

        if item.get("negotiation_notes"):
            st.markdown(" **協商備忘（最低底線）**")
            st.caption(item["negotiation_notes"])

        if item.get("source"):
            st.caption(f" 參考來源：{item['source']}")


def render_missing_block(item: dict):
    with st.container(border=True):
        st.markdown(f"** {item.get('clause', '未命名條款')}**")
        if item.get("why_missing"):
            st.markdown("**缺漏原因**")
            st.write(item["why_missing"])
        if item.get("suggestion"):
            st.markdown("**建議補入與應對方向**")
            st.write(item["suggestion"])

        if item.get("suggested_draft"):
            st.markdown(" **建議補充條文草稿**")
            st.info(item["suggested_draft"])

        if item.get("template_snippet"):
            with st.expander(" 點擊展開：企業智庫原文參考"):
                st.info(item["template_snippet"])

        if item.get("source"):
            st.caption(f" 參考來源：{item['source']}")


def render_compliance_scan(compliance_scan: list):
    if not compliance_scan:
        return

    st.markdown("---")
    st.markdown("###  章則規範義務稽核")

    covered = [r for r in compliance_scan if r.get("is_covered")]
    uncovered = [r for r in compliance_scan if not r.get("is_covered")]

    col1, col2 = st.columns(2)
    col1.metric(" 已涵蓋義務", f"{len(covered)} 項")
    col2.metric(" 缺漏義務", f"{len(uncovered)} 項")

    if uncovered:
        st.markdown("####  尚未涵蓋的廠商義務")
        for r in uncovered:
            with st.container(border=True):
                st.markdown(f"**{r.get('requirement', '未知義務')}**")
                if r.get("gap_description"):
                    st.warning(r["gap_description"])
                if r.get("suggested_addition"):
                    st.markdown(" **建議補入條文**")
                    st.info(r["suggested_addition"])

    if covered:
        with st.expander(f"✅ 點擊展開：已涵蓋的 {len(covered)} 項義務"):
            for r in covered:
                st.markdown(f"- **{r.get('requirement')}**")
                if r.get("found_clause"):
                    st.caption(f"  合約對應文字：{r['found_clause'][:100]}...")


def render_gap_analysis(gap_analysis: dict):
    if not gap_analysis or not gap_analysis.get("gaps"):
        return

    st.markdown("---")
    st.markdown("###  跨合約差距分析")

    if gap_analysis.get("gap_summary"):
        st.info(f"**差距摘要**：{gap_analysis['gap_summary']}")

    for g in gap_analysis.get("gaps", []):
        with st.container(border=True):
            st.markdown(f"**差距主題：{g.get('topic', '未知')}**")

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(" **歷史合約作法**")
                st.write(g.get("other_vendors_coverage") or "無資料")
            with col_r:
                st.markdown(" **本次廠商現況**")
                st.write(g.get("current_vendor_status") or "未知")

            scenario = g.get("vendor_refuse_scenario", {})
            if scenario:
                with st.expander("⚠️ 若廠商拒絕配合：風險與應對方案"):
                    if scenario.get("risk_description"):
                        st.error(f"**具體風險**：{scenario['risk_description']}")
                    if scenario.get("cost_bearing_suggestion"):
                        st.markdown(f"💰 **費用分擔建議**：{scenario['cost_bearing_suggestion']}")
                    if scenario.get("alternative_clause"):
                        st.markdown("📝 **替代條文草稿**")
                        st.info(scenario["alternative_clause"])


def render_review_dashboard(data: dict):
    col1, col2 = st.columns([1, 3])
    with col1:
        major = data.get("major_issues", [])
        missing = data.get("missing_clauses", [])
        general = data.get("general_issues", [])

        if major or missing:
            status_text = "🔴 高風險"
        elif general:
            status_text = "🟡 部分瑕疵"
        else:
            status_text = "🟢 符合基準"

        st.metric("🛡️ 企業合規狀態", status_text)

    with col2:
        st.info(
            f"** 合約類型判定**：{data.get('contract_type_guess', '未判定')}\n\n"
            f"** 系統總結摘要**：{data.get('summary', '無摘要')}"
        )

    used_templates = data.get("used_templates", [])
    if used_templates:
        st.markdown("###  本次比對歷史基準與法遵智庫")
        for t in used_templates:
            with st.container(border=True):
                st.markdown(f"**📄 {t.get('file_name', '未知檔案')}**")
                st.caption(
                    f"🏷️ 類型：{t.get('contract_type', '其他')} ｜ 🎯 核心主題："
                    + "、".join(t.get("core_topics", []))
                )

    if major:
        st.markdown("---")
        st.markdown("### 🚨 重大違規 / 衝突")
        for item in major:
            render_issue_block(item)

    if general:
        st.markdown("---")
        st.markdown("### ⚠️ 一般違規")
        for item in general:
            render_issue_block(item)

    if missing:
        st.markdown("---")
        st.markdown("### 🧩 歷史規範缺漏條款")
        for item in missing:
            render_missing_block(item)

    render_compliance_scan(data.get("compliance_scan", []))
    render_gap_analysis(data.get("gap_analysis", {}))


# 主程式 UI
st.set_page_config(
    page_title="自動化合約審查系統 | Core",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("## 系統導覽")
    page = st.radio("功能模組：", ["合約審查系統", "歷史合約管理"])
    st.markdown("---")

    if st.button("注入模擬歷史報價資料"):
        if count_history_records() == 0:
            insert_history_records([
                {"vendor_name": "萬旭浤", "amount": 1000000},
                {"vendor_name": "萬旭浤", "amount": 1100000},
            ])
            st.success("✅ 歷史資料已注入。")
        else:
            st.info("ℹ️ 資料已存在。")


if page == "合約審查系統":
    st.markdown("##  合約檢核與生成系統")
    st.info(
        " **系統終端指令**：\n"
        " `/review`（合約審查）\n"
        " `/risk `（報價分析）\n"
        " `/generate `（合約生成）\n"
        " **直接輸入文字** 即可與 AI 法務助理對話討論合約內容。",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "您好！我是企業 AI 合約審查系統。請上傳廠商合約草稿後輸入 `/review`，系統將自動比對公司過往合約與資安法遵標準，為您產出合規落差報告。"
        }]
    if "draft_content" not in st.session_state:
        st.session_state.draft_content = ""
    if "review_context" not in st.session_state:
        st.session_state.review_context = None
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0
    if "last_generated_path" not in st.session_state:
        st.session_state.last_generated_path = None

    with st.sidebar:
        st.markdown("### 合約審查")
        draft_file = st.file_uploader("上傳廠商合約草稿", type=["pdf", "docx"], key="draft_file")

        if st.button("清空對話記憶"):
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.draft_content = ""
            st.session_state.review_context = None
            st.session_state.last_generated_path = None
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict):
                render_review_dashboard(msg["content"])
            else:
                st.markdown(msg["content"])

    user_msg = st.chat_input("輸入 /review、/risk、/generate 或直接提問...")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        msg_text = user_msg.strip()

        if msg_text.startswith("/review"):
            current_time = time.time()
            if current_time - st.session_state.last_request_time < 30:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "系統正在冷卻中。為確保運算資源穩定，請等待 30 秒後再提交新的審查請求。"
                })
                st.rerun()

            st.session_state.last_request_time = current_time
            draft_content = msg_text.replace("/review", "", 1).strip()

            if not draft_content and draft_file is not None:
                with st.spinner("正在解析草稿文本..."):
                    if draft_file.name.lower().endswith(".pdf"):
                        draft_content = extract_text_from_pdf(draft_file)
                    else:
                        draft_content = extract_text_from_docx(draft_file)

            if not draft_content:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ 系統未偵測到草稿內容。請貼上文字或上傳 PDF / DOCX。"
                })
                st.rerun()
            else:
                with st.spinner("正在掃描企業合規與歷史基準..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/review",
                            json={"draft_text": draft_content, "top_k": 5},
                            timeout=6000
                        )
                        response.raise_for_status()
                        review_json = response.json()

                        st.session_state.draft_content = draft_content
                        st.session_state.review_context = review_json

                        st.session_state.messages.append({
                            "role": "assistant", "content": review_json
                        })
                    except requests.exceptions.RequestException as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ API 呼叫失敗，請確認 FastAPI 伺服器已啟動且運作正常。詳細錯誤: {e}"
                        })
                st.rerun()

        elif msg_text.startswith("/risk"):
            with st.spinner("正在呼叫法務中樞 API 比對報價..."):
                user_input = msg_text.replace("/risk", "", 1).strip()
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/risk",
                        json={"user_input": user_input},
                        timeout=60
                    )
                    response.raise_for_status()
                    risk_report = response.json().get("report", "分析失敗")
                    st.session_state.messages.append({"role": "assistant", "content": risk_report})
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", "content": f"⚠️ API 呼叫失敗。錯誤: {e}"
                    })
            st.rerun()

        elif msg_text.startswith("/generate"):
            with st.spinner("正在生成合約..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/generate",
                        json={"user_input": user_msg},
                        timeout=90
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.last_generated_path = data.get("file_path")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🎉 合約生成完畢！已成功套用知識庫範本 `{data.get('template_used')}`。"
                        })
                    elif response.status_code == 404:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ 查無相關基準檔案，請先到企業智庫上傳過往合約或規範。"
                        })
                    else:
                        error_detail = response.json().get("detail", "未知錯誤")
                        st.session_state.messages.append({
                            "role": "assistant", "content": f"⚠️ 生成合約失敗：{error_detail}"
                        })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", "content": f"⚠️ API 呼叫失敗。錯誤: {e}"
                    })
            st.rerun()

        else:
            with st.spinner("正在思考您的問題..."):
                try:
                    payload_messages = []
                    for m in st.session_state.messages:
                        content = m["content"]
                        if isinstance(content, dict):
                            content = "【系統顯示了合規檢核落差報告】"
                        payload_messages.append({"role": m["role"], "content": str(content)})

                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "messages": payload_messages,
                            "draft_text": st.session_state.get("draft_content", ""),
                            "review_context": st.session_state.get("review_context")
                        },
                        timeout=6000
                    )
                    response.raise_for_status()
                    reply = response.json().get("reply", "系統目前無法回應。")
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", "content": f"⚠️ 對話 API 呼叫失敗。錯誤: {e}"
                    })
            st.rerun()

    if st.session_state.get("last_generated_path"):
        try:
            with open(st.session_state.last_generated_path, "rb") as f:
                st.download_button(
                    "📥 下載系統生成的 DOCX 合約",
                    data=f.read(),
                    file_name=os.path.basename(st.session_state.last_generated_path)
                )
        except Exception:
            pass


# 企業法遵與歷史合約管理
elif page == "歷史合約管理":
    st.markdown("## 歷史合約中樞")

    with st.container():
        new_files = st.file_uploader(
            "擴充企業智庫（上傳過往簽訂之合約 / 資安規範文檔）",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if st.button("執行語義切塊與資料入庫"):
            if new_files:
                with st.spinner("寫入中..."):
                    files_data = [("files", (f.name, f.getvalue(), f.type)) for f in new_files]
                    try:
                        res = requests.post(f"{API_BASE_URL}/upload", files=files_data, timeout=600)
                        res.raise_for_status()
                        data = res.json()
                        inserted = data.get("inserted", 0)
                        skipped = data.get("skipped", 0)

                        st.success(f"✅ 成功完成 {inserted} 份檔案的向量化入庫。")
                        if skipped:
                            st.info(f"ℹ️ 略過 {skipped} 份重複或無法解析的檔案。")

                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"上傳失敗，請確認 FastAPI 伺服器已啟動。錯誤: {e}")
            else:
                st.warning("請先選擇要上傳的檔案。")

    st.markdown("---")

    docs = cached_get_all_templates()
    st.markdown("### 企業資料庫庫資產總覽")

    if not docs:
        st.info("目前的資料庫尚無檔案！")
    else:
        types_count = {}
        for d in docs:
            ctype = d.get("contract_type", "其他")
            types_count[ctype] = types_count.get(ctype, 0) + 1

        cols = st.columns(len(types_count) + 1)
        cols[0].metric(" 資料庫檔案總數", f"{len(docs)} 份")
        for i, (ctype, count) in enumerate(types_count.items(), 1):
            cols[i].metric(f"🏷️ {ctype}", f"{count} 份")

        st.markdown("### 🔍 檢索與管理")

        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_kw = st.text_input("關鍵字搜尋（支援檔名、主題、關鍵字）", placeholder="輸入想找的合約特徵...")
        with col_filter:
            all_types = ["全部"] + list(types_count.keys())
            selected_type = st.selectbox("文件類型篩選", all_types)

        filtered_docs = []
        for d in docs:
            match_type = (selected_type == "全部" or d.get("contract_type", "其他") == selected_type)
            search_text = (
                f"{d.get('file_name', '')} "
                f"{' '.join(d.get('keywords', []))} "
                f"{' '.join(d.get('core_topics', []))}".lower()
            )
            match_kw = (not search_kw) or (search_kw.lower() in search_text)
            if match_type and match_kw:
                filtered_docs.append(d)

        st.caption(f"顯示 {len(filtered_docs)} / {len(docs)} 筆結果")

        for doc in filtered_docs:
            with st.expander(f"📄 {doc.get('file_name')} | 類型: {doc.get('contract_type', '其他')}"):
                info_col, action_col = st.columns([4, 1])

                with info_col:
                    st.write(f"**AI 分析**：{doc.get('summary', '')}")
                    if doc.get("keywords"):
                        st.caption("關鍵字：" + "、".join(doc.get("keywords", [])))
                    if doc.get("core_topics"):
                        st.caption("核心主題：" + "、".join(doc.get("core_topics", [])))
                    st.caption(f"入庫時間：{doc.get('created_at')}")

                with action_col:
                    if st.button("🗑️ 永久刪除", key=f"del_{doc['doc_id']}", type="primary", width="stretch"):
                        try:
                            res = requests.delete(f"{API_BASE_URL}/templates/{doc['doc_id']}", timeout=30)
                            res.raise_for_status()
                            st.success("刪除成功！")
                            st.cache_data.clear()
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"刪除失敗，請確認 FastAPI 運行狀態。錯誤: {e}")

    st.markdown("---")
    st.markdown("### ⚖️ 企業內部規則庫")
    st.caption("集中管理合約審查標準。系統將依據此處定義之規範，自動稽核廠商合約是否有遺漏或偏離。")

    current_rules = cached_get_compliance_rules()
    topics = list(current_rules.keys())

    with st.container(border=True):
        st.markdown("**快速新增義務主題**")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            new_topic = st.text_input("新主題名稱", placeholder="輸入新主題，例如：營業秘密保護", label_visibility="collapsed")
        with col_btn:
            if st.button("＋ 建立主題", type="primary", width="stretch"):
                if not new_topic.strip():
                    st.warning("⚠️ 請輸入主題名稱")
                elif new_topic.strip() in topics:
                    st.error("⚠️ 此主題已存在！")
                else:
                    upsert_compliance_rule(new_topic.strip(), ["請在此輸入第一條規範範例..."])
                    st.toast(f"已建立新主題：{new_topic}", icon="🎉")
                    st.cache_data.clear()
                    time.sleep(0.4)
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"#### 現有規範清單 ({len(topics)} 項)")

    if not topics:
        st.info("📭 目前尚無任何規則，請透過上方欄位新增。")
    else:
        for topic in topics:
            examples = current_rules[topic]
            with st.expander(f"**{topic}** (包含 {len(examples)} 條細則)"):
                st.caption("每換一行代表一項獨立義務。編輯完成後請點擊下方儲存。")
                text_key = f"text_{topic}"
                examples_text = st.text_area(
                    f"編輯 {topic}",
                    value="\n".join(examples),
                    height=180,
                    label_visibility="collapsed",
                    key=text_key
                )
                col_save, col_space, col_del = st.columns([2, 5, 2])
                with col_save:
                    if st.button("儲存修改", key=f"save_{topic}", type="primary", width="stretch"):
                        new_examples = [e.strip() for e in examples_text.split("\n") if e.strip()]
                        if not new_examples:
                            st.warning("⚠️ 規範內容不能為空")
                        else:
                            upsert_compliance_rule(topic, new_examples)
                            st.toast(f"「{topic}」已更新成功！", icon="✅")
                            st.cache_data.clear()
                            time.sleep(0.4)
                            st.rerun()
                
                with col_del:
                    with st.popover("刪除主題", width="stretch"):
                        st.markdown(f"確定要永久刪除 **{topic}** 嗎？")
                        if st.button("確認刪除", key=f"del_{topic}", type="primary", width="stretch"):
                            delete_compliance_rule(topic)
                            st.toast(f"已刪除主題：{topic}")
                            st.cache_data.clear()
                            time.sleep(0.4)
                            st.rerun()