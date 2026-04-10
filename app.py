import streamlit as st
import os
import datetime
import logging
import time

# 載入資料庫模組
from database import (
    count_history_records, insert_history_records, get_all_templates, 
    template_collection, chunk_collection, delete_template_by_doc_id, get_template_by_selector
)

# 載入工具模組
from utils import (
    extract_text_from_pdf, extract_text_from_docx, parse_template_selector
)

# 載入服務模組
from services import (
    search_relevant_templates, review_articles_individually, normalize_review_json,
    assess_price_risk, llm_parse_user_request, generate_contract_from_template,
    handle_upload
)
from database import search_templates_sql

# =========================
# UI 渲染輔助函式 (Helpers)
# =========================
def risk_label(risk: str) -> str:
    r = (risk or "").lower()
    if r == "critical":
        return "🔴 極高風險"
    if r == "high":
        return "🟠 高風險"
    if r == "medium":
        return "🟡 中度風險"
    return "🟢 低風險"

def render_issue_block(item: dict):
    issue_type = item.get("type", "")
    issue_type_label = ""
    if issue_type == "conflict":
        issue_type_label = "（衝突）"
    elif issue_type == "deviation":
        issue_type_label = "（偏離）"

    st.markdown(f"#### {risk_label(item.get('risk'))}｜{item.get('clause', '未命名條款')} {issue_type_label}")
    with st.container(border=True):
        if item.get("draft_text"):
            st.markdown("⚡ **草稿內容**")
            st.write(item["draft_text"])

        if item.get("template_basis"):
            st.markdown("🛡️ **模板標準依據**")
            st.write(item["template_basis"])

        if item.get("analysis"):
            st.markdown("👁️‍🗨️ **風險分析**")
            st.write(item["analysis"])

        if item.get("suggestion"):
            st.markdown("🔧 **建議修正方式**")
            st.write(item["suggestion"])

        if item.get("source"):
            st.caption(f"📚 參考來源：{item['source']}")

def render_missing_block(item: dict):
    with st.container(border=True):
        st.markdown(f"**🧩 {item.get('clause', '未命名條款')}**")
        if item.get("why_missing"):
            st.markdown("**缺漏原因**")
            st.write(item["why_missing"])
        if item.get("suggestion"):
            st.markdown("**建議補入方向**")
            st.write(item["suggestion"])
        if item.get("source"):
            st.caption(f"📚 參考來源：{item['source']}")

def render_review_dashboard(data: dict):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("🛡️ 合約安全指數", f"{data.get('score', 100)} 分")
    with col2:
        st.info(
            f"**📂 合約類型判定**：{data.get('contract_type_guess', '未判定')}\n\n"
            f"**🤖 AI 總結摘要**：{data.get('summary', '無摘要')}"
        )

    used_templates = data.get("used_templates", [])
    if used_templates:
        st.markdown("### 📚 本次審查主要參考知識庫")
        for t in used_templates:
            with st.container(border=True):
                st.markdown(f"**📄 {t.get('file_name', '未知模板')}**")
                st.caption(f"🏷️ 類型：{t.get('contract_type', '其他')} ｜ 🎯 核心主題：" + "、".join(t.get("core_topics", [])))

    major = data.get("major_issues", [])
    general = data.get("general_issues", [])
    missing = data.get("missing_clauses", [])

    if major:
        st.markdown("---")
        st.markdown("### 🚨 重大偏離 / 衝突")
        for item in major:
            render_issue_block(item)

    if general:
        st.markdown("---")
        st.markdown("### ⚠️ 一般偏離")
        for item in general:
            render_issue_block(item)

    if missing:
        st.markdown("---")
        st.markdown("### 🧩 缺漏條款")
        for item in missing:
            render_missing_block(item)

# =========================
# 主程式 UI 佈局與自訂 CSS
# =========================
st.set_page_config(
    page_title="企業智能法務中樞 | Core",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 強調數據字體 */
    [data-testid="stMetricValue"] {
        font-size: 3rem !important;
        color: #3B82F6 !important; 
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* 圓角按鈕 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    /* 調整對話框外觀 */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 系統導覽")
    page = st.radio("功能模組：", ["🤖 智能法務助理", "🗃️ 企業知識庫管理"])
    st.markdown("---")

    if st.button("🧪 注入模擬歷史報價資料"):
        if count_history_records() == 0:
            insert_history_records([
                {"vendor_name": "萬旭浤", "amount": 1000000},
                {"vendor_name": "萬旭浤", "amount": 1100000},
            ])
            st.success("✅ 歷史資料已注入。")
        else:
            st.info("ℹ️ 資料已存在。")

if page == "🤖 智能法務助理":
    st.markdown("## ⚖️ 智能合約審查與生成系統")
    st.info(
        "💡 **系統終端指令**：\n"
        "• `/review`（系統自動檢索知識庫並進行動態語義審查）\n"
        "• `/risk [文字]`（啟動歷史報價預警分析）\n"
        "• `/generate [文字]`（啟動自動合約套版生成）",
        icon="⚡"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "您好！我是您的企業 AI 法務中樞。請上傳合約草稿後輸入 `/review`，系統將自動調用 RAG 向量資料庫進行精準的風險打擊。"
        }]
    
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    with st.sidebar:
        st.markdown("### 📄 快速審查通道")
        draft_file = st.file_uploader("📂 上傳合約草稿", type=["pdf", "docx"], key="draft_file")

        if st.button("🧹 清空對話記憶"):
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.last_generated_path = None
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict):
                render_review_dashboard(msg["content"])
            else:
                st.markdown(msg["content"])

    user_msg = st.chat_input("輸入 /review、/risk 或 /generate...")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        msg_text = user_msg.strip()

        if msg_text.startswith("/review"):
            current_time = time.time()
            if current_time - st.session_state.last_request_time < 30:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⏳ 系統正在冷卻中。為確保運算資源穩定，請等待 30 秒後再提交新的審查請求。"
                })
                st.rerun()
                
            st.session_state.last_request_time = current_time

            draft_content = msg_text.replace("/review", "", 1).strip()

            if not draft_content and draft_file is not None:
                with st.spinner("⚡ 正在解析草稿文本..."):
                    if draft_file.name.lower().endswith(".pdf"):
                        draft_content = extract_text_from_pdf(draft_file)
                    else:
                        draft_content = extract_text_from_docx(draft_file)

            if not draft_content:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ 系統未偵測到草稿內容。請貼上文字或上傳 PDF / DOCX。"
                })
            else:
                with st.spinner("🧠 正在啟動 RAG 檢索與動態語義審查..."):
                    top_templates, top_chunks, articles = search_relevant_templates(draft_content, top_k=5)

                    if not top_templates:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ 目前知識庫中找不到可用模板。請先到「企業知識庫管理」上傳模板。"
                        })
                    else:
                        raw_review = review_articles_individually(
                            draft_content, top_templates, articles
                        )
                        review_json = normalize_review_json(
                            raw_review, top_templates, articles, top_chunks, draft_content
                        )
                        st.session_state.messages.append({
                            "role": "assistant", "content": review_json
                        })
            st.rerun()

        elif msg_text.startswith("/risk"):
            with st.spinner("📊 正在比對歷史報價資料庫..."):
                risk_report = assess_price_risk(msg_text.replace("/risk", "", 1).strip())
            st.session_state.messages.append({"role": "assistant", "content": risk_report})
            st.rerun()

        elif msg_text.startswith("/generate"):
            with st.spinner("⚙️ 啟動 DocxTemplate 套版生成引擎..."):
                parsed = llm_parse_user_request(user_msg)
                ct = parsed.get("contract_type", "其他")

                base = (
                    get_template_by_selector(parse_template_selector(user_msg))
                    or (search_templates_sql(ct, user_msg, 1) or [None])[0]
                )

                if not base:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "⚠️ 查無相關範本，請先到知識庫上傳模板。"
                    })
                else:
                    out_name = f"{ct}_自動生成_{datetime.date.today().strftime('%Y%m%d')}.docx"
                    from utils import make_output_path
                    out_path = make_output_path(out_name)

                    ok = generate_contract_from_template(base["storage_path"], out_path, parsed.get("fields", {}))

                    if ok:
                        st.session_state.last_generated_path = out_path
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🎉 合約生成完畢！已成功套用知識庫範本 `{base['file_name']}`。"
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚠️ 生成合約失敗，請確認 DOCX 範本內含正確的變數標籤。"
                        })
            st.rerun()

        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "未能識別指令。請使用 `/review`、`/risk` 或 `/generate`。"
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

elif page == "🗃️ 企業知識庫管理":
    st.markdown("## 🗃️ 企業知識庫維護中樞")

    with st.container():
        new_files = st.file_uploader(
            "📂 擴充向量知識庫（支援 PDF / DOCX）",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )

        if st.button("🚀 執行語義切塊與向量入庫"):
            if new_files:
                with st.spinner("⚙️ 引擎全速運轉中..."):
                    inserted, skipped = handle_upload(new_files)
                    st.success(f"✅ 成功完成 {inserted} 份模板的向量化入庫。")
                    if skipped:
                        st.info(f"ℹ️ 略過 {skipped} 份重複或無法解析的檔案。")
            else:
                st.warning("請先選擇要上傳的檔案。")

    st.markdown("---")
    docs = get_all_templates()

    # =========================
    # 知識庫儀表板與篩選器
    # =========================
    st.markdown("### 📊 知識庫資產總覽")
    
    if not docs:
        st.info("目前的知識庫空空如也，趕快上傳第一份標準模板吧！")
    else:
        # 自動統計各類合約數量
        types_count = {}
        for d in docs:
            ctype = d.get("contract_type", "其他")
            types_count[ctype] = types_count.get(ctype, 0) + 1
        
        # 動態渲染統計指標
        cols = st.columns(len(types_count) + 1)
        cols[0].metric("📚 知識庫總數", f"{len(docs)} 份")
        for i, (ctype, count) in enumerate(types_count.items(), 1):
            cols[i].metric(f"🏷️ {ctype}", f"{count} 份")
        
        st.markdown("### 🔍 檢索與管理")
        
        # 搜尋與篩選列
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_kw = st.text_input("關鍵字搜尋 (支援檔名、主題、關鍵字)", placeholder="輸入想找的合約特徵...")
        with col_filter:
            all_types = ["全部"] + list(types_count.keys())
            selected_type = st.selectbox("合約類型篩選", all_types)

        # 執行過濾邏輯
        filtered_docs = []
        for d in docs:
            match_type = (selected_type == "全部" or d.get("contract_type", "其他") == selected_type)
            
            # 建立搜尋文本池 (檔名 + 關鍵字 + 核心主題)
            search_text = f"{d.get('file_name', '')} {' '.join(d.get('keywords', []))} {' '.join(d.get('core_topics', []))}".lower()
            match_kw = (not search_kw) or (search_kw.lower() in search_text)
            
            if match_type and match_kw:
                filtered_docs.append(d)

        st.caption(f"顯示 {len(filtered_docs)} / {len(docs)} 筆結果")

        # =========================
        # 條列顯示過濾後的合約
        # =========================
        for doc in filtered_docs:
            with st.expander(f"📄 {doc.get('file_name')} | 類型: {doc.get('contract_type', '其他')}"):
                info_col, action_col = st.columns([4, 1])
                
                with info_col:
                    st.write(f"**🤖 AI 意圖分析**：{doc.get('summary', '')}")
                    if doc.get("keywords"):
                        st.caption("🔑 關鍵字：" + "、".join(doc.get("keywords", [])))
                    if doc.get("core_topics"):
                        st.caption("🎯 核心主題：" + "、".join(doc.get("core_topics", [])))
                    st.caption(f"🕒 入庫時間：{doc.get('created_at')}")
                
                with action_col:
                    if st.button("🗑️ 永久刪除", key=f"del_{doc['doc_id']}", type="primary", use_container_width=True):
                        # 刪除向量資料庫
                        try:
                            template_collection.delete(ids=[doc["doc_id"]])
                        except Exception as e:
                            logging.warning(f"刪除模板全文向量失敗: {e}")

                        try:
                            chunk_ids = [f"{doc['doc_id']}_chunk_{i}" for i in range(500)]
                            chunk_collection.delete(ids=chunk_ids)
                        except Exception as e:
                            logging.warning(f"刪除模板片段向量時發生警告: {e}")

                        # 刪除 SQLite 記錄與實體檔案
                        delete_template_by_doc_id(doc["doc_id"])
                        try:
                            if os.path.exists(doc["storage_path"]):
                                os.remove(doc["storage_path"])
                        except Exception:
                            pass

                        st.rerun()