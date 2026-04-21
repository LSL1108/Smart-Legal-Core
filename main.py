from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import datetime
import os

from models import ReviewReport, UserRequestIntent, ChatRequest
from services import (
    search_relevant_templates, review_articles_individually, normalize_review_json,
    assess_price_risk, llm_parse_user_request, generate_contract_from_template,
    llm_chat, handle_upload  
)
from utils import make_output_path, parse_template_selector 
from database import (
    insert_audit_log, get_template_by_selector, search_templates_sql
)

# 初始化 FastAPI 應用程式
app = FastAPI(
    title="企業智能法務中樞 API",
    description="提供合規檢核、報價風險分析與合約生成之核心服務", 
    version="1.0.0"
)

# Request Models
class ReviewRequest(BaseModel):
    draft_text: str
    top_k: int = 5

class RiskRequest(BaseModel):
    user_input: str

class GenerateRequest(BaseModel):
    user_input: str

# API Endpoints 
@app.post("/api/review", response_model=ReviewReport, summary="動態語義合約審查")
def api_review_contract(request: ReviewRequest):
    """
    接收合約草稿文本，調用 RAG 向量檢索與 LLM 進行逐條審查，回傳完整的風險報告。
    """
    if not request.draft_text.strip():
        raise HTTPException(status_code=400, detail="草稿內容不能為空")

    try:
        top_templates, top_chunks, articles = search_relevant_templates(
            request.draft_text, top_k=request.top_k
        )

        if not top_templates:
            return ReviewReport(
                summary="⚠️ 目前企業智庫中找不到可用之歷史基準或規範，無法進行合規掃描。", 
                score=0
            )
        
        raw_review = review_articles_individually(
            request.draft_text, top_templates, articles
        )
 
        review_json = normalize_review_json(
            raw_review, top_templates, articles, top_chunks, request.draft_text
        )

        insert_audit_log(
            username="api_user",
            action="合規掃描 (API)", 
            target="草稿",
            detail=f"參考企業基準數={len(top_templates)}，發現重大風險={len(review_json.get('major_issues', []))}項" 
        )

        return review_json

    except Exception as e:
        logging.error(f"API 合規掃描發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk", summary="歷史報價風險預警")
def api_assess_risk(request: RiskRequest):
    """
    分析使用者輸入中的廠商名稱與報價金額，比對歷史資料庫並回傳風險評估報告。
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="輸入內容不能為空")

    try:
        risk_report = assess_price_risk(request.user_input)
        
        insert_audit_log(
            username="api_user",
            action="報價風險分析 (API)",
            detail=request.user_input[:200]
        )
        
        return {"report": risk_report}
        
    except Exception as e:
        logging.error(f"API 風險評估發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", summary="自動合約套版生成")
def api_generate_contract(request: GenerateRequest):
    """
    解析使用者意圖，尋找對應基準文件，並將參數套入 DOCX 生成合約實體檔案。
    回傳生成的檔案路徑。
    """
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="輸入內容不能為空")

    try:
        parsed = llm_parse_user_request(request.user_input)
        ct = parsed.get("contract_type", "其他")

        base = (
            get_template_by_selector(parse_template_selector(request.user_input))
            or (search_templates_sql(ct, request.user_input, 1) or [None])[0]
        )

        if not base:
            raise HTTPException(status_code=404, detail="查無相關基準檔案，請先至企業智庫上傳過往合約或規範。") 

        out_name = f"{ct}_自動生成_{datetime.date.today().strftime('%Y%m%d')}.docx"
        out_path = make_output_path(out_name)

        ok = generate_contract_from_template(base["storage_path"], out_path, parsed.get("fields", {}))

        if ok:
            insert_audit_log(
                username="api_user",
                action="合約生成 (API)",
                target=base["file_name"],
                detail=f"輸出檔案：{out_name}"
            )
            return {"status": "success", "file_path": out_path, "file_name": out_name, "template_used": base["file_name"]}
        else:
            raise HTTPException(status_code=500, detail="DOCX 檔案生成失敗。")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"API 合約生成發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", summary="法務助理自由對話")
def api_chat_assistant(request: ChatRequest):
    """
    接收前端的對話紀錄與當前合約上下文，交由 LLM 生成回覆
    """
    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        reply = llm_chat(messages_dict, request.draft_text, request.review_context)
        
        insert_audit_log(
            username="api_user",
            action="法遵與法務對話 (API)", 
            detail=f"User: {messages_dict[-1]['content'][:50]}..." if messages_dict else "新對話"
        )
        
        return {"reply": reply}
        
    except Exception as e:
        logging.error(f"API 對話發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 企業智庫管理 API 
@app.post("/api/upload", summary="批次上傳企業基準入庫") 
def api_upload_template(files: List[UploadFile] = File(...)):
    """
    接收前端傳來的檔案，並轉交給服務層進行向量化入庫
    """
    class DummyFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        def getvalue(self):
            return self.content
    
    dummy_files = []
    for f in files:
        content = f.file.read()
        dummy_files.append(DummyFile(f.filename, content))
    
    inserted, skipped = handle_upload(dummy_files)
    
    insert_audit_log(
        username="api_user",
        action="批次上傳企業基準 (API)", 
        detail=f"成功入庫={inserted}，略過={skipped}，檔案數={len(dummy_files)}"
    )
    return {"inserted": inserted, "skipped": skipped}


@app.delete("/api/templates/{doc_id}", summary="永久刪除基準檔案") 
def api_delete_template(doc_id: str):
    """
    接收前端的刪除指令，清除 SQLite 與 ChromaDB 中的資料
    """
    from database import (
        template_collection, chunk_collection, 
        delete_template_by_doc_id, get_template_by_doc_id, insert_audit_log
    )
    import os, logging

    try:
        doc = get_template_by_doc_id(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="找不到該基準檔案")
        try:
            template_collection.delete(ids=[doc_id])
            results = chunk_collection.get(where={"doc_id": doc_id})
            chunk_ids = results.get("ids", [])
            if chunk_ids:
                chunk_collection.delete(ids=chunk_ids)
        except Exception as e:
            logging.warning(f"刪除向量資料失敗: {e}")
        delete_template_by_doc_id(doc_id)
        if os.path.exists(doc["storage_path"]):
            try:
                os.remove(doc["storage_path"])
            except Exception:
                pass
        insert_audit_log(
            username="api_user",
            action="刪除基準檔案 (API)",
            target=doc.get("file_name", "未知"),
            detail=f"doc_id={doc_id}"
        )
        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "法務與合規中樞 API 運作正常"}