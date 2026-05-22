from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import datetime
import json
import queue
import threading

from models import ReviewReport, ChatRequest
from services import (
    search_relevant_templates,
    review_articles_individually,
    normalize_review_json,
    assess_price_risk,
    llm_parse_user_request,
    generate_contract_from_template,
    handle_upload,
)
from chat_service import answer_contract_chat
from utils import make_output_path, parse_template_selector
from database import (
    insert_audit_log,
    get_template_by_selector,
    search_templates_sql,
)
from intent_detector import detect_intent


app = FastAPI(
    title="企業智能法務中樞 API",
    description="提供合規檢核、報價風險分析與合約生成之核心服務",
    version="1.0.0",
)


@app.on_event("startup")
def _warmup_reranker_on_startup():
    try:
        import threading
        from reranker import warmup
        threading.Thread(target=warmup, name="reranker-warmup", daemon=True).start()
    except Exception as e:
        logging.warning("Reranker warmup 啟動失敗（不影響系統運作）: %s", e)


class ReviewRequest(BaseModel):
    draft_text: str
    top_k: int = 5


class RiskRequest(BaseModel):
    user_input: str


class GenerateRequest(BaseModel):
    user_input: str


def should_use_review_context(user_input: str) -> bool:

    if not user_input:
        return False

    keywords = [
        "根據剛剛",
        "根據前面",
        "根據上述",
        "剛剛的審查",
        "前面的審查",
        "上述審查",
        "審查結果",
        "這份合約",
        "上述合約",
        "前面那份合約",
        "剛剛那份合約",
        "幫我改這份",
        "補進合約",
        "加入合約",
        "修改條款",
        "針對這份",
        "依照這份",
    ]

    return any(keyword in user_input for keyword in keywords)


# API Endpoints
@app.post("/api/review", response_model=ReviewReport, summary="動態語義合約審查")
def api_review_contract(request: ReviewRequest):
    if not request.draft_text.strip():
        raise HTTPException(status_code=400, detail="草稿內容不能為空")

    try:
        top_templates, top_chunks, articles = search_relevant_templates(
            request.draft_text,
            top_k=request.top_k,
        )

        if not top_templates:
            return ReviewReport(
                summary="⚠️ 目前企業智庫中找不到可用之歷史基準或規範，無法進行合規掃描。",
                score=0,
            )

        raw_review = review_articles_individually(
            request.draft_text,
            top_templates,
            articles,
        )

        review_json = normalize_review_json(
            raw_review,
            top_templates,
            articles,
            top_chunks,
            request.draft_text,
        )

        review_json["risk_cards"] = review_json.get("risk_cards") or raw_review.get(
            "risk_cards",
            review_json.get("major_issues", []) + review_json.get("general_issues", []),
        )

        insert_audit_log(
            username="api_user",
            action="合規掃描 (API)",
            target="草稿",
            detail=f"參考企業基準數={len(top_templates)}，發現重大風險={len(review_json.get('major_issues', []))}項",
        )

        return review_json

    except Exception as e:
        logging.exception("API 合規掃描發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/stream", summary="動態語義合約審查（SSE 串流進度）")
def api_review_contract_stream(request: ReviewRequest):
    if not request.draft_text.strip():
        raise HTTPException(status_code=400, detail="草稿內容不能為空")

    q = queue.Queue()

    def push_event(event_type: str, data: Dict[str, Any]):
        q.put(
            {
                "event": event_type,
                "data": data,
            }
        )

    def progress_callback(event: Dict[str, Any]):
        push_event("progress", event)

    def worker():
        try:
            push_event(
                "progress",
                {
                    "stage": "parse_request",
                    "message": "開始解析合約內容",
                    "percent": 5,
                },
            )

            top_templates, top_chunks, articles = search_relevant_templates(
                request.draft_text,
                top_k=request.top_k,
            )

            if not top_templates:
                push_event(
                    "result",
                    {
                        "contract_type_guess": "未判定",
                        "summary": "⚠️ 目前企業智庫中找不到可用之歷史基準或規範，無法進行合規掃描。",
                        "score": 0,
                        "used_templates": [],
                        "major_issues": [],
                        "general_issues": [],
                        "risk_cards": [],
                        "missing_clauses": [],
                        "compliance_scan": [],
                        "gap_analysis": {"gap_summary": "", "gaps": []},
                    },
                )
                push_event(
                    "progress",
                    {
                        "stage": "done",
                        "message": "審查完成",
                        "percent": 100,
                    },
                )
                return

            push_event(
                "progress",
                {
                    "stage": "select_templates",
                    "message": f"已找到 {len(top_templates)} 份歷史基準",
                    "percent": 25,
                },
            )

            raw_review = review_articles_individually(
                request.draft_text,
                top_templates,
                articles,
                progress_callback=progress_callback,
            )

            push_event(
                "progress",
                {
                    "stage": "finalize_report",
                    "message": "正在整理最終報告",
                    "percent": 95,
                },
            )

            review_json = normalize_review_json(
                raw_review,
                top_templates,
                articles,
                top_chunks,
                request.draft_text,
            )

            review_json["risk_cards"] = review_json.get("risk_cards") or raw_review.get(
                "risk_cards",
                review_json.get("major_issues", []) + review_json.get("general_issues", []),
            )

            insert_audit_log(
                username="api_user",
                action="合規掃描 (API SSE)",
                target="草稿",
                detail=f"參考企業基準數={len(top_templates)}，發現重大風險={len(review_json.get('major_issues', []))}項",
            )

            push_event("result", review_json)

            push_event(
                "progress",
                {
                    "stage": "done",
                    "message": "審查完成",
                    "percent": 100,
                },
            )

        except Exception as e:
            logging.exception("SSE 合規掃描發生錯誤")
            push_event("error", {"message": str(e)})

        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_generator():
        while True:
            item = q.get()
            if item is None:
                break

            yield f"event: {item['event']}\n"
            yield f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/risk", summary="歷史報價風險預警")
def api_assess_risk(request: RiskRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="輸入內容不能為空")

    try:
        risk_report = assess_price_risk(request.user_input)

        insert_audit_log(
            username="api_user",
            action="報價風險分析 (API)",
            detail=request.user_input[:200],
        )

        return {"report": risk_report}

    except Exception as e:
        logging.exception("API 風險評估發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", summary="自動合約套版生成")
def api_generate_contract(request: GenerateRequest):

    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="輸入內容不能為空")

    try:
        parsed = llm_parse_user_request(request.user_input)
        ct = parsed.get("contract_type", "其他")

        selector = parse_template_selector(request.user_input)

        base = get_template_by_selector(selector)

        if not base:
            candidates = search_templates_sql(
                query_text=request.user_input,
                filters={"contract_type": ct} if ct and ct != "其他" else None,
                limit=1,
            )
            base = candidates[0] if candidates else None

        if not base:
            raise HTTPException(
                status_code=404,
                detail="查無相關基準檔案，請先至企業智庫上傳過往合約或規範。",
            )

        out_name = f"{ct}_自動生成_{datetime.date.today().strftime('%Y%m%d')}.docx"
        out_path = make_output_path(out_name)

        ok = generate_contract_from_template(
            base["storage_path"],
            out_path,
            parsed.get("fields", {}),
        )

        if not ok:
            raise HTTPException(status_code=500, detail="DOCX 檔案生成失敗。")

        insert_audit_log(
            username="api_user",
            action="合約生成 (API)",
            target=base.get("file_name", "未知模板"),
            detail=f"輸出檔案：{out_name}",
        )

        return {
            "status": "success",
            "file_path": out_path,
            "file_name": out_name,
            "template_used": base.get("file_name", "未知模板"),
        }

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("API 合約生成發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", summary="法務助理自由對話")
def api_chat_assistant(request: ChatRequest):

    try:
        messages_dict = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]

        latest_user_input = ""
        for msg in reversed(messages_dict):
            if msg.get("role") == "user":
                latest_user_input = msg.get("content", "")
                break

        intent = detect_intent(latest_user_input)

        effective_review_context = request.review_context
        effective_draft_text = request.draft_text

        if intent in ["chat", "security_explanation"]:
            effective_review_context = None
            effective_draft_text = ""

        elif not should_use_review_context(latest_user_input):
            effective_review_context = None

        chat_result = answer_contract_chat(
            messages=messages_dict,
            draft_text=effective_draft_text,
            review_context=effective_review_context,
        )

        reply = chat_result.get("reply", "")

        insert_audit_log(
            username="api_user",
            action="法遵與法務對話 (API)",
            detail=(
                f"Intent: {intent} | Tool: {chat_result.get('tool_name')} | User: {latest_user_input[:50]}..."
                if latest_user_input
                else "新對話"
            ),
        )

        return {
            "reply": reply,
            "intent": intent,
            "used_review_context": effective_review_context is not None,
            "used_draft_text": bool(effective_draft_text and effective_draft_text.strip()),
            "tool_name": chat_result.get("tool_name"),
            "latency_sec": chat_result.get("latency_sec"),
            "extra": chat_result.get("extra"),
        }

    except Exception as e:
        logging.exception("API 對話發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", summary="批次上傳企業基準入庫")
def api_upload_template(files: List[UploadFile] = File(...)):

    class DummyFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content

        def getvalue(self):
            return self.content

    try:
        dummy_files = []

        for f in files:
            content = f.file.read()
            dummy_files.append(DummyFile(f.filename, content))

        inserted, skipped = handle_upload(dummy_files)

        insert_audit_log(
            username="api_user",
            action="批次上傳企業基準 (API)",
            detail=f"成功入庫={inserted}，略過={skipped}，檔案數={len(dummy_files)}",
        )

        return {"inserted": inserted, "skipped": skipped}

    except Exception as e:
        logging.exception("API 批次上傳發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/templates/{doc_id}", summary="永久刪除基準檔案")
def api_delete_template(doc_id: str):

    from database import (
        template_collection,
        chunk_collection,
        delete_template_by_doc_id,
        get_template_by_doc_id,
    )
    import os

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

        storage_path = doc.get("storage_path")
        if storage_path and os.path.exists(storage_path):
            try:
                os.remove(storage_path)
            except Exception as e:
                logging.warning(f"刪除實體檔案失敗: {e}")

        insert_audit_log(
            username="api_user",
            action="刪除基準檔案 (API)",
            target=doc.get("file_name", "未知"),
            detail=f"doc_id={doc_id}",
        )

        return {"status": "success"}

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("API 刪除基準檔案發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "法務與合規中樞 API 運作正常",
    }