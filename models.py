from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from datetime import datetime


# 📝 企業智庫與歷史基準模型
class TemplateDoc(BaseModel):
    doc_id: str
    file_name: str
    file_type: str
    storage_path: str
    sha256: str
    byte_size: int
    created_at: Union[str, datetime]
    contract_type: str = Field(default="其他")
    summary: str = Field(default="")
    keywords: List[str] = Field(default_factory=list)
    template_role: str = Field(default="歷史基準與規範")
    core_topics: List[str] = Field(default_factory=list)
    source_text: str = Field(default="")
    vendor_name: str = Field(default="")
    system_name: str = Field(default="")
    service_scope: str = Field(default="")
    maintenance_type: str = Field(default="")
    industry: str = Field(default="")
    contract_name: str = Field(default="")


# ⚖️ 合規審查報告模型
class IssueDetail(BaseModel):
    article_key: str = Field(default="")
    clause: str = Field(default="未命名條款")
    issue_topic: str = Field(default="")
    type: str = Field(default="deviation", description="deviation 或 conflict")
    risk: str = Field(default="Medium", description="Critical/High/Medium/Low")
    draft_text: str = Field(default="")
    template_basis: str = Field(default="")
    template_snippet: str = Field(default="")
    analysis: str = Field(default="")
    suggestion: str = Field(default="")
    adjusted_clause: str = Field(default="")
    negotiation_notes: str = Field(default="")
    source: str = Field(default="")


class MissingClause(BaseModel):
    clause: str
    issue_topic: str = Field(default="")
    why_missing: str
    suggestion: str
    suggested_draft: str = Field(default="")
    source: str
    template_snippet: str = Field(default="")


class ComplianceScanResult(BaseModel):
    requirement: str = Field(default="")
    is_covered: bool = Field(default=False)
    found_clause: Optional[str] = Field(default=None)
    gap_description: Optional[str] = Field(default=None)
    suggested_addition: Optional[str] = Field(default=None)


class VendorRefuseScenario(BaseModel):
    risk_description: str = Field(default="")
    cost_bearing_suggestion: str = Field(default="")
    alternative_clause: str = Field(default="")


class ContractGap(BaseModel):
    topic: str = Field(default="")
    other_vendors_coverage: str = Field(default="")
    current_vendor_status: str = Field(default="")
    vendor_refuse_scenario: VendorRefuseScenario = Field(default_factory=VendorRefuseScenario)


class GapAnalysis(BaseModel):
    gap_summary: str = Field(default="")
    gaps: List[ContractGap] = Field(default_factory=list)


class ReviewReport(BaseModel):
    contract_type_guess: str = Field(default="未判定")
    summary: str = Field(default="無摘要")
    used_templates: List[dict] = Field(default_factory=list)
    major_issues: List[IssueDetail] = Field(default_factory=list)
    general_issues: List[IssueDetail] = Field(default_factory=list)
    missing_clauses: List[MissingClause] = Field(default_factory=list)
    score: int = Field(default=100)
    compliance_scan: List[ComplianceScanResult] = Field(default_factory=list)
    gap_analysis: GapAnalysis = Field(default_factory=GapAnalysis)


# 💬 LLM 解析模型
class UserRequestFields(BaseModel):
    party_a: str = Field(default="")
    party_b: str = Field(default="")
    amount: str = Field(default="")
    term: str = Field(default="")
    system_name: str = Field(default="")
    vendor_name: str = Field(default="")
    service_scope: str = Field(default="")
    maintenance_type: str = Field(default="")
    industry: str = Field(default="")
    contract_name: str = Field(default="")


class UserRequestIntent(BaseModel):
    intent: str = Field(default="generate")
    contract_type: str = Field(default="其他")
    fields: UserRequestFields = Field(default_factory=UserRequestFields)
    notes: str = Field(default="")


# 💬 對話與聊天模型
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    draft_text: Optional[str] = Field(default="")
    review_context: Optional[Dict[str, Any]] = Field(default=None)
