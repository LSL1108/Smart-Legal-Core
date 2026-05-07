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
    contract_type: str = Field(default="", description="本次草稿判定之合約類型，例如：開發合約、維護合約、保險代理合約")
    clause_topic: str = Field(default="", description="草稿條文本身所屬主題")
    rule_topic: str = Field(default="", description="命中的企業紅線或法遵規則主題")
    evidence_topic: str = Field(default="", description="RAG / 歷史基準引用片段所屬主題")
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
    contract_type: str = Field(default="", description="本次草稿判定之合約類型")
    evidence_topic: str = Field(default="", description="缺漏判斷所依據之歷史基準或法遵 topic")
    why_missing: str
    suggestion: str
    suggested_draft: str = Field(default="")
    source: str
    template_snippet: str = Field(default="")


class ComplianceScanResult(BaseModel):
    requirement: str = Field(default="")
    topic: str = Field(default="", description="章則義務所屬 topic")
    contract_type: str = Field(default="", description="適用的合約類型")
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
    contract_type: str = Field(default="", description="本次差距分析適用的合約類型")
    evidence_topic: str = Field(default="", description="歷史合約或法遵依據所屬 topic")
    other_vendors_coverage: str = Field(default="")
    current_vendor_status: str = Field(default="")
    vendor_refuse_scenario: VendorRefuseScenario = Field(default_factory=VendorRefuseScenario)


class GapAnalysis(BaseModel):
    gap_summary: str = Field(default="")
    gaps: List[ContractGap] = Field(default_factory=list)


class ReviewReport(BaseModel):
    contract_type_guess: str = Field(default="未判定")
    contract_mode: str = Field(default="", description="合約審查模式，例如：開發、維護、保險代理")
    enabled_ruleset: List[str] = Field(default_factory=list, description="依合約類型啟用的規則 topic")
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
    agency_scope: str = Field(default="", description="保險代理合約適用：代理招攬或委託服務範圍")
    insurance_product_type: str = Field(default="", description="保險代理合約適用：壽險、產險、投資型保險等商品類型")
    commission_terms: str = Field(default="", description="保險代理合約適用：佣酬、佣金或服務費約定")


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
