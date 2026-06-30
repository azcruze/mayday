from typing import TypedDict

class GraphState(TypedDict):
    document_id: str               # unique ID for the uploaded document
    raw_text: str                  # full extracted text from the document
    clauses: list[dict]            # [{"clause_id": "clause_1", "text": "..."}]
    risk_flags: list[dict]         # [{"clause_id": "clause_1", "risk_level": "high", "reason": "..."}]
    simplified_clauses: list[dict] # [{"clause_id": "clause_1", "simplified_text": "..."}]
    validation_status: str         # "passed" or "failed"
    retry_count: int               # how many times the validator sent it back for redo
    final_report: str              # the final compiled output
