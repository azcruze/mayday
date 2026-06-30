"""
Clause Extractor Agent (Joshlin)

Splits state["raw_text"] into a list of clauses, stores them in
state["clauses"], and persists each clause to a Supabase `clauses` table.

Strategy:
  1. Rule-based splitter first (regex on numbered sections / "Article X" /
     paragraph breaks). Fast, free, deterministic.
  2. If the rule-based split looks bad (e.g. found 0-1 clauses on a
     multi-paragraph doc, or one "clause" is suspiciously huge), fall back
     to an LLM call that does semantic splitting.

Run standalone for a quick test:
    python extractor.py path/to/sample.txt
"""
from __future__ import annotations

import os
import re
import json
import uuid
from typing import List

from state_schema import GraphState

Clause = dict  # {"clause_id": str, "text": str} — per team's GraphState contract

# ---------------------------------------------------------------------------
# 1. Rule-based splitter
# ---------------------------------------------------------------------------

# Matches the START of a new clause: numbered ("1.", "2.1", "(a)"), or
# "Article X" / "Section X" headers. Each match marks a split point.
_CLAUSE_HEADER_RE = re.compile(
    r"""
    ^\s*(
        \d+(\.\d+)*\.?          # 1.   2.3   12.4.1
        |
        \([a-zA-Z0-9]+\)        # (a)  (1)  (iv)
        |
        Article\s+\d+           # Article 4
        |
        Section\s+\d+(\.\d+)*    # Section 2.1
        |
        Clause\s+\d+(\.\d+)*     # Clause 3
    )
    \s*[\.\)\-:]?\s+
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)


def _split_rule_based(raw_text: str) -> List[Clause]:
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    matches = list(_CLAUSE_HEADER_RE.finditer(raw_text))

    if matches:
        clauses: List[Clause] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
            chunk = raw_text[start:end].strip()
            if not chunk:
                continue
            # Sequential id per team convention (e.g. "clause_1"); the original
            # header text (e.g. "Article 4") stays in the clause body itself.
            clauses.append({"clause_id": f"clause_{len(clauses) + 1}", "text": chunk})
        if clauses:
            return clauses

    # Fallback within rule-based: no numbered headers found, split on blank lines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    return [{"clause_id": f"clause_{i + 1}", "text": p} for i, p in enumerate(paragraphs)]


def _looks_bad(clauses: List[Clause], raw_text: str) -> bool:
    """Heuristic check for whether the rule-based split is poor quality and
    we should consider falling back to an LLM."""
    if len(clauses) <= 1 and len(raw_text) > 800:
        return True  # long doc collapsed into one "clause" — header pattern didn't match
    if clauses:
        longest = max(len(c["text"]) for c in clauses)
        if longest > 0.7 * len(raw_text) and len(clauses) > 1:
            return True  # one clause swallowed most of the document
    return False


# ---------------------------------------------------------------------------
# 2. LLM-based splitter (fallback, optional)
# ---------------------------------------------------------------------------

def _split_with_llm(raw_text: str) -> List[Clause]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[extractor_node] ANTHROPIC_API_KEY not set — skipping LLM fallback, "
              "keeping rule-based result.")
        return []

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatAnthropic(
        model=os.environ.get("AGENT_MODEL", "claude-sonnet-4-5-20250929"),
        temperature=0,
        max_tokens=4096,
    )

    system = (
        "You split legal/contract documents into individual clauses or sections. "
        "Return ONLY valid JSON: a list of objects with keys 'clause_id' and 'text'. "
        "clause_id must be sequential in the form 'clause_1', 'clause_2', 'clause_3', etc. "
        "text should be the verbatim clause text, not summarized or altered. "
        "Do not include any preamble, explanation, or markdown fences — JSON only."
    )
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=raw_text)])
    except Exception as e:
        print(f"[extractor_node] LLM fallback call failed ({e}) — keeping rule-based result.")
        return []

    content = response.content.strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(content)
        clauses = [{"clause_id": str(c["clause_id"]), "text": c["text"]} for c in parsed]
        return clauses
    except (json.JSONDecodeError, KeyError, TypeError):
        # LLM gave malformed output — better to keep the rule-based result
        # than crash the pipeline. Caller decides what to do with empty list.
        return []


# ---------------------------------------------------------------------------
# 3. Supabase persistence
# ---------------------------------------------------------------------------

def _save_clauses_to_supabase(document_id: str, clauses: List[Clause]) -> None:
    """Insert each clause as a row into the `clauses` table.
    Requires SUPABASE_URL and SUPABASE_KEY env vars. Silently skips (with a
    warning) if they aren't set, so the node still works in local/dev mode
    without a live Supabase project."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("[extractor_node] SUPABASE_URL/SUPABASE_KEY not set — skipping persistence.")
        return

    from supabase import create_client

    client = create_client(url, key)
    rows = [
        {
            "clause_id": c["clause_id"],
            "document_id": document_id,
            "original_text": c["text"],
        }
        for c in clauses
    ]
    if rows:
        # upsert on (document_id, clause_id) so reruns on the same document
        # overwrite existing rows instead of erroring on the unique constraint
        client.table("clauses").upsert(rows, on_conflict="document_id,clause_id").execute()


# ---------------------------------------------------------------------------
# 4. The node
# ---------------------------------------------------------------------------

def extractor_node(state: GraphState) -> GraphState:
    raw_text = state.get("raw_text", "")
    document_id = state.get("document_id") or str(uuid.uuid4())

    clauses = _split_rule_based(raw_text)

    if _looks_bad(clauses, raw_text):
        llm_clauses = _split_with_llm(raw_text)
        if llm_clauses:
            clauses = llm_clauses

    _save_clauses_to_supabase(document_id, clauses)

    state["document_id"] = document_id
    state["clauses"] = clauses
    return state


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = (
            "1. Definitions. In this Agreement, the following terms shall have "
            "the meanings set forth below.\n\n"
            "2. Term. This Agreement shall commence on the Effective Date and "
            "continue for a period of twelve (12) months.\n\n"
            "3. Termination. Either party may terminate this Agreement upon "
            "thirty (30) days written notice to the other party.\n\n"
            "4. Confidentiality. Each party agrees to keep confidential all "
            "non-public information disclosed by the other party."
        )

    state: GraphState = {
        "document_id": "test-doc-001",
        "raw_text": text,
        "clauses": [],
        "risk_flags": [],
        "simplified_clauses": [],
        "validation_status": "",
        "retry_count": 0,
        "final_report": "",
    }
    result = extractor_node(state)

    print(f"Found {len(result['clauses'])} clauses:\n")
    for c in result["clauses"]:
        preview = c["text"][:80].replace("\n", " ")
        print(f"  [{c['clause_id']}] {preview}...")
