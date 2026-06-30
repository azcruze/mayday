# validator.py
# Member 5 — Validator Agent
# Reads:  simplified_clauses, clauses
# Writes: validation_status, retry_count

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state_schema import GraphState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

MAX_RETRIES = 3  # how many times we allow the simplifier to redo before giving up


# ── BOT 1: VALIDATOR NODE ─────────────────────────────────────
# Compares each original clause vs its simplified version.
# Writes "passed", "failed", or "needs_human_review" into validation_status.

def validator_node(state: GraphState) -> GraphState:
    """
    Goes through each clause pair (original vs simplified).
    If ANY clause fails the check → mark as failed.
    If ALL pass → mark as passed.
    If retry limit hit → mark as needs_human_review.
    """

    clauses = state["clauses"]                     # original clauses (from Member 2)
    simplified_clauses = state["simplified_clauses"]  # simplified (from Member 4)
    retry_count = state["retry_count"]             # how many retries so far

    # Safety check — if retry limit already hit, skip validation entirely
    if retry_count >= MAX_RETRIES:
        print(f"⚠️  Max retries ({MAX_RETRIES}) hit — marking as needs_human_review")
        state["validation_status"] = "needs_human_review"
        return state

    # Build a lookup so we can easily find simplified version by clause_id
    # e.g. { "clause_1": "You can cancel anytime.", "clause_2": "..." }
    simplified_lookup = {
        item["clause_id"]: item["simplified_text"]
        for item in simplified_clauses
    }

    failed_clauses = []   # we'll collect any failures here

    for clause in clauses:
        clause_id = clause["clause_id"]
        original_text = clause["text"]
        simplified_text = simplified_lookup.get(clause_id, "")

        # If simplified version is missing entirely, that's an auto-fail
        if not simplified_text:
            failed_clauses.append(clause_id)
            continue

        prompt = f"""
        You are a legal validation expert.

        Compare this original legal clause with its simplified version.

        ORIGINAL:
        "{original_text}"

        SIMPLIFIED:
        "{simplified_text}"

        Does the simplified version preserve the full legal meaning of the original?
        Things to check:
        - Are all key obligations still present?
        - Are deadlines or numbers (dates, fees, days) preserved?
        - Is any important condition missing or changed?
        - Is the meaning accidentally reversed or softened too much?

        Respond in EXACTLY this format, nothing else:
        RESULT: <passed/failed>
        REASON: <one sentence explanation — only needed if failed, write "OK" if passed>
        """

        response = llm.invoke(prompt)
        raw = response.content.strip()

        result = "failed"   # default to failed for safety
        reason = ""

        for line in raw.split("\n"):
            if line.startswith("RESULT:"):
                result = line.replace("RESULT:", "").strip().lower()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        print(f"  {'✅' if result == 'passed' else '❌'} {clause_id}: {result} — {reason}")

        if result == "failed":
            failed_clauses.append(clause_id)

    # Decide overall status
    if failed_clauses:
        print(f"\n❌ Validation FAILED for: {failed_clauses}")
        state["validation_status"] = "failed"
        state["retry_count"] = retry_count + 1   # increment retry counter
    else:
        print(f"\n✅ Validation PASSED — all clauses look good!")
        state["validation_status"] = "passed"

    return state


# ── BOT 2: VALIDATION ROUTER ──────────────────────────────────
# This is NOT a bot — it's a router function.
# LangGraph calls this after validator_node to decide where to go next.
# It must return the NAME of the next node as a string.

def validation_router(state: GraphState) -> str:
    """
    Reads validation_status and decides the next step.

    Returns:
      "simplifier"      → go back, redo the simplification
      "supervisor"      → all good, move forward
      "needs_human_review" → gave up after max retries, move forward anyway
    """

    status = state["validation_status"]

    if status == "passed":
        print("🟢 Router: sending to supervisor")
        return "supervisor"

    elif status == "needs_human_review":
        print("🟠 Router: max retries hit, sending to supervisor with human review flag")
        return "supervisor"

    else:  # "failed"
        print(f"🔴 Router: failed (retry {state['retry_count']}/{MAX_RETRIES}), sending back to simplifier")
        return "simplifier"


# ── TEST IT ───────────────────────────────────────────────────
# We intentionally feed it a BAD simplification to prove the loop fires.

if __name__ == "__main__":

    print("=" * 50)
    print("TEST 1: Bad simplification (should FAIL and retry)")
    print("=" * 50)

    bad_state: GraphState = {
        "document_id": "doc_001",
        "raw_text": "",
        "clauses": [
            {
                "clause_id": "clause_1",
                "text": "This agreement automatically renews each year unless cancelled in writing 30 days prior to the renewal date."
            },
            {
                "clause_id": "clause_2",
                "text": "By using this service, you agree to binding arbitration and waive your right to a jury trial."
            }
        ],
        "risk_flags": [],
        "simplified_clauses": [
            {
                "clause_id": "clause_1",
                # ❌ intentionally bad — removed the 30 day notice requirement
                "simplified_text": "The contract renews every year."
            },
            {
                "clause_id": "clause_2",
                # ❌ intentionally bad — completely missed the arbitration/no jury part
                "simplified_text": "You agree to use this service."
            }
        ],
        "validation_status": "",
        "retry_count": 0,
        "final_report": ""
    }

    result = validator_node(bad_state)
    next_step = validation_router(result)
    print(f"\n➡️  Router decision: go to '{next_step}'")

    print("\n")
    print("=" * 50)
    print("TEST 2: Good simplification (should PASS)")
    print("=" * 50)

    good_state: GraphState = {
        "document_id": "doc_002",
        "raw_text": "",
        "clauses": [
            {
                "clause_id": "clause_1",
                "text": "This agreement automatically renews each year unless cancelled in writing 30 days prior to the renewal date."
            },
            {
                "clause_id": "clause_2",
                "text": "By using this service, you agree to binding arbitration and waive your right to a jury trial."
            }
        ],
        "risk_flags": [],
        "simplified_clauses": [
            {
                "clause_id": "clause_1",
                # ✅ good — preserved the 30 day notice detail
                "simplified_text": "This contract automatically renews every year. To stop it, you must cancel in writing at least 30 days before the renewal date."
            },
            {
                "clause_id": "clause_2",
                # ✅ good — preserved arbitration and no jury trial
                "simplified_text": "By using this service, you give up your right to go to court or have a jury trial. Any disputes will be handled through arbitration instead."
            }
        ],
        "validation_status": "",
        "retry_count": 0,
        "final_report": ""
    }

    result2 = validator_node(good_state)
    next_step2 = validation_router(result2)
    print(f"\n➡️  Router decision: go to '{next_step2}'")

    print("\n")
    print("=" * 50)
    print("TEST 3: Max retries hit (should mark needs_human_review)")
    print("=" * 50)

    maxed_state: GraphState = {
        **bad_state,
        "retry_count": 3,     # already hit the limit
        "validation_status": ""
    }

    result3 = validator_node(maxed_state)
    next_step3 = validation_router(result3)
    print(f"\n➡️  Router decision: go to '{next_step3}'")
