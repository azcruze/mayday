# simplifier.py
# Member 4 — Simplifier Agent
# Reads:  clauses
# Writes: simplified_clauses

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from state_schema import GraphState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


def simplifier_node(state: GraphState) -> GraphState:
    """
    Reads each clause and rewrites it in plain English.
    Writes results into state["simplified_clauses"].
    """

    clauses = state["clauses"]
    simplified_clauses = []

    for clause in clauses:
        clause_id = clause["clause_id"]
        clause_text = clause["text"]

        prompt = f"""
        You are a legal document simplifier.
        Rewrite the following legal clause in plain simple English that anyone can understand.

        Rules:
        - Keep ALL the important details (dates, numbers, fees, deadlines)
        - Do NOT remove or soften any obligations
        - Use short sentences
        - Write for someone with no legal knowledge
        - Do NOT add anything that wasn't in the original

        LEGAL CLAUSE:
        "{clause_text}"

        Respond with ONLY the simplified version, nothing else. No labels, no preamble.
        """

        response = llm.invoke(prompt)
        simplified_text = response.content.strip()

        simplified_clauses.append({
            "clause_id": clause_id,
            "simplified_text": simplified_text
        })

        print(f"✅ Simplified {clause_id}")

    state["simplified_clauses"] = simplified_clauses
    return state


# ── TEST IT ───────────────────────────────────────────────────

if __name__ == "__main__":

    fake_state: GraphState = {
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
            },
            {
                "clause_id": "clause_3",
                "text": "The company may share your personal data with affiliated third parties for marketing purposes."
            }
        ],
        "risk_flags": [],
        "simplified_clauses": [],
        "validation_status": "",
        "retry_count": 0,
        "final_report": ""
    }

    result = simplifier_node(fake_state)

    print("\n📝 SIMPLIFIED CLAUSES:\n")
    for item in result["simplified_clauses"]:
        print(f"[{item['clause_id']}]")
        print(f"  {item['simplified_text']}")
        print()
