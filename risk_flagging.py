# risk_detector.py
# Member 3 — Risk Flagging Agent
# Reads: clauses
# Writes: risk_flags

from langchain_groq import ChatGroq          
from dotenv import load_dotenv
from state_schema import GraphState   # import the shared clipboard

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")  


def risk_detector_node(state: GraphState) -> GraphState:
    """
    Reads each clause from state["clauses"].
    For each clause, asks Claude if it's risky.
    Writes results into state["risk_flags"].
    """

    clauses = state["clauses"]   # read the clauses (filled by Member 2)
    risk_flags = []              # we'll build this list up

    for clause in clauses:
        clause_id = clause["clause_id"]
        clause_text = clause["text"]

        prompt = f"""
        You are a legal risk detector. Analyze the following clause from a legal document.

        Clause:
        "{clause_text}"

        Your job:
        1. Decide the risk level: "high", "medium", or "low"
        2. Write a short reason explaining why it is risky (1-2 sentences in plain English)

        Rules:
        - "high"   → clause is very one-sided, waives important rights, hides big costs
        - "medium" → clause is concerning but not extreme
        - "low"    → clause is standard and mostly fine

        Respond in EXACTLY this format, nothing else:
        RISK_LEVEL: <high/medium/low>
        REASON: <your explanation>
        """

        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Parse the response into the exact format the team expects
        risk_level = "low"    # default fallback
        reason = "No specific risk identified."

        for line in raw.split("\n"):
            if line.startswith("RISK_LEVEL:"):
                risk_level = line.replace("RISK_LEVEL:", "").strip().lower()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        # Add this clause's result to the list
        risk_flags.append({
            "clause_id": clause_id,       # matches the clause it came from
            "risk_level": risk_level,     # "high", "medium", or "low"
            "reason": reason              # plain English explanation
        })

    # Write into state and return — touch NOTHING else
    state["risk_flags"] = risk_flags
    return state


# ── TEST IT LOCALLY ───────────────────────────────────────────
# This runs only when YOU run this file directly.
# Your supervisor teammate doesn't need to touch this.

if __name__ == "__main__":

    # Fake data simulating what Member 2 (Extractor) would pass you
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
            },
            {
                "clause_id": "clause_4",
                "text": "Either party may terminate this agreement with 30 days written notice."
            }
        ],
        "risk_flags": [],
        "simplified_clauses": [],
        "validation_status": "",
        "retry_count": 0,
        "final_report": ""
    }

    # Run your bot
    result = risk_detector_node(fake_state)

    # Print results nicely
    print("\n🚨 RISK FLAGS:\n")
    for flag in result["risk_flags"]:
        level = flag["risk_level"].upper()
        emoji = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
        print(f"{emoji} [{level}] {flag['clause_id']}")
        print(f"   {flag['reason']}")
        print()
