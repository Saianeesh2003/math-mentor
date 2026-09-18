
import json
from dotenv import load_dotenv
from google import genai
import os

from typing import TypedDict
load_dotenv()
MODEL_NAME = "gemini-3.5-flash"
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

KNOWLEDGE_BASE = {
    "algebra": """Quadratic formula: x = (-b ± √(b²-4ac)) / 2a. Discriminant D=b²-4ac: D>0 gives two real roots, D=0 one real root, D<0 no real roots.
Arithmetic progression: nth term = a + (n-1)d; sum = n/2 × (2a + (n-1)d).
Geometric progression: nth term = ar^(n-1); sum of n terms = a(1-r^n)/(1-r).""",
    "calculus": """Derivative rules: d/dx(x^n)=nx^(n-1), d/dx(sin x)=cos x, d/dx(cos x)=-sin x, d/dx(e^x)=e^x, d/dx(ln x)=1/x.
Integration: ∫x^n dx=x^(n+1)/(n+1)+C; ∫sin(x)dx=-cos(x)+C; ∫cos(x)dx=sin(x)+C.
For extrema, solve f'(x)=0; f''(x)<0 indicates a maximum and f''(x)>0 a minimum.""",
    "probability": """P(A∪B)=P(A)+P(B)-P(A∩B). P(A|B)=P(A∩B)/P(B). Independent events: P(A∩B)=P(A)P(B).
Bayes theorem: P(A|B)=P(B|A)P(A)/P(B). Binomial: P(X=r)=C(n,r)p^r(1-p)^(n-r), mean=np, variance=np(1-p).""",
}
class State(TypedDict):
    query: str
    route_label: str | None
    final_answer: str | None
    retrieved_context: str | None
    verification_notes: str | None
    explanation: str | None
    needs_hitl: bool | None
    verified: bool | None
    parsed_problem: dict | None
    total_cost: float 

def retrieve_context(query: str, topic: str) -> str:
    return KNOWLEDGE_BASE.get(topic, "No relevant formulas found.")


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Generate a response with Gemini while keeping agent prompts explicit."""
    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=f"System instructions:\n{system_prompt}\n\nUser request:\n{user_prompt}",
    )
    return interaction.output_text.strip()

def parser_agent(state: State):
    query = state["query"]
    
   
    SYSTEM_PROMPT = """You are a math problem parser.
Clean and structure the given math problem.
Return ONLY valid JSON, no explanation, no markdown, no backticks.

Format:
{
  "problem_text": "cleaned version of the problem",
  "topic": "algebra or calculus or probability",
  "variables": ["list", "of", "variables"],
  "constraints": ["any constraints like x > 0"],
  "needs_clarification": false
}

Set needs_clarification to true if ANY of these:
- no equation or expression is given (e.g. just "find x" or "solve the equation")
- the problem is missing key numbers or variables
- it is impossible to solve without more information
- the question has fewer than 5 words and no math symbols"""

    raw = generate_text(SYSTEM_PROMPT, f"Parse this math problem: {query}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        
        parsed = {
            "problem_text": query,
            "topic": "unknown",
            "variables": [],
            "constraints": [],
            "needs_clarification": False
        }
    
    return {
        "query": parsed["problem_text"], 
        "parsed_problem": parsed,
        "needs_hitl": parsed["needs_clarification"],
        "total_cost": 0.0
    }

     

def classify_msg(state:State):
    query=state["query"]
    SYSTEM_PROMPT = """You classify JEE-style math questions.
Allowed labels:
A = Algebra
P = Probability
C = Calculus
Return exactly one uppercase letter only: A or P or C.
Do not explain.
Do not add punctuation.
Do not add any other words.
"""
   
    label = generate_text(SYSTEM_PROMPT, f"Question: {query}").upper()
    return {"route_label": label, "total_cost": state.get("total_cost") or 0.0}
    


def route_bot(state: State) -> str:
    label = (state.get("route_label") or "").strip().upper()
    if label == "A":
        return "algebra_bot"
    elif label == "C":
        return "calculus_bot"
    elif label == "P":
        return "prob_bot"
    else:
        raise ValueError(f"Unrecognized classifier result: {label}")
   
    
def algebra_bot(state:State)    :
    query = state["query"]
    context = retrieve_context(query, "algebra")
    SYSTEM_PROMPT = """
        You are an expert JEE Algebra tutor.
Solve the problem step by step.
Keep the explanation clear and concise.
You are given relevant formulas and context to help solve the problem.
Use the provided context to solve step by step.
Give the final answer separately at the end.
    """
    answer = generate_text(SYSTEM_PROMPT, f"Relevant formulas:\n{context}\n\nQuestion: {query}")
    return {
    "final_answer": answer,
    "retrieved_context": context,
    "total_cost": state.get("total_cost") or 0.0
}

def calculus_bot(state:State):
    query=state["query"]
    context = retrieve_context(query, "calculus")
    SYSTEM_PROMPT = """
        You are an expert JEE Calculus tutor.
Solve limits, derivatives, basic integrals, and simple optimization step by step.
Keep the explanation clear and concise.
You are given relevant formulas and context to help solve the problem.
Use the provided context to solve step by step.
Give the final answer separately at the end
    """
    answer = generate_text(SYSTEM_PROMPT, f"Relevant formulas:\n{context}\n\nQuestion: {query}")
    return {
    "final_answer": answer,
    "retrieved_context": context,
    "total_cost": state.get("total_cost") or 0.0
}
def prob_bot(state:State):
    query=state["query"]
    context = retrieve_context(query, "probability")
    SYSTEM_PROMPT = """
        You are an expert JEE Probability tutor.
Solve problems involving conditional probability, Bayes' theorem, expectation, and variance step by step.
Keep the explanation clear and concise.
You are given relevant formulas and context to help solve the problem.
Use the provided context to solve step by step.
Give the final answer separately at the end
    """
    answer = generate_text(SYSTEM_PROMPT, f"Relevant formulas:\n{context}\n\nQuestion: {query}")
    return {
    "final_answer": answer,
    "retrieved_context": context,
    "total_cost": state.get("total_cost") or 0.0
}
def verifier(state: State):
    query = state["query"]
    answer = state["final_answer"]
    context = state["retrieved_context"]

    SYSTEM_PROMPT = """You are a strict JEE math answer verifier.
Check the answer for:
1. Correctness — is the math right?
2. Domain/units — are constraints respected (e.g. no sqrt of negative, no log of zero)?
3. Edge cases — are boundary conditions handled?

Respond in this exact format:
VERDICT: CONFIDENT or UNSURE
NOTES: one sentence explaining what you checked or what looks wrong"""

    raw = generate_text(SYSTEM_PROMPT, f"Formulas:\n{context}\n\nQuestion: {query}\n\nAnswer:\n{answer}")
    verdict, notes = "CONFIDENT", ""
    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().upper()
        elif line.startswith("NOTES:"):
            notes = line.replace("NOTES:", "").strip()
    return {
    "verified": verdict == "CONFIDENT",
    "verification_notes": notes,
    "needs_hitl": verdict != "CONFIDENT",
    "total_cost": state.get("total_cost") or 0.0
}


def explainer(state: State):
    query = state["query"]
    answer = state["final_answer"]
    context = state["retrieved_context"]
    SYSTEM_PROMPT = """You are a friendly JEE tutor explaining a solution to a student.
Do NOT re-solve. Explain WHY each step works in simple language.
Use numbered steps, mention which formula was used and why.
Highlight common mistakes to avoid. End with a one-line takeaway."""
    explanation = generate_text(SYSTEM_PROMPT, f"Formulas:\n{context}\n\nQuestion: {query}\n\nSolution:\n{answer}\n\nExplain to a JEE student.")
    return {
        "explanation": explanation,
        "total_cost": state.get("total_cost") or 0.0
    }

class TutorGraph:
    """Dependency-free orchestration for the tutor's agent stages."""

    def invoke(self, state: State) -> State:
        state.update(parser_agent(state))
        state.update(classify_msg(state))
        route = route_bot(state)
        if route == "algebra_bot":
            state.update(algebra_bot(state))
        elif route == "calculus_bot":
            state.update(calculus_bot(state))
        else:
            state.update(prob_bot(state))
        state.update(verifier(state))
        state.update(explainer(state))
        return state


graph = TutorGraph()


def main():
    print("Type 'exit' to quit.")
    while True:
        user = input("> ").strip()
        if user.lower() in {"exit", "quit", "q"}:
            break

        state: State = {
            "query": user,
            "route_label": None,
            "final_answer": None,
            "retrieved_context": None,
            "verification_notes": None,
            "explanation": None,
            "needs_hitl": None,
            "verified": None,
            "parsed_problem": None,
            "total_cost": 0.0
        }

        result = graph.invoke(state)
        print(f"\n📊 Topic: {result['route_label']}")
        print(f"🔎 Parsed: {result['parsed_problem']}")
        print(f"✅ Verified: {result['verified']} — {result['verification_notes']}")
        if result["needs_hitl"]:
            print("⚠️  HITL triggered — verifier was not confident")
        print(f"\n📝 Answer:\n{result['final_answer']}")
        print(f"\n🎓 Explanation:\n{result['explanation']}\n")


if __name__ == "__main__":
    main()
