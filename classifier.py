
import json
from dotenv import load_dotenv
from anthropic import Anthropic
import os

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_db = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)
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
    results = vector_db.similarity_search(
        query=query,
        k=3,
        filter={"topic": topic})
    if not results:
        return "No relevant formulas found."
    return "\n\n".join([r.page_content for r in results])


def calculate_cost(response, model: str) -> float:
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    if "sonnet" in model:
        cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
    else:  
        cost = (input_tokens * 0.000001) + (output_tokens * 0.000005)
    return cost

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

    response = client.messages.create(
        
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Parse this math problem: {query}"}]
    )
    cost = calculate_cost(response, "haiku")
    
    raw = response.content[0].text.strip()
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
        "total_cost": cost
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
   
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  
        max_tokens=50,
        temperature=0,
        system=SYSTEM_PROMPT,
         messages=[
            {
                "role": "user",
                "content": f"Question: {query}"
            }])
    label = response.content[0].text.strip().upper()
    cost = calculate_cost(response, "haiku")
    existing_cost = state.get("total_cost") or 0.0
    return {"route_label": label, "total_cost": existing_cost + cost}
    


def route_bot(state:State)-> Literal["algebra_bot", "calculus_bot","prob_bot"]:
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
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",  
        max_tokens=2000,
        system=SYSTEM_PROMPT,
         messages=[
            {
                "role": "user",
                "content": f"Relevant formulas:\n{context}\n\nQuestion: {query}"
            }])
    cost = calculate_cost(response, "sonnet")
    existing_cost = state.get("total_cost") or 0
    return {
    "final_answer": response.content[0].text,
    "retrieved_context": context,
    "total_cost": existing_cost + cost
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
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  
        max_tokens=2000,
        system=SYSTEM_PROMPT,
         messages=[
            {
                "role": "user",
                "content": f"Relevant formulas:\n{context}\n\nQuestion: {query}"
            }])
    

    cost = calculate_cost(response, "haiku")
    existing_cost = state.get("total_cost") or 0.0
    return {
    "final_answer": response.content[0].text,
    "retrieved_context": context,
    "total_cost": existing_cost + cost
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
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  
        max_tokens=2000,
        system=SYSTEM_PROMPT,
         messages=[
            {
                "role": "user",
                "content": f"Relevant formulas:\n{context}\n\nQuestion: {query}"
            }])
    cost = calculate_cost(response, "haiku")
    existing_cost = state.get("total_cost") or 0.0
    return {
    "final_answer": response.content[0].text,
    "retrieved_context": context,
    "total_cost": existing_cost + cost
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

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Formulas:\n{context}\n\nQuestion: {query}\n\nAnswer:\n{answer}"}]
    )
    raw = response.content[0].text.strip()
    verdict, notes = "CONFIDENT", ""
    for line in raw.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().upper()
        elif line.startswith("NOTES:"):
            notes = line.replace("NOTES:", "").strip()
    cost = calculate_cost(response, "haiku")
    existing_cost = state.get("total_cost") or 0.0
    return {
    "verified": verdict == "CONFIDENT",
    "verification_notes": notes,
    "needs_hitl": verdict != "CONFIDENT",
    "total_cost": existing_cost + cost
}


def explainer(state: State):
    query = state["query"]
    answer = state["final_answer"]
    context = state["retrieved_context"]
    SYSTEM_PROMPT = """You are a friendly JEE tutor explaining a solution to a student.
Do NOT re-solve. Explain WHY each step works in simple language.
Use numbered steps, mention which formula was used and why.
Highlight common mistakes to avoid. End with a one-line takeaway."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Formulas:\n{context}\n\nQuestion: {query}\n\nSolution:\n{answer}\n\nExplain to a JEE student."}]
    )
    cost = calculate_cost(response, "haiku")
    existing_cost = state.get("total_cost") or 0.0
    return {
        "explanation": response.content[0].text,
        "total_cost": existing_cost + cost
    }

graph_builder = StateGraph(State)
graph_builder.add_node("parser_agent", parser_agent)
graph_builder.add_node("classify_msg", classify_msg)
graph_builder.add_node("algebra_bot", algebra_bot)
graph_builder.add_node("calculus_bot", calculus_bot)
graph_builder.add_node("prob_bot", prob_bot)
graph_builder.add_node("verifier", verifier)
graph_builder.add_node("explainer", explainer)

graph_builder.add_edge(START, "parser_agent")
graph_builder.add_edge("parser_agent", "classify_msg")
graph_builder.add_conditional_edges("classify_msg", route_bot)
graph_builder.add_edge("algebra_bot", "verifier")
graph_builder.add_edge("calculus_bot", "verifier")
graph_builder.add_edge("prob_bot", "verifier")
graph_builder.add_edge("verifier", "explainer")
graph_builder.add_edge("explainer", END)

graph = graph_builder.compile()


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