from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import os

load_dotenv()

mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["math_mentor"]
collection = db["solved_problems"]

def save_to_memory(state: dict, feedback: str = "correct"):
    doc = {
        "timestamp": datetime.utcnow(),
        "original_query": state["query"],
        "parsed_problem": state["parsed_problem"],
        "retrieved_context": state["retrieved_context"],
        "final_answer": state["final_answer"],
        "verified": state["verified"],
        "verification_notes": state["verification_notes"],
        "feedback": feedback
    }
    result=collection.insert_one(doc)
    print(f"Saved to MongoDB: {result.inserted_id}")  

def get_similar_problems(query: str, limit: int = 3) -> list:
    try:
        count = collection.count_documents({})
        print(f"Total docs in collection: {count}") 
        results = collection.find(
            {"feedback": "correct"},
            limit=limit
        ).sort("timestamp", -1)  
        return list(results)
    except Exception as e:
        print(f"Memory error: {e}")
        return []