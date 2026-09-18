from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import os

load_dotenv()


def get_collection():
    """Return the memory collection only when MongoDB is reachable.

    Memory should enrich the tutor, not prevent the entire application from
    starting when the managed database is paused or its connection string has
    changed.
    """
    mongo_url = os.getenv("MONGODB_URL")
    if not mongo_url:
        return None

    try:
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5_000)
        client.admin.command("ping")
        return client["math_mentor"]["solved_problems"]
    except Exception as exc:
        print(f"Memory unavailable: {exc}")
        return None

def save_to_memory(state: dict, feedback: str = "correct"):
    collection = get_collection()
    if collection is None:
        return False

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
    return True

def get_similar_problems(query: str, limit: int = 3) -> list:
    try:
        collection = get_collection()
        if collection is None:
            return []
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
