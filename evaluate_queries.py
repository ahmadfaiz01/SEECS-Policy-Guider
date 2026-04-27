import sys
from pathlib import Path
import json

ROOT = Path("/Users/husnain/Labs/Big data/SEECS-Policy-Guider-ahmad")
sys.path.insert(0, str(ROOT / "src"))

from retrieval.pipeline import RetrievalPipeline
from generation.answer_gen import AnswerGenerator
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

queries = [
    "What is the minimum CGPA requirement for graduation?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is the grading scale at SEECS?",
    "What are the requirements for the Dean's Honor List?",
    "What is the maximum course load per semester?",
    "What is the policy on academic probation?",
    "How are final grades calculated?",
    "What is the procedure for dropping a course?",
    "What is the internship requirement for graduation?",
    "What are the thesis requirements for MS students?",
    "What is the policy on plagiarism?",
    "How can a student appeal a grade?",
    "What are the requirements for changing a major?",
]

def main():
    print("Loading pipeline...")
    pipeline = RetrievalPipeline(str(ROOT / "data" / "indexes"), str(ROOT / "data" / "processed" / "chunks.jsonl"))
    print("Loading generator...")
    generator = AnswerGenerator()

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/15] QUERY: {q}")
        all_results = pipeline.query_all(q, top_k=5)
        results = all_results["hybrid_lsh"].chunks
        
        # Format the top chunks
        formatted_chunks = []
        for chunk in results:
            formatted_chunks.append({
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "start_page": chunk.get("start_page", ""),
                "end_page": chunk.get("end_page", ""),
                "section": chunk.get("section", "")
            })
            
        ans = generator.generate(q, formatted_chunks)
        print("ANSWER:")
        print(ans.get("answer", "NO ANSWER"))

if __name__ == "__main__":
    main()
