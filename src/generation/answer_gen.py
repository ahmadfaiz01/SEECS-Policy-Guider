"""
answer_gen.py
We are using the Groq API via standard HTTP requests instead of the official library.
Why? Because the official Groq library uses Pydantic, which uses Rust-compiled C-extensions 
that get blocked by strict Windows Security policies. Using 'requests' bypasses this entirely!
"""
import os
<<<<<<< HEAD
from google import genai
from google.genai import types
=======
import requests
>>>>>>> origin/ahmad
from typing import Dict, List, Optional

_SYSTEM_PROMPT = """You are an official academic policy advisor for SEECS, NUST.

Instructions:
1. Provide a direct, clear, and professional answer based strictly on the provided handbook excerpts.
2. Structure your answer using clean markdown. Use bullet points if listing multiple conditions or rules.
<<<<<<< HEAD
3. 
4. Maintain an objective, formal academic tone. 
5.
=======
3. If the excerpts do not contain the answer, state clearly: "The provided handbook excerpts do not contain information regarding this query."
4. Maintain an objective, formal academic tone. 
5. Do not use conversational filler like "Based on the handbook..." or "Here is the answer...". Just state the policy directly.
>>>>>>> origin/ahmad
6. Do NOT manually list the sources at the bottom. The system UI will display the citations separately.
"""


class AnswerGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
<<<<<<< HEAD
        model: str = "gemma-3-27b-it",  
    ):

        self.client = genai.Client()
        self.model_name = model
=======
        model: str = "llama-3.3-70b-versatile",  
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing! Make sure it's in your .env file.")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
>>>>>>> origin/ahmad

    def generate(
        self,
        question: str,
        chunks: List[Dict],
        max_tokens: int = 512,
    ) -> Dict:
        """
<<<<<<< HEAD
        Passes the question and our top chunks to the Gemini LLM.
=======
        Passes the question and our top chunks to the LLM via a REST API call.
>>>>>>> origin/ahmad
        """
        context = self._build_context(chunks)

        user_message = (
<<<<<<< HEAD
            f"{_SYSTEM_PROMPT}\n\n"
=======
>>>>>>> origin/ahmad
            f"Handbook excerpts:\n{context}\n\n"
            f"Student query: {question}\n\n"
            f"Policy Answer:"
        )

<<<<<<< HEAD
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                )
            )
            answer = response.text.strip()
            
            # Log the answer to the terminal
            print("\n" + "="*50)
            print(f"QUERY: {question}")
            print(f"GENERATED ANSWER:\n{answer}")
            print("="*50 + "\n")
=======
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code != 200:
                answer = f"Groq API Error {response.status_code}: {response.text}"
            else:
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
>>>>>>> origin/ahmad
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            
        sources = self._extract_sources(chunks)

<<<<<<< HEAD
        return {"answer": answer, "sources": sources, "model": self.model_name}
=======
        return {"answer": answer, "sources": sources, "model": self.model}
>>>>>>> origin/ahmad

    def _build_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown").replace(".pdf", "")
            start_page = chunk.get("start_page", "?")
            end_page = chunk.get("end_page", start_page)
            section = chunk.get("section", "General").strip()
            
            pages = f"p.{start_page}" if start_page == end_page else f"pp.{start_page}-{end_page}"
            
            parts.append(
                f"[Excerpt {i}] {source} | {section} | {pages}\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        seen = set()
        sources = []
        for chunk in chunks:
            source = chunk.get("source", "Unknown").replace(".pdf", "")
            page = chunk.get("start_page", "?")
            section = chunk.get("section", "General").strip()
            ref = f"{source} - {section} (Pg. {page})"
            if ref not in seen:
                seen.add(ref)
                sources.append(ref)
        return sources
