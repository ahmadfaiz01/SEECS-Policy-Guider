# 🎓 SEECS Policy Guider: Scalable Academic QA System

Hey there! Welcome to the **SEECS Policy Guider**. This project is a custom-built, Retrieval-Augmented Generation (RAG) system designed to answer student policy questions quickly and accurately. Instead of relying on a generic chatbot that hallucinates rules, this system is strictly grounded in the official NUST SEECS Undergraduate and Postgraduate handbooks. 

If it's in the handbook, this system will find it. If it's not, it won't make it up!

---

## 🏗️ How It Works (The Architecture)

We built this entirely from scratch, focusing on high-performance retrieval algorithms rather than just throwing everything into a vector database.

1. **Document Ingestion:** We parse the raw PDF handbooks, clean the text, and chunk them into meaningful, easily digestible paragraphs (roughly 150 words each).
2. **The Retrieval Engine:** This is the core of the project. We implemented three distinct search algorithms to test speed vs. accuracy:
   * **TF-IDF (The Exact Baseline):** Does a full mathematical comparison of the query against every chunk. It's accurate but computationally heavy.
   * **MinHash LSH:** An approximate retrieval method. We generate "shingles" of the text, compress them into signatures, and use Locality Sensitive Hashing to group similar chunks into buckets. This makes searching incredibly fast!
   * **SimHash:** Another approximate method using 64-bit fingerprints and Hamming distance to find overlaps.
3. **Knowledge Graph (PageRank):** Policies reference each other (e.g., *"Subject to rules in Clause 4"*). We built a network graph of these cross-references and ran the **PageRank** algorithm to find the most "authoritative" sections of the handbook. This acts as a smart tie-breaker to boost highly-referenced policies during a search.
4. **Answer Synthesis:** Once we fetch the top 5 most relevant excerpts, we send *only those excerpts* to a massive LLM (Llama 3 70B via Groq) to synthesize a clean, natural-language answer for the student.

---

## 🚀 Running the Project Locally

Want to spin this up on your own machine? It's super easy.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API Key:**
   Rename `.env.example` to `.env` and paste in your Groq API key (this is required for the answer generation phase).

3. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *Note: If you haven't built the indexes yet, the app will automatically build them from the PDFs on its very first startup!*

---

## 🧪 Experiments & Benchmarks

If you want to see the raw data proving that our approximate retrieval (MinHash/SimHash) is faster than the exact baseline, check out the `experiments/` folder. Running `python experiments/benchmark.py` will generate automated latency, memory, and precision tests that you can view directly in the "Performance Analytics" tab of the UI!
