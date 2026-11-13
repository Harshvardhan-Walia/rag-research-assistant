# RAG Research Assistant

## What it is
A local RAG (Retrieval-Augmented Generation) tool that reads research papers and pulls out the **risks of AI in medical devices** and the **mitigations** the papers propose.

## What it does
1. Reads PDFs and splits them into small chunks
2. Turns each chunk into a vector (`nomic-embed-text`) and stores it in Chroma
3. When you ask a question, finds the 6 most relevant chunks
4. Sends those chunks to a local LLM (`qwen2.5:7b` via Ollama), which answers as **Risk → Mitigation** pairs, with the source papers listed
5. Served as an API with FastAPI (`POST /ask`)

## Why I'm building this
Reviewing literature by hand means reading every paper and pulling out each risk and mitigation manually. This project explores how far RAG can automate that, and it's my hands-on way of learning RAG, from a naive baseline up to more advanced techniques.

## How to run
```bash
ollama pull nomic-embed-text && ollama pull qwen2.5:7b
pip install -r requirements.txt
# put your PDFs in a papers/ folder (not included: copyright)
python ingest.py
uvicorn app:app
```
Then open http://127.0.0.1:8000/docs to ask questions.

## Limitations (v1)
- **Naive RAG:** a basic retrieve-then-answer pipeline with no advanced retrieval techniques.
- **Small local LLM:** a 7B model running on a home GPU; less capable than large hosted models.
- **No API models:** everything is local, so no paid services, but also no access to stronger models.
- **Misses papers:** it only takes the 6 most similar chunks, so relevant papers can be left out (in testing, 1 of 3 papers was never retrieved).
- **No fact-checking:** the model is *told* to use only the papers, but nothing checks that it did. Some risk–mitigation pairs are linked by the model itself.
- **Weak citations:** it lists which papers were retrieved, not which paper each risk came from.

## Future work: v2 (faithful RAG)
**What v2 will do:**
- For every risk and mitigation, check that it **actually exists** in the papers
- Show the **exact quote** and **where it is** (paper + page)
- Only report risks that pass that check

**What changes in the RAG:**
- Each chunk is tagged with its paper and page before it reaches the LLM
- The question is split into sub-questions (e.g. by lifecycle stage) to cover more of the papers
- The LLM returns a quote with each risk, and code verifies the quote against the original text
- Unverified results are dropped instead of shown
