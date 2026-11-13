# app.py — serves the RAG tool as a web API using FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
LLM_MODEL = "qwen2.5:7b"

app = FastAPI(title="RAG Research Assistant")

# Load the database and models once at startup (not on every request)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
llm = OllamaLLM(model=LLM_MODEL)

# Define the shape of an incoming request
class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG Research Assistant is running. POST a question to /ask"}

@app.post("/ask")
def ask(query: Query):
    # 1. Retrieve relevant chunks
    results = db.similarity_search(query.question, k=6)
    context = "\n\n".join([d.page_content for d in results])

    # 2. Build the prompt
    prompt = f"""You are a research assistant. Using ONLY the context below from research papers,
identify the key RISKS and the MITIGATIONS proposed for each. Present them as clear
Risk -> Mitigation pairs. If the context does not mention something, do not invent it.

CONTEXT:
{context}

QUESTION: {query.question}

ANSWER:"""

    # 3. Generate the answer
    answer = llm.invoke(prompt)

    # 4. Return it, with the sources used
    sources = [d.metadata.get("source", "?") for d in results]
    return {"question": query.question, "answer": answer, "sources": sources}
