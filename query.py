# query.py — retrieves relevant chunks and has the LLM extract risks & mitigations
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
LLM_MODEL = "qwen2.5:7b"   # build model; swap to "qwen2.5:7b" later for quality

def main():
    # 1. Load the existing vector database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # 2. The question we want answered from the papers
    question = (
        "What are the main risks of using AI in medical devices, "
        "and what mitigations do the papers propose for them?"
    )

    # 3. Retrieve the most relevant chunks
    print("Retrieving relevant passages...")
    results = db.similarity_search(question, k=6)
    context = "\n\n".join([f"[Source: {d.metadata.get('source','?')}]\n{d.page_content}" for d in results])

    # 4. Build the prompt and ask the LLM
    prompt = f"""You are a research assistant. Using ONLY the context below from research papers,
identify the key RISKS of AI in medical devices and the MITIGATIONS proposed for each.
Present the answer as a clear list of Risk -> Mitigation pairs. If the context does not
mention something, do not invent it.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    print("Asking the model (uses the GPU)...\n")
    llm = OllamaLLM(model=LLM_MODEL)
    answer = llm.invoke(prompt)

    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()
