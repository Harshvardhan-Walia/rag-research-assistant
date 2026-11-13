# ingest.py — loads PDFs, splits them, embeds them, stores in Chroma
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PAPERS_DIR = "papers"
DB_DIR = "chroma_db"

def main():
    # 1. Load all PDFs from the papers folder
    docs = []
    pdf_files = list(Path(PAPERS_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s).")

    for pdf in pdf_files:
        print(f"  Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} pages total.")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embed and store in Chroma
    print("Embedding chunks (this uses the GPU, may take a minute)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"Done. Vector database saved to '{DB_DIR}'.")

if __name__ == "__main__":
    main()
