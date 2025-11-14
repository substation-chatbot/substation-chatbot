import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def load_documents():
    docs = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                docs.extend(loader.load())
    return docs

def create_vector_store():
    print("🔄 Loading documents from:", DATA_PATH)
    docs = load_documents()
    print(f"✅ Loaded {len(docs)} documents")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vectordb.persist()
    print("✅ Vector store created successfully at:", CHROMA_PATH)

if __name__ == "__main__":
    create_vector_store()
