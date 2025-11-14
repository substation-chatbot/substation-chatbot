import os
import langchain
if not hasattr(langchain, "verbose"):
    langchain.verbose = False  # temporary patch for Google GenAI compatibility

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

# ===================================================
# Load environment variables

BASE_DIR = os.path.dirname(__file__)
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(f"❌ Missing GOOGLE_API_KEY in .env file. Tried loading from: {env_path}")

# ===================================================
# Define paths

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# ===================================================
# Embedding model

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ===================================================
# Chroma Vector DB

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ===================================================
# Initialize Gemini LLM

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.4)

# ===================================================
# Load and Split Data

def load_documents():
    """Load all files and embed them into Chroma DB."""
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load_and_split()
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
            docs = loader.load()
        else:
            continue
        chunks = splitter.split_documents(docs)
        documents.extend(chunks)

    db.add_documents(documents)
    print(f"✅ Indexed {len(documents)} chunks.")

# ===================================================
# RAG Answer Function (Dynamic Reasoning Mode)

def get_answer(query: str, force_mode: str = None):
    """
    Retrieve context and ask the LLM.
    Automatically determine the mode (Info / Diagnostic / Hybrid)
    based on question phrasing and context.
    Only respond to queries related to substations, transformers, breakers,
    relays, protection systems, and related electrical equipment.
    """

    # === Retrieve contextual documents ===
    try:
        docs = retriever.invoke(query)
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            docs = []

    context_chunks: List[str] = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        if text:
            context_chunks.append(text.strip())

    context = "\n\n".join(context_chunks)

    # === Domain-limited reasoning ===
    prompt = f"""
You are a Substation Maintenance Expert AI assistant.

Your domain is strictly limited to:
- Substations and electrical systems (transformers, breakers, relays, busbars, isolators, CTs, PTs, switchgear, protection, SCADA, etc.)
- Maintenance, diagnostics, safety procedures, and testing methods for such equipment.

If the user's question is **not related** to substations, electrical maintenance, or power systems,
DO NOT answer it. Instead, respond exactly with this sentence:
"I’m a Substation Maintenance Expert and can only assist with substation or electrical equipment-related queries."

When the question is relevant:
- Identify the intent internally (no need to show it).
- Provide the best answer possible using the given context.
- If the context lacks sufficient data, say you're using general engineering knowledge.

Context:
{context if context else "[No or limited context available]"}

User Question: {query}

Answer directly below:
"""

    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
        text = text.strip()

        # === Filter irrelevant response ===
        irrelevant_msg = "I’m a Substation Maintenance Expert and can only assist with substation or electrical equipment-related queries."
        if irrelevant_msg.lower() in text.lower():
            # For irrelevant questions, no context info
            return irrelevant_msg

        # === For valid questions, return clean output (no mode, no debug info) ===
        context_len = len(context)
        return f"(RetrievedContextChars: {context_len})\n\n{text}"

    except Exception as e:
        return f"⚠️ Error while generating response: {e}"


# ===================================================
# Local Test

if __name__ == "__main__":
    q = "The transformer oil temperature is rising abnormally. What could be the reason?"
    print("Question:", q)
    print("Answer:", get_answer(q))

# ===================================================
# Utility: Add New Documents

def load_and_add_document(file_path: str):
    """Add a new PDF or TXT document to Chroma DB."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs = [Document(page_content=text)]
    else:
        raise ValueError("Unsupported file type. Please upload PDF or TXT files only.")

    chunks = text_splitter.split_documents(docs)
    db.add_documents(chunks)
    print(f"✅ Added {len(chunks)} chunks from {os.path.basename(file_path)}")
