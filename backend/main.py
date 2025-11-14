from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.rag_engine import get_answer, load_and_add_document

import os
import shutil

app = FastAPI(title="Substation Maintenance Chatbot")

# ======================================================
# Enable CORS for frontend (Streamlit)
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:8501"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# Paths
# Use absolute path for /data folder (safe on Windows)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data"))

os.makedirs(DATA_PATH, exist_ok=True)

# ======================================================
# Routes

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    """Answer user queries using the RAG pipeline."""
    answer = get_answer(query)
    return {"answer": answer}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and embed a new file into the Chroma vector database."""
    try:
        save_path = os.path.join(DATA_PATH, file.filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        load_and_add_document(save_path)
        return {"message": f"✅ File '{file.filename}' uploaded and embedded successfully!"}

    except Exception as e:
        return {"error": f"⚠️ Upload failed: {str(e)}"}
