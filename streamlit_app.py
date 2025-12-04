import os
import io
import threading
import time
from typing import List

import streamlit as st
import requests
import chromadb

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument

from google.genai import Client
from chromadb.utils.embedding_functions import EmbeddingFunction
import uvicorn

# --------------------------------------------------
# ENV + GEMINI
# --------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your .env")

client = Client(api_key=API_KEY)

# --------------------------------------------------
# CHROMADB
# --------------------------------------------------
chroma_client = chromadb.Client()
COLLECTION_NAME = "rag_docs"

try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(COLLECTION_NAME)


class GeminiEmbedding(EmbeddingFunction):
    def __call__(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            resp = client.models.embed_content(
                model="text-embedding-004",
                contents=text,
            )
            vectors.append(resp.embeddings[0].values)
        return vectors


embedding_fn = GeminiEmbedding()

# --------------------------------------------------
# CHUNKING
# --------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[str]:
    text = text.replace("\n", " ").strip()
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# --------------------------------------------------
# FILE EXTRACTORS
# --------------------------------------------------
def extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_docx(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

# --------------------------------------------------
# FASTAPI BACKEND
# --------------------------------------------------
app = FastAPI(title="RAG API")

class Question(BaseModel):
    query: str
    k: int = 3


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename.lower()
    data = await file.read()

    if name.endswith(".pdf"):
        text = extract_pdf(data)
    elif name.endswith(".docx"):
        text = extract_docx(data)
    else:
        text = data.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "No text extracted")

    # Chunking
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No chunks created")

    embeddings = embedding_fn(chunks)

    start_id = collection.count()
    ids = [str(start_id + i + 1) for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"filename": file.filename} for _ in chunks],
    )

    return {
        "chunks_added": len(chunks),
        "total_chunks": collection.count(),
    }


@app.post("/ask")
async def ask(q: Question):
    q_emb = embedding_fn([q.query])

    results = collection.query(
        query_embeddings=q_emb,
        n_results=q.k,
        include=["documents", "metadatas"],
    )

    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"[source: {meta.get('filename')}]\n{doc}\n\n"

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

CONTEXT:
{context}

QUESTION:
{q.query}
"""

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return {"answer": resp.text}

# --------------------------------------------------
# FASTAPI THREAD
# --------------------------------------------------
def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if "api_started" not in st.session_state:
    st.session_state.api_started = True
    threading.Thread(target=run_api, daemon=True).start()
    time.sleep(1)

# --------------------------------------------------
# STREAMLIT FRONTEND
# --------------------------------------------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("RAG Chat — Gemini + ChromaDB")

st.sidebar.header("Upload Document")
uploaded = st.sidebar.file_uploader(
    "PDF, DOCX, or TXT",
    type=["pdf", "docx", "txt"],
)

if uploaded:
    with st.spinner("Uploading..."):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        r = requests.post(f"{API_URL}/upload", files=files)
        if r.ok:
            st.sidebar.success("Uploaded successfully")
        else:
            st.sidebar.error(r.text)

st.sidebar.markdown("---")

st.subheader("Ask a question")
query = st.text_area("Question", height=120)
k = 3

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            r = requests.post(
                f"{API_URL}/ask",
                json={"query": query, "k": k},
            )
            st.markdown("### Answer")
            st.write(r.json()["answer"])

st.markdown("---")
