# main.py
import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from pypdf import PdfReader
from docx import Document as DocxDocument
from dotenv import load_dotenv

from google.genai import Client
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment or .env file")

# Gemini client
client = Client(api_key=API_KEY)

# ChromaDB setup (in-memory by default)
chroma_client = chromadb.Client()
collection_name = "rag_docs"
# If collection already exists, use it, otherwise create:
try:
    collection = chroma_client.get_collection(name=collection_name)
except Exception:
    collection = chroma_client.create_collection(name=collection_name)

# Embedding wrapper for Gemini
class GeminiEmbedding(EmbeddingFunction):
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # note: use 'contents' param and extract resp.embeddings[0].values
            resp = client.models.embed_content(model="text-embedding-004", contents=text)
            vec = resp.embeddings[0].values
            embeddings.append(vec)
        return embeddings

embedding_fn = GeminiEmbedding()

app = FastAPI(title="RAG Gemini + ChromaDB")

class Question(BaseModel):
    query: str
    k: int = 3  # optional: number of retrieved docs


def extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def extract_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])



@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    file_bytes = await file.read()

    # extract text depending on extension
    if filename.endswith(".pdf"):
        text = extract_pdf(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_docx(file_bytes)
    elif filename.endswith(".doc"):
        raise HTTPException(status_code=400, detail="Please convert .doc to .docx first")
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file")

    new_id = str(collection.count() + 1)
    docs = [text]
    ids = [new_id]
    metadatas = [{"filename": file.filename}]
    embeddings = embedding_fn(docs)

    # Try modern signature first (metadatas), fallback to older/other behavior
    try:
        collection.add(
            documents=docs,
            ids=ids,
            metadatas=metadatas,    # <- modern Chroma expects 'metadatas' (plural)
            embeddings=embeddings,
        )
    except TypeError as e:
        # If metadatas is not supported, try without it
        if "metadatas" in str(e) or "metadata" in str(e) or "unexpected keyword" in str(e):
            try:
                collection.add(
                    documents=docs,
                    ids=ids,
                    embeddings=embeddings,
                )
            except Exception as e2:
                # If that also fails, raise an HTTP 500 with the error message
                raise HTTPException(status_code=500, detail=f"Chroma add failed: {e2}")
        else:
            # some other TypeError — re-raise
            raise HTTPException(status_code=500, detail=f"Chroma add failed: {e}")

    return {"status": "uploaded", "id": new_id, "total_docs": collection.count()}

def retrieve_context(query: str, k: int = 3):
    # get embeddings for the query
    q_embs = embedding_fn([query])
    
    results = collection.query(
        query_embeddings=q_embs,
        n_results=k,
        include=["documents", "metadatas", "distances"]  # ids removed
    )

    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]           # still accessible
    metas = results.get("metadatas", [[]])[0]

    context_parts = []

    for doc, docid, meta in zip(docs, ids, metas):
        filename = meta.get("filename") if isinstance(meta, dict) else None
        header = f"[source_id={docid}"
        if filename:
            header += f" filename={filename}"
        header += "]"
        context_parts.append(header + "\n" + doc)

    context = "\n\n---\n\n".join(context_parts)
    return context, results



@app.post("/ask")
async def ask_question(payload: Question):
    query = payload.query
    k = payload.k or 3
    context, raw_results = retrieve_context(query, k=k)
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question. If the context does not contain the answer, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

Answer concisely and include the source ids for any factual claims."""
    # generate
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    # We'll return the raw context and the response, plus the retrieved metadata for reference
    return {
        "answer": resp.text
    }
