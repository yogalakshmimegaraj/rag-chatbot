"""Core RAG logic: load PDFs -> chunk -> embed -> store -> retrieve -> answer."""

import os
from pathlib import Path
from dataclasses import dataclass

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

# Load environment variables from project directory
load_dotenv(Path(__file__).resolve().parent / ".env")

# Validate API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your-key-here":
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add your API key to the .env file."
    )

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # shared characters between neighbouring chunks
TOP_K = 4              # how many chunks we give the LLM

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the provided documents."
Do not use outside knowledge. Be concise.

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class Chunk:
    text: str
    source: str   # file name
    page: int     # 1-based page number


# ---------- 1. Load and chunk ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())          # collapse whitespace
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def load_pdf(file, name: str) -> list[Chunk]:
    """`file` can be a path or a file-like object (Streamlit upload)."""
    reader = PdfReader(file)
    out = []
    for page_no, page in enumerate(reader.pages, start=1):
        for piece in chunk_text(page.extract_text() or ""):
            out.append(Chunk(piece, name, page_no))
    return out


# ---------- 2. Embeddings (Gemini) ----------
def _client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed(texts: list[str], task: str) -> list[list[float]]:
    """task: 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for questions."""
    from google.genai import types
    client, vectors = _client(), []
    for i in range(0, len(texts), 90):                 # API accepts limited batch sizes
        res = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 90],
            config=types.EmbedContentConfig(task_type=task),
        )
        vectors += [e.values for e in res.embeddings]
    return vectors


# ---------- 3. Vector store (ChromaDB) ----------
class RagStore:
    def __init__(self, path="chroma_db", embed_fn=embed, name="docs"):
        self.embed_fn = embed_fn
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"})

    def reset(self):
        ids = self.col.get()["ids"]
        if ids:
            self.col.delete(ids=ids)

    def add(self, chunks: list[Chunk]):
        if not chunks:
            return
        base = self.col.count()
        self.col.add(
            ids=[f"c{base + i}" for i in range(len(chunks))],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
            embeddings=self.embed_fn([c.text for c in chunks], "RETRIEVAL_DOCUMENT"),
        )

    def search(self, question: str, k: int = TOP_K):
        if self.col.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embed_fn([question], "RETRIEVAL_QUERY"),
            n_results=min(k, self.col.count()),
        )
        return [
            {"text": d, "source": m["source"], "page": m["page"], "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


# ---------- 4. Generate the answer ----------
def generate(prompt: str) -> str:
    res = _client().models.generate_content(model=CHAT_MODEL, contents=prompt)
    return (res.text or "").strip()


def answer(store: RagStore, question: str, llm=generate):
    hits = store.search(question)
    if not hits:
        return "Please upload and index a document first.", []
    context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)
    return llm(PROMPT.format(context=context, question=question)), hits
