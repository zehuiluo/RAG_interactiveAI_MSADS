"""
embeddings.py — Text chunking + embedding generation for the MSADS RAG system.
"""
import json, pathlib
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

CHUNK_SIZE   = 512
CHUNK_OVERLAP = 64
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            end = boundary if boundary > start else end
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def build_chunks(documents: List[Dict]) -> List[Dict]:
    all_chunks = []
    # Use a global counter so IDs are always unique regardless of URL
    global_idx = 0
    for doc in documents:
        raw_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id": f"chunk_{global_idx:04d}",   # unique: chunk_0000, chunk_0001, ...
                "url":      doc["url"],
                "title":    doc["title"],
                "text":     chunk,
            })
            global_idx += 1
    print(f"✅  Created {len(all_chunks)} chunks from {len(documents)} pages.")
    return all_chunks


def generate_embeddings(chunks: List[Dict]) -> Tuple[List[Dict], List]:
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks…")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print("✅  Embeddings generated.")
    return chunks, embeddings.tolist()


if __name__ == "__main__":
    raw = json.loads(pathlib.Path("data/knowledge_base.json").read_text())
    chunks = build_chunks(raw)
    pathlib.Path("data/chunks.json").write_text(json.dumps(chunks, indent=2))
    print(f"Chunks saved to data/chunks.json")
