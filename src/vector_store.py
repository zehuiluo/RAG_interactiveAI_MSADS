import json, pathlib
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "msads_knowledge_base"
PERSIST_DIR     = "data/chroma_db"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 4

class MSADSVectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        pathlib.Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model  = SentenceTransformer(MODEL_NAME)
        self._col   = None

    @property
    def collection(self):
        if self._col is None:
            self._col = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._col

    def reset(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"Deleted old collection.")
        except Exception:
            pass
        self._col = None

    def build_from_chunks(self, chunks_path: str = "data/chunks.json"):
        chunks = json.loads(pathlib.Path(chunks_path).read_text())
        texts  = [c["text"]     for c in chunks]
        ids    = [c["chunk_id"] for c in chunks]
        metas  = [{"url": c["url"], "title": c["title"]} for c in chunks]
        print(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64).tolist()
        for i in range(0, len(ids), 500):
            self.collection.upsert(
                ids=ids[i:i+500],
                documents=texts[i:i+500],
                embeddings=embeddings[i:i+500],
                metadatas=metas[i:i+500],
            )
        print(f"Indexed {len(ids)} chunks into '{COLLECTION_NAME}'.")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        total = self.collection.count()
        if total == 0:
            return []
        k = min(top_k, total)
        query_vec = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text":      doc,
                "url":       meta.get("url", ""),
                "title":     meta.get("title", ""),
                "relevance": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str, default="")
    args = parser.parse_args()

    store = MSADSVectorStore()

    if args.build:
        sys.path.insert(0, "src")
        store.reset()
        chunk_path = pathlib.Path("data/chunks.json")
        if not chunk_path.exists():
            print("Building chunks from data/knowledge_base.json ...")
            from embeddings import build_chunks
            docs   = json.loads(pathlib.Path("data/knowledge_base.json").read_text())
            chunks = build_chunks(docs)
            chunk_path.write_text(json.dumps(chunks, indent=2))
        store.build_from_chunks(str(chunk_path))

    if args.query:
        for p in store.retrieve(args.query):
            print(f"\n[{p['relevance']:.3f}] {p['title']}\n{p['text'][:300]}")
