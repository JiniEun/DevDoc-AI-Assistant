import json
import logging
import os
from typing import Callable

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

DIMENSION = 384

_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(DIMENSION)
_chunks: list[dict] = []
_documents: list[dict] = []


def _index_path() -> str:
    return os.path.join(settings.data_dir, "faiss.index")


def _meta_path() -> str:
    return os.path.join(settings.data_dir, "metadata.json")


def _docs_path() -> str:
    return os.path.join(settings.data_dir, "documents.json")


def _save() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    faiss.write_index(_index, _index_path())
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(_chunks, f, ensure_ascii=False)
    with open(_docs_path(), "w", encoding="utf-8") as f:
        json.dump(_documents, f, ensure_ascii=False, indent=2)


def load() -> None:
    global _index
    if not (os.path.exists(_index_path()) and os.path.exists(_meta_path())):
        return
    try:
        _index = faiss.read_index(_index_path())
        with open(_meta_path(), "r", encoding="utf-8") as f:
            _chunks.extend(json.load(f))
        if os.path.exists(_docs_path()):
            with open(_docs_path(), "r", encoding="utf-8") as f:
                _documents.extend(json.load(f))
        logger.info("Loaded %d chunks across %d documents", len(_chunks), len(_documents))
    except Exception as e:
        logger.error("Failed to load vector store: %s", e)


def add_document(
    doc_id: str,
    filename: str,
    upload_time: str,
    text_chunks: list[str],
    embeddings: np.ndarray,
) -> None:
    _index.add(embeddings.astype("float32"))
    for i, chunk in enumerate(text_chunks):
        _chunks.append({"doc_id": doc_id, "filename": filename, "text": chunk, "chunk_idx": i})
    _documents.append(
        {
            "doc_id": doc_id,
            "filename": filename,
            "upload_time": upload_time,
            "chunk_count": len(text_chunks),
        }
    )
    _save()


def remove_document(doc_id: str, embed_fn: Callable[[list[str]], np.ndarray]) -> bool:
    global _index

    if not any(c["doc_id"] == doc_id for c in _chunks):
        return False

    kept = [c for c in _chunks if c["doc_id"] != doc_id]
    _chunks.clear()
    _chunks.extend(kept)
    _documents[:] = [d for d in _documents if d["doc_id"] != doc_id]

    _index = faiss.IndexFlatL2(DIMENSION)
    if _chunks:
        vecs = embed_fn([c["text"] for c in _chunks])
        _index.add(vecs.astype("float32"))

    _save()
    return True


def search(query_vec: np.ndarray, k: int = 3) -> list[dict]:
    if _index.ntotal == 0:
        return []
    k = min(k, _index.ntotal)
    _, I = _index.search(np.array([query_vec]).astype("float32"), k)
    return [_chunks[i] for i in I[0] if 0 <= i < len(_chunks)]


def get_documents() -> list[dict]:
    return list(_documents)


def total_chunks() -> int:
    return len(_chunks)
