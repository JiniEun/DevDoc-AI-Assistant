import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, show_progress_bar=False)
