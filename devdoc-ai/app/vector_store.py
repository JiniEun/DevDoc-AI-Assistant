import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)

documents = []

def add(chunks, embeddings):
    index.add(np.array(embeddings).astype("float32"))
    documents.extend(chunks)

def search(query_embedding, k=3):
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [documents[i] for i in I[0]]