from fastapi import FastAPI
from embedding import embed
from vector_store import add
from rag import generate_answer
from pydantic import BaseModel

app = FastAPI()

# 초기 문서 로딩
def load_data():
    with open("/data/data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+300] for i in range(0, len(text), 300)]
    embeddings = embed(chunks)

    add(chunks, embeddings)

load_data()

@app.get("/")
def root():
    return {"message": "DevDoc AI Running"}

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(req: QueryRequest):
    answer = generate_answer(req.query)
    return {"answer": answer}