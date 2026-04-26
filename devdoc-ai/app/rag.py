import requests
from embedding import embed
from vector_store import search

OLLAMA_URL = "http://ollama:11434/api/generate"

def ask_llm(prompt):
    res = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]

def generate_answer(query):
    query_vec = embed([query])[0]
    docs = search(query_vec)

    context = "\n".join(docs)

    prompt = f"""
    아래 문서를 기반으로만 답변해라.
    모르면 모른다고 해라.

    문서:
    {context}

    질문:
    {query}
    """

    return ask_llm(prompt)