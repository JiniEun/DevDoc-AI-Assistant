import json
import logging
from collections.abc import Generator

import requests

from config import settings
from embedding import embed
from vector_store import search

logger = logging.getLogger(__name__)


def _build_prompt(query: str, docs: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[출처: {d['filename']}]\n{d['text']}" for d in docs
    )
    return f"""다음 문서들을 기반으로만 질문에 답변하세요.
문서에 없는 내용이라면 반드시 "해당 문서에서 찾을 수 없습니다."라고 답하세요.
추측하거나 일반 지식을 사용하지 마세요.

=== 참고 문서 ===
{context}

=== 질문 ===
{query}

=== 답변 ==="""


def generate_answer(query: str, top_k: int = 3) -> tuple[str, list[dict]]:
    query_vec = embed([query])[0]
    docs = search(query_vec, k=top_k)

    if not docs:
        return "등록된 문서가 없거나 관련 내용을 찾을 수 없습니다.", []

    prompt = _build_prompt(query, docs)
    try:
        res = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        res.raise_for_status()
        return res.json()["response"], docs
    except requests.RequestException as e:
        logger.error("LLM request failed: %s", e)
        raise RuntimeError(f"LLM 요청 실패: {e}") from e


def stream_answer(query: str, top_k: int = 3) -> Generator[str, None, None]:
    query_vec = embed([query])[0]
    docs = search(query_vec, k=top_k)

    if not docs:
        yield f"data: {json.dumps({'type': 'token', 'content': '등록된 문서가 없거나 관련 내용을 찾을 수 없습니다.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
        return

    sources = [
        {"filename": d["filename"], "text": d["text"][:200] + ("..." if len(d["text"]) > 200 else "")}
        for d in docs
    ]
    prompt = _build_prompt(query, docs)

    try:
        with requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        ) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                if data.get("done"):
                    yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"
                    return
    except requests.RequestException as e:
        logger.error("LLM streaming failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
