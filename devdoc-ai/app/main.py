import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from document import chunk_text, extract_text_from_pdf, extract_text_from_txt
from embedding import embed
from models import AskResponse, DocumentInfo, QueryRequest, SourceChunk, UploadResponse
from rag import generate_answer, stream_answer
from vector_store import add_document, get_documents, load, remove_document, total_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DevDoc AI Assistant",
    description="문서 기반 AI 질의응답 시스템",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

_static = Path("/static")
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    load()
    _ensure_model()
    logger.info("DevDoc AI Assistant v2.0 ready")


def _ensure_model() -> None:
    """Pull the LLM model from Ollama if not already available."""
    try:
        res = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        if res.status_code != 200:
            return
        names = [m["name"] for m in res.json().get("models", [])]
        if not any(n.startswith(settings.ollama_model) for n in names):
            logger.info("Pulling model %s from Ollama...", settings.ollama_model)
            requests.post(
                f"{settings.ollama_url}/api/pull",
                json={"name": settings.ollama_model},
                timeout=600,
            )
            logger.info("Model %s ready", settings.ollama_model)
    except Exception as e:
        logger.warning("Could not verify Ollama model: %s", e)


def _check_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status": "ok",
        "version": "2.0.0",
        "documents": len(get_documents()),
        "chunks": total_chunks(),
    }


# ─── Web UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    html_file = Path("/static/index.html")
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DevDoc AI Assistant</h1><p>Static files not found.</p>")


# ─── Upload ───────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload(
    file: UploadFile = File(...),
    _: None = Depends(_check_api_key),
) -> UploadResponse:
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "txt"):
        raise HTTPException(status_code=400, detail="PDF 또는 TXT 파일만 지원합니다.")

    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기는 {settings.max_file_size_mb}MB를 초과할 수 없습니다.",
        )

    try:
        text = extract_text_from_pdf(content) if ext == "pdf" else extract_text_from_txt(content)
    except Exception as e:
        logger.error("File processing error for %s: %s", filename, e)
        raise HTTPException(status_code=422, detail=f"파일 처리 실패: {e}") from e

    if not text.strip():
        raise HTTPException(status_code=422, detail="파일에서 텍스트를 추출할 수 없습니다.")

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="텍스트 청크 생성에 실패했습니다.")

    embeddings = embed(chunks)
    doc_id = str(uuid.uuid4())
    upload_time = datetime.now().isoformat()

    add_document(doc_id, filename, upload_time, chunks, embeddings)
    logger.info("Uploaded %s → %d chunks (doc_id=%s)", filename, len(chunks), doc_id)

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunk_count=len(chunks),
        message="문서가 성공적으로 업로드되었습니다.",
    )


# ─── Documents ────────────────────────────────────────────────────────────────

@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
def list_docs(_: None = Depends(_check_api_key)) -> list[dict]:
    return get_documents()


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_doc(doc_id: str, _: None = Depends(_check_api_key)) -> dict:
    if not remove_document(doc_id, embed):
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    logger.info("Deleted document doc_id=%s", doc_id)
    return {"message": "문서가 삭제되었습니다.", "doc_id": doc_id}


# ─── Ask ──────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["QA"])
def ask(req: QueryRequest, _: None = Depends(_check_api_key)) -> AskResponse:
    try:
        answer, docs = generate_answer(req.query, req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return AskResponse(
        answer=answer,
        sources=[SourceChunk(filename=d["filename"], text=d["text"][:200]) for d in docs],
    )


@app.post("/ask/stream", tags=["QA"])
def ask_stream(req: QueryRequest, _: None = Depends(_check_api_key)) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(req.query, req.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
