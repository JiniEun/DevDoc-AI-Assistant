from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    upload_time: str
    chunk_count: int


class SourceChunk(BaseModel):
    filename: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
