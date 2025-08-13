from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class DocumentStatus(BaseModel):
    loaded: bool
    filename: str
    chunks: int
    error: str = None
