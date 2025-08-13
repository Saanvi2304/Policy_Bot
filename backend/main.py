from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
import logging
import os
import shutil

from config import ALLOWED_ORIGINS, DATA_DIR
from models import QueryRequest
from document_processor import load_and_process_document, find_relevant_chunks, document_status
from document_processor import document_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama Policy QA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    load_and_process_document()

@app.get("/")
async def root():
    return {"message": "Llama Policy QA API is running", "version": "1.0.0"}

@app.get("/document-status")
async def get_document_status():
    return document_status

@app.post("/reload-document")
async def reload_document():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    load_and_process_document()
    return document_status

@app.post("/query")
async def query_document(request: QueryRequest):
    if not document_status["loaded"]:
        if document_status["error"]:
            raise HTTPException(status_code=500, detail=f"Document loading failed: {document_status['error']}")
        raise HTTPException(status_code=400, detail="No document loaded yet")

    relevant_chunks, similarities = find_relevant_chunks(request.question)
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant information found")

    context = "\n\n".join(relevant_chunks)
    prompt = f"""Based ONLY on the following policy document context, answer the user's question. 
If the answer cannot be found, say "I cannot find this information in the provided policy document."

Context:
{context}

Question: {request.question}

Answer:"""

    response = ollama.chat(
        model='gemma3:4b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1, 'top_p': 0.9}
    )
    answer = response['message']['content']
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    confidence = max(0.0, min(1.0, avg_similarity))

    return {"answer": answer, "sources": [document_status["filename"]], "confidence": confidence}

@app.get("/health")
async def health_check():
    try:
        ollama.list()
        return {"status": "healthy", "llama_available": True, "document_loaded": document_status["loaded"]}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "llama_available": False, "document_loaded": document_status["loaded"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
