import os
import logging
import PyPDF2
import docx
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import POLICY_DOCUMENT_PATH
from utils import save_processed_data, load_processed_data

logger = logging.getLogger(__name__)

document_chunks = []
vectorizer = None
chunk_vectors = None
document_status = {"loaded": False, "filename": "", "chunks": 0, "error": None}

def extract_text_from_file(file_path: str) -> str:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == '.txt':
        return file_path.read_text(encoding='utf-8')

    elif file_path.suffix.lower() == '.pdf':
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)

    elif file_path.suffix.lower() in ['.doc', '.docx']:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end].strip())
        if end == text_length:
            break
        start = end - overlap
    return chunks

def load_and_process_document():
    global document_chunks, vectorizer, chunk_vectors, document_status
    try:
        loaded_chunks, loaded_status, loaded_vectorizer, loaded_vectors = load_processed_data()
        if loaded_chunks is not None:
            document_chunks, document_status, vectorizer, chunk_vectors = (
                loaded_chunks, loaded_status, loaded_vectorizer, loaded_vectors
            )
            logger.info("Using previously processed document data")
            return

        if not os.path.exists(POLICY_DOCUMENT_PATH):
            error_msg = f"Policy document not found at: {POLICY_DOCUMENT_PATH}"
            logger.error(error_msg)
            document_status.update({"loaded": False, "error": error_msg})
            return

        text = extract_text_from_file(POLICY_DOCUMENT_PATH)
        document_chunks = chunk_text(text)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        chunk_vectors = vectorizer.fit_transform(document_chunks)

        document_status.update({
            "loaded": True,
            "filename": os.path.basename(POLICY_DOCUMENT_PATH),
            "chunks": len(document_chunks),
            "error": None
        })
        save_processed_data(document_chunks, document_status, vectorizer, chunk_vectors)
        logger.info("Successfully loaded and processed document")

    except Exception as e:
        error_msg = f"Error loading document: {str(e)}"
        logger.error(error_msg)
        document_status.update({"loaded": False, "error": error_msg})

def find_relevant_chunks(query: str, top_k: int = 3):
    if vectorizer is None or chunk_vectors is None:
        return [], []
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [document_chunks[i] for i in top_indices], [similarities[i] for i in top_indices]
