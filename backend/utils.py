import os
import pickle
import numpy as np
import logging
from config import DATA_DIR

logger = logging.getLogger(__name__)

def save_processed_data(document_chunks, document_status, vectorizer, chunk_vectors):
    """Save processed data to disk"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    data = {
        'chunks': document_chunks,
        'status': document_status
    }
    with open(os.path.join(DATA_DIR, 'document_data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    if vectorizer is not None:
        with open(os.path.join(DATA_DIR, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

    if chunk_vectors is not None:
        np.save(os.path.join(DATA_DIR, 'vectors.npy'), chunk_vectors)

def load_processed_data():
    """Load processed data from disk"""
    try:
        with open(os.path.join(DATA_DIR, 'document_data.pkl'), 'rb') as f:
            data = pickle.load(f)
            document_chunks = data['chunks']
            document_status = data['status']

        with open(os.path.join(DATA_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)

        chunk_vectors = np.load(os.path.join(DATA_DIR, 'vectors.npy'))

        logger.info("Loaded processed data from disk")
        return document_chunks, document_status, vectorizer, chunk_vectors
    except Exception as e:
        logger.info(f"Could not load processed data: {e}")
        return None, None, None, None
