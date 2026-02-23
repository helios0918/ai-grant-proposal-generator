from parser import load_all_pdfs
from text_chunker import chunk_text
from embedder import Embedder
import numpy as np

FUNDING_FOLDER = "data/funding_calls"

def build_vector_store():
    docs = load_all_pdfs(FUNDING_FOLDER)

    all_chunks = []
    metadata = []

    for filename, text in docs.items():
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "source": filename,
                "content": chunk
            })

    embedder = Embedder()
    embeddings = embedder.embed_texts(all_chunks)
    embeddings = np.array(embeddings)

    embedder.save_faiss_index(embeddings, metadata)


if __name__ == "__main__":
    build_vector_store()