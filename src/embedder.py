from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def save_faiss_index(self, embeddings, metadata, index_path="vector_store/faiss.index"):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        os.makedirs("vector_store", exist_ok=True)

        faiss.write_index(index, index_path)

        with open("vector_store/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print("FAISS index saved.")

    def load_faiss_index(self, index_path="vector_store/faiss.index"):
        index = faiss.read_index(index_path)

        with open("vector_store/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return index, metadata