from embedder import Embedder
import numpy as np


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.index, self.metadata = self.embedder.load_faiss_index()

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.embed_texts([query])
        query_embedding = np.array(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results


if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("Eligibility criteria for research grant")

    for r in results:
        print("\n---")
        print("Source:", r["source"])
        print(r["content"][:500])