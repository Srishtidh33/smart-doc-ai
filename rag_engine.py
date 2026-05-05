import numpy as np
import faiss

class RAGEngine:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.chunks = []
        self.index = None

    # -------------------------
    # BUILD INDEX
    # -------------------------
    def build_index(self, chunks):
        self.chunks = chunks

        embeddings = self.model.encode(chunks)

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    # -------------------------
    # SEARCH FUNCTION
    # -------------------------
    def search(self, query, top_k=3):
        if self.index is None:
            return []

        query_vec = self.model.encode([query]).astype("float32")

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])

        return results