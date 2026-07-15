import faiss
import pickle
import numpy as np

from src.query.query_processor import QueryProcessor


class Retriever:

    def __init__(
        self,
        encoder,
        index_path="outputs/faiss.index",
        metadata_path="outputs/image_paths.pkl"
    ):

        self.processor = QueryProcessor(encoder)

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.image_paths = pickle.load(f)

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = self.processor.process(query)

        query_embedding = np.expand_dims(
            query_embedding,
            axis=0
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            results.append(
                {
                    "rank": len(results) + 1,
                    "image": self.image_paths[idx],
                    "score": float(score)
                }
            )

        return results