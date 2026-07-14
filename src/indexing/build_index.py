import faiss
import pickle
import numpy as np
from tqdm import tqdm


class FAISSIndexer:

    def __init__(self, encoder):

        self.encoder = encoder

        self.dimension = 512

        self.index = faiss.IndexFlatIP(self.dimension)

        self.image_paths = []


    def build_index(self, images):

        embeddings = []

        for image in tqdm(images):

            embedding = self.encoder.encode_image(
                image["path"]
            )

            embedding = embedding.cpu().numpy()

            embedding = embedding.squeeze()

            embedding = embedding.astype(np.float32)

            embeddings.append(embedding)

            self.image_paths.append(
                image["path"]
            )

        embeddings = np.array(embeddings)

        self.index.add(embeddings)

        print(f"Indexed {len(images)} images.")


    def save_index(
        self,
        index_path="outputs/faiss.index",
        metadata_path="outputs/image_paths.pkl"
    ):

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.image_paths, f)

        print("FAISS index saved successfully.")


    def load_index(
        self,
        index_path="outputs/faiss.index",
        metadata_path="outputs/image_paths.pkl"
    ):

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.image_paths = pickle.load(f)

        print("FAISS index loaded successfully.")