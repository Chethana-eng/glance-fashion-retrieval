from src.dataset.dataset_loader import DatasetLoader
from src.models.fashion_clip import FashionCLIPEncoder
from src.indexing.build_index import FAISSIndexer


def main():

    loader = DatasetLoader("data/images")
    images = loader.load_images()
    encoder = FashionCLIPEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    indexer = FAISSIndexer(encoder)
    indexer.build_index(images)
    indexer.save_index()


if __name__ == "__main__":
    main()