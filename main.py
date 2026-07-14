from src.dataset.dataset_loader import DatasetLoader
from src.models.fashion_clip import FashionCLIPEncoder


def main():
    loader = DatasetLoader("data/images")
    images = loader.load_images()

    encoder = FashionCLIPEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    embedding = encoder.encode_image(images[0]["path"])

    print(embedding.shape)


if __name__ == "__main__":
    main()