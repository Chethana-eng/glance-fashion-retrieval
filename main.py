from PIL import Image
import matplotlib.pyplot as plt

from src.models.fashion_clip import FashionCLIPEncoder
from src.retrieval.retrieve import Retriever


def main():

    encoder = FashionCLIPEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    retriever = Retriever(encoder)

    query = "A blue puffer jacket"

    results = retriever.retrieve(query=query, top_k=5)

    for result in results:
        print(result)

        image = Image.open(result["image"])

        plt.figure(figsize=(4,4))
        plt.imshow(image)
        plt.title(f"Score: {result['score']:.4f}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()