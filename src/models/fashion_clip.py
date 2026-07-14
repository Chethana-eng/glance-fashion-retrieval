from pathlib import Path

import open_clip
import torch
from PIL import Image


class FashionCLIPEncoder:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"Device      : {self.device}")
        print(f"Model       : {model_name}")
        print(f"Checkpoint  : {pretrained}")

    @torch.no_grad()
    def encode_image(self, image_path: str):
        image = Image.open(Path(image_path)).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        embedding = self.model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu()

    @torch.no_grad()
    def encode_text(self, text: str):
        tokens = self.tokenizer([text]).to(self.device)

        embedding = self.model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu()