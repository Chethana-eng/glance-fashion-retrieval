import numpy as np


class QueryEncoder:

    def __init__(self, encoder):

        self.encoder = encoder

    def encode(self, prompts):

        weighted_embedding = None

        for prompt, weight in prompts:

            embedding = self.encoder.encode_text(prompt)

            embedding = embedding.cpu().numpy().squeeze()

            total_weight = sum(weight for _, weight in prompts)
            embedding *= (weight / total_weight)

            if weighted_embedding is None:

                weighted_embedding = embedding

            else:

                weighted_embedding += embedding

        weighted_embedding /= np.linalg.norm(weighted_embedding)

        return weighted_embedding.astype(np.float32)