from pathlib import Path
import yaml


class QueryBuilder:

    def __init__(self, config_path="configs/query_weights.yaml"):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.weights = config["weights"]

    def build_prompts(self, parsed_query, original_query):

        prompts = []

        garment = parsed_query["garment"]
        color = parsed_query["color"]
        scene = parsed_query["scene"]
        style = parsed_query["style"]

        if garment:

            garment_text = " and ".join(garment)

            if color:
                color_text = " ".join(color)
                prompt = f"A person wearing a {color_text} {garment_text}."
            else:
                prompt = f"A person wearing a {garment_text}."

            prompts.append(
                (
                    prompt,
                    self.weights["garment"]
                )
            )

        if scene:

            scene_text = " and ".join(scene)

            prompts.append(
                (
                    f"A person in {scene_text}.",
                    self.weights["scene"]
                )
            )

        if style:

            style_text = " and ".join(style)

            prompts.append(
                (
                    f"A person wearing a {style_text} outfit.",
                    self.weights["style"]
                )
            )

        if color and not garment:

            color_text = " ".join(color)

            prompts.append(
                (
                    f"A person wearing {color_text} clothing.",
                    self.weights["color"]
                )
            )

        prompts.append(
            (
                original_query,
                self.weights["full_query"]
            )
        )

        return prompts