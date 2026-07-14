from pathlib import Path
import yaml


class QueryParser:

    def __init__(self, config_path="configs/query_vocab.yaml"):

        with open(config_path, "r") as file:
            vocab = yaml.safe_load(file)

        self.vocab = {
            key: sorted(values, key=len, reverse=True)
            for key, values in vocab.items()
        }

    def parse(self, query):

        query = query.lower()

        result = {
            "color": [],
            "garment": [],
            "scene": [],
            "style": []
        }

        for category, words in self.vocab.items():

            remaining = query

            for phrase in words:

                if phrase in remaining:

                    result[category].append(phrase)

                    remaining = remaining.replace(phrase, "")
        return result