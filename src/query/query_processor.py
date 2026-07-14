from src.query.query_parser import QueryParser
from src.query.query_builder import QueryBuilder
from src.query.query_encoder import QueryEncoder


class QueryProcessor:

    def __init__(self, encoder):

        self.parser = QueryParser()

        self.builder = QueryBuilder()

        self.encoder = QueryEncoder(encoder)

    def process(self, query):

        print("\nOriginal Query:")
        print(query)

        parsed_query = self.parser.parse(query)

        print("\nParsed Query:")
        print(parsed_query)

        prompts = self.builder.build_prompts(
            parsed_query,
            query
        )

        print("\nGenerated Prompts:")
        for prompt, weight in prompts:
            print(f"{weight:.2f} -> {prompt}")

        query_embedding = self.encoder.encode(prompts)

        return query_embedding