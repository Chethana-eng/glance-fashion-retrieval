from src.query.query_parser import QueryParser

parser = QueryParser()

query = "A person wearing a red dress sitting on a park bench"

print(parser.parse(query))