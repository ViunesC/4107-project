from preprocessing import *
from indexing import *
import json
import time

print("Preprocessing documents...")
start = time.time()

doc_path = '../scifact/corpus.jsonl'

# result file
documents = []

# load file
with open(doc_path, "r", encoding="utf-8") as f:
    for line in f:
        doc_json = json.loads(line)
        documents.append(doc_json)

documents = preprocess_documents(documents)
end = time.time()
print(f"Preprocessing completed. Time elapsed: {end - start} seconds.")


# print("Indexing tokenized documents...")
# start = time.time()
#
# doc_path = 'preprocessed.jsonl'
#
# documents = load_preprocessed_docs(doc_path)
# index = build_index(documents)
# save_index_tofile(index,"inverted_index.json")
#
# end = time.time()
# print(f"Indexing completed. Time elapsed: {end - start} seconds.")

print("Preprocessing queries...")
start = time.time()

query_path = '../scifact/queries.jsonl'

# result file
queries = []

# load file
with open(query_path, "r", encoding="utf-8") as f:
    for line in f:
        query_json = json.loads(line)
        queries.append(query_json)

queries = preprocess_queries(queries)
end = time.time()
print(f"Preprocessing completed. Time elapsed: {end - start} seconds.")