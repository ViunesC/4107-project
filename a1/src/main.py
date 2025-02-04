from preprocessing import *
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
