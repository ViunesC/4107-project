from preprocessing import *
import json

print("Step 1 starts")
doc_path ='./scifact/corpus.jsonl'

#result file
documents = []

#load file
with open(doc_path, "r", encoding="utf-8") as f:
    for line in f:
        doc_json = json.loads(line)
        documents.append(doc_json)

documents = preprocess_documents(documents)
print("Step 1 ends")