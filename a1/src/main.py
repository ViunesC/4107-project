import preprocessing
import indexing
import retrieval
import json
import time


# Preprocess corpus and queries
print("Preprocessing documents and queries...")
start = time.time()

# Modify if required, by default place corpus.jsonl and queries.jsonl in /src/ folder (same as this file)
doc_path = '../scifact/corpus.jsonl'
query_path = '../scifact/queries.jsonl'

# result file
documents = []
queries = []

# preprocess document
with open(doc_path, "r", encoding="utf-8") as f:
    for line in f:
        doc_json = json.loads(line)
        documents.append(doc_json)
f.close()

documents = preprocessing.preprocess_documents(documents)

# preprocess query
with open(query_path, "r", encoding="utf-8") as f:
    for line in f:
        query_json = json.loads(line)
        queries.append(query_json)
f.close()

queries = preprocessing.preprocess_queries(queries)

end = time.time()
print(f"Preprocessing completed. Time elapsed: {end - start} seconds.")


# Indexing tokenized document
print("Indexing tokenized documents...")
start = time.time()

preprocessed_doc_path = 'preprocessed_docs.jsonl'

documents = preprocessing.load_preprocessed_docs(preprocessed_doc_path)
index = indexing.build_index(documents)
indexing.save_index_tofile(index, "inverted_index.json")

end = time.time()
print(f"Indexing completed. Time elapsed: {end - start} seconds.")


# Retrieve and rank the result
retrieval.search_with_index(
        "preprocessed_docs.jsonl",
        "preprocessed_queries.jsonl",
        "inverted_index.json",
        "Results_title_only.txt",
        use_full_text=False
    )

retrieval.search_with_index(
    "preprocessed_docs.jsonl",
    "preprocessed_queries.jsonl",
    "inverted_index.json",
    "Results_title_full.txt",
    use_full_text=True
)

print("Rank completed. Results have been stored in following files: ")
print("Results_title_full.txt, Results_title_only.txt")