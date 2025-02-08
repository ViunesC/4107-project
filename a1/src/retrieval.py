import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_index(index_file="inverted_index.json"):
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)
index = load_index()

def load_queries(query_file="preprocessed_queries.jsonl"):
    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = {}
    for query in data:
        query_id = int(query["_id"])
        if query_id % 2 == 1:
            queries[query_id] = " ".join(query["text"])

    return queries
queries = load_queries()

def load_corpus(corpus_file="../scifact/corpus.jsonl"):
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc["text"]
    return corpus
corpus = load_corpus()

def retrieve_tf_idf(queries, corpus):
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(doc_texts)

    results = []
    for qid, query in queries.items():
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, doc_vectors).flatten()

        ranked_docs = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked_docs[:100], 1):
            results.append(f"{qid} Q0 {doc_id} {rank} {score:.4f} run_name")

    return results

def save_results(results, output_file="Results.txt"):
    with open(output_file, "w") as f:
        f.write("\n".join(results))

results = retrieve_tf_idf(queries, corpus)
save_results(results)
print("Results.txt saved.")
