import json
import numpy as np
from scipy.ndimage import prewitt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def load_inverted_index(index_file):
    with open(index_file, "r") as f:
        return json.load(f)


def load_docs(doc_file):
    docs = {}
    with open(doc_file, "r") as f:
        for line in f:
            doc = json.loads(line)
            docs[doc["_id"]] = " ".join(doc["title"] + doc["text"])
    return docs


def load_queries(query_file):
    queries = {}
    with open(query_file, "r") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = " ".join(query["text"])
    return queries


def get_relevant_docs(query, inverted_index):
    relevant_docs = set()
    query_terms = query.split()

    for term in query_terms:
        if term in inverted_index:
            relevant_docs.update(inverted_index[term].keys())

    return list(relevant_docs)


# TF-IDF and retrieval
def search_with_index(corpus_file, query_file, index_file, output_file):
    inverted_index = load_inverted_index(index_file)
    docs = load_docs(corpus_file)
    queries = load_queries(query_file)

    results = {}

    for query_id, query_text in queries.items():
        relevant_doc_ids = get_relevant_docs(query_text, inverted_index)

        if not relevant_doc_ids:
            results[query_id] = []
            continue

        #relevant_docs TF-IDF
        relevant_docs = [docs[doc_id] for doc_id in relevant_doc_ids]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(relevant_docs)
        query_tfidf = vectorizer.transform([query_text])

        #cosine_similarities
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
        ranked_indices = np.argsort(cosine_similarities)[::-1]


        print(f" Search {query_id}  finished")
        results[query_id] = [(relevant_doc_ids[idx], cosine_similarities[idx]) for idx in ranked_indices[:100]]


    with open(output_file, "w") as f:
        for query_id, ranked_docs in results.items():
            for rank, (doc_id, score) in enumerate(ranked_docs, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} run_time\n")

    print(f"saved as {output_file}")



if __name__ == "__main__":
    search_with_index(
        "preprocessed_docs.jsonl",
        "preprocessed_queries.jsonl",
        "inverted_index.json",
        "Results.txt"
    )
