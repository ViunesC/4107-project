import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def safe_join(field):
    """
    If the field is a list, it is connected with spaces, and if it is a string, it is returned directly.
    """
    if isinstance(field, list):
        return " ".join(field)
    elif isinstance(field, str):
        return field
    else:
        return str(field)


def preprocess_text(text):
    """
    Convert to lowercase and remove punctuation
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def load_inverted_index(index_file):
    with open(index_file, "r") as f:
        return json.load(f)


def load_docs(doc_file, use_full_text=True):
    """
    If use_full_text is True. represented as a stitching of title + text;

    If it is False, only title is used.
    """
    docs = {}
    with open(doc_file, "r") as f:
        for line in f:
            doc = json.loads(line)
            title_tokens = doc.get("title", [])
            text_tokens = doc.get("text", [])
            if use_full_text:
                combined_tokens = title_tokens + text_tokens
            else:
                combined_tokens = title_tokens
            full_text = preprocess_text(safe_join(combined_tokens))
            docs[doc["_id"]] = full_text
    return docs


def load_queries(query_file):
    queries = {}
    with open(query_file, "r") as f:
        for line in f:
            query = json.loads(line)
            query_text = preprocess_text(safe_join(query.get("text", [])))
            queries[query["_id"]] = query_text
    return queries


def get_relevant_docs(query, inverted_index):
    """
    if it exists in the inverted index, add id .
    """
    relevant_docs = set()
    query_terms = query.split()
    for term in query_terms:
        if term in inverted_index:
            relevant_docs.update(inverted_index[term].keys())
    return list(relevant_docs)


def search_with_index(corpus_file, query_file, index_file, output_file, use_full_text=True):
    inverted_index = load_inverted_index(index_file)
    docs = load_docs(corpus_file, use_full_text=use_full_text)
    queries = load_queries(query_file)

    #build TF-IDF model
    doc_ids = list(docs.keys())
    doc_texts = [docs[doc_id] for doc_id in doc_ids]

    vectorizer = TfidfVectorizer()
    global_tfidf_matrix = vectorizer.fit_transform(doc_texts)
    doc_index_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    results = {}

    for query_id, query_text in queries.items():
        candidate_doc_ids = get_relevant_docs(query_text, inverted_index)
        if not candidate_doc_ids:
            results[query_id] = []
            continue

        # filter out the documents that are not in the corpus
        candidate_indices = [doc_index_map[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_index_map]
        if not candidate_indices:
            results[query_id] = []
            continue

        candidate_matrix = global_tfidf_matrix[candidate_indices]
        query_vector = vectorizer.transform([query_text])
        cosine_similarities = cosine_similarity(query_vector, candidate_matrix).flatten()
        ranked_order = np.argsort(cosine_similarities)[::-1]

        print(f"Search {query_id} finished (use_full_text={use_full_text})")
        results[query_id] = [(candidate_doc_ids[idx], cosine_similarities[idx])
                             for idx in ranked_order[:100]]

    with open(output_file, "w") as f:
        for query_id, ranked_docs in results.items():
            for rank, (doc_id, score) in enumerate(ranked_docs, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} run_time\n")

    print(f"Saved results as {output_file}")


# if __name__ == "__main__":
#     search_with_index(
#         "preprocessed_docs.jsonl",
#         "preprocessed_queries.jsonl",
#         "inverted_index.json",
#         "Results_title_only.txt",
#         use_full_text=False
#     )
#
#     search_with_index(
#         "preprocessed_docs.jsonl",
#         "preprocessed_queries.jsonl",
#         "inverted_index.json",
#         "Results_title_full.txt",
#         use_full_text=True
#     )
