import preprocessing
import indexing
import retrieval
import json
import time
from sentence_transformers import SentenceTransformer, util
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

# BERT
model = SentenceTransformer("all-MiniLM-L6-v2")

# True for a2. False for a1
flag = True

if not flag:
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

else:
    # BERT preprocessed data
    # doc_joined {"id" : "string"} query_joined {"id" : "string"} query_ids {"id". "id"...} doc_ids {"query_id":["doc_id","doc_id"...]}
    doc_joined, query_joined, query_ids, doc_ids = preprocessing.preprocess_bert("Results_title_full.txt")

    # Doc2Vec
    # doc_lookup["id"] = ["token","token"...] query_lookup["id"] = ["token","token"...]
    # doc_lookup,query_lookup = preprocessing.preprocess_Doc2Vec()

    # Can use doc_lookup["id"] = ["token","token"...] to get doc token with doc id
    print("Starting BERT-based neural re-ranking...")
    doc_embeddings = {}
    for doc_id, text in doc_joined.items():
        doc_embeddings[doc_id] = model.encode(text)

    query_embeddings = {}
    for qid, text in query_joined.items():
        query_embeddings[qid] = model.encode(text)

    reranked_results = {}
    for qid in query_ids:
        candidates = doc_ids[qid]  # ["doc_id1", "doc_id2", ...]
        q_embed = query_embeddings[qid]

        # Calculate similarity
        scores = []
        for doc_id in candidates:
            d_embed = doc_embeddings[doc_id]
            score = util.cos_sim(q_embed, d_embed).item()  # 余弦相似度
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        reranked_results[qid] = scores

    with open("Results_neural_rerank.txt", "w", encoding="utf-8") as fout:
        for qid, doc_score_list in reranked_results.items():
            for rank, (doc_id, score) in enumerate(doc_score_list, 1):
                fout.write(f"{qid} Q0 {doc_id} {rank} {score:.4f} neural_run\n")

    print("Neural re-ranking done. Results saved to Results_neural_rerank.txt")

    # Doc2Vec
    print("Starting Doc2Vec-based neural re-ranking...")
    # 构建TaggedDocument列表用于Doc2Vec训练
    tagged_docs = []
    for doc_id, text in doc_joined.items():
        tokens = text.split()
        tagged_docs.append(TaggedDocument(words=tokens, tags=[doc_id]))

    # 训练Doc2Vec模型
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
    doc2vec_model.build_vocab(tagged_docs)
    doc2vec_model.train(tagged_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    # 生成Doc2Vec嵌入
    d2v_doc_embeddings = {}
    for doc_id, text in doc_joined.items():
        tokens = text.split()
        d2v_doc_embeddings[doc_id] = doc2vec_model.infer_vector(tokens)

    d2v_query_embeddings = {}
    for qid, text in query_joined.items():
        tokens = text.split()
        d2v_query_embeddings[qid] = doc2vec_model.infer_vector(tokens)


    # Cosine Similarity
    def cosine_sim(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)


    d2v_reranked_results = {}
    for qid in query_ids:
        candidates = doc_ids[qid]
        q_embed = d2v_query_embeddings[qid]
        scores = []
        for doc_id in candidates:
            d_embed = d2v_doc_embeddings[doc_id]
            score = cosine_sim(q_embed, d_embed)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        d2v_reranked_results[qid] = scores

    with open("Results_neural_rerank_doc2vec.txt", "w", encoding="utf-8") as fout:
        for qid, doc_score_list in d2v_reranked_results.items():
            for rank, (doc_id, score) in enumerate(doc_score_list, 1):
                fout.write(f"{qid} Q0 {doc_id} {rank} {score:.4f} neural_doc2vec\n")
    print("Doc2Vec-based neural re-ranking done. Results saved to Results_neural_rerank_doc2vec.txt")
