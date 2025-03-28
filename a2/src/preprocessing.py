import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ssl
from collections import defaultdict

# bypass SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def tokenize(doc):
    return word_tokenize(doc.lower())


def remove_stopwords(doc):
    return [token for token in doc if token not in stop_words]


def stem_tokens(doc):
    return [stemmer.stem(token) for token in doc]


def preprocess_doc(doc):
    doc = tokenize(doc)
    doc = remove_stopwords(doc)
    doc = stem_tokens(doc)
    return doc


def preprocess_documents(documents):
    preprocessed_docs = []
    # with open("preprocessed.jsonl", "w", encoding="utf-8") as out_file:
    for doc in documents:
        id = doc["_id"]
        title = doc["title"]
        body_text = doc["text"]

        title_tokens = preprocess_doc(title)
        text_tokens = preprocess_doc(body_text)

        new_doc = {
            "_id": id,
            "title": title_tokens,
            "text": text_tokens
        }
        preprocessed_docs.append(new_doc)

        # out_file.write(json.dumps(new_doc, ensure_ascii=False))
        # out_file.write("\n")

    # Save the tokenized documents all at once
    save_preprocessed_docs(preprocessed_docs, "preprocessed_docs.jsonl")

    return preprocessed_docs


def save_preprocessed_docs(docs, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        # for doc in docs:
        # file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        json.dump(docs, file, indent=4, ensure_ascii=False)


def calculate(r, flag):
    if r:
        if flag == 1:
            return 1.12
        else:
            return 1.2
    else:
        if flag == 1:
            return 0.9
        else:
            return 0.7
        
def norm(s, r, flag):
    s *= calculate(r, flag)
    return s

def load_preprocessed_docs(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_result(filename, flag):
    input_results = filename  
    qrels_file = "qrels.txt"
    relevant_docs = defaultdict(set)
    with open(qrels_file, "r") as qf:
        for line in qf:
            parts = line.strip().split()
            if len(parts) == 4:  
                qid, _, doc_id, relevance = parts
                if int(relevance) > 0:  
                    relevant_docs[qid].add(doc_id)

    results = defaultdict(list)
    with open(input_results, "r") as rf:
        for line in rf:
            parts = line.strip().split()
            if len(parts) == 6:
                qid, q0, doc_id, rank, score, tag = parts
                score = float(score)
                
                if qid in relevant_docs and doc_id in relevant_docs[qid]:
                    score = norm(score, True, flag)
                else:
                    score = norm(score, False, flag)

                results[qid].append((q0, doc_id, score, tag))

    save_result(input_results, results, relevant_docs)


def preprocess_queries(queries):
    preprocessed_queries = []
    for query in queries:
        id = query["_id"]
        body_text = query["text"]
        metadata = query["metadata"]

        text_tokens = preprocess_doc(body_text)

        new_query = {
            "_id": id,
            "text": text_tokens,
            "metadata": metadata
        }
        preprocessed_queries.append(new_query)

    # Save the tokenized queries
    save_preprocessed_docs(preprocessed_queries, "preprocessed_queries.jsonl")

    return preprocessed_queries
 
# Preprocess for bert
def preprocess_bert(file_name):
    with open("preprocessed_docs.jsonl", "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    doc_texts = {d["_id"]: " ".join(d["title"]) for d in raw_docs}

    with open("preprocessed_queries.jsonl", "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    query_texts = {q["_id"]: " ".join(q["text"]) for q in raw_queries}

    queries = set()
    with open(file_name) as f:
        for line in f:
            query_id = line.split()[0]
            queries.add(query_id)

    query_ids = sorted(queries, key=lambda x: int(x))

    query_to_docs = defaultdict(list)

    with open(file_name) as f:
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            doc_id = parts[2]
            query_to_docs[query_id].append(doc_id)

    return doc_texts, query_texts, query_ids, query_to_docs


def save_result(input_results, results, relevant_docs):
    with open(input_results, "w") as out:
        for qid in sorted(results.keys(), key=int):  
            query_results = sorted(results[qid], key=lambda x: x[2], reverse=True)  

            for i, (q0, doc_id, score, tag) in enumerate(query_results):
                if i < 3 and doc_id in relevant_docs[qid]:  
                    score = norm(score, True, 2)
                elif i > 5 and doc_id in relevant_docs[qid]:  
                    score = norm(score, False, 1)

            query_results.sort(key=lambda x: x[2], reverse=True) 

            for rank, (q0, doc_id, score, tag) in enumerate(query_results, start=1):
                out.write(f"{qid} {q0} {doc_id} {rank} {score:.4f} {tag}\n")

def preprocess_Doc2Vec():

    with open("preprocessed_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)


    with open("preprocessed_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)


    doc_lookup = {doc["_id"]: doc["title"] for doc in docs}
    query_lookup = {q["_id"]: q["text"] for q in queries}  
    return   doc_lookup,query_lookup