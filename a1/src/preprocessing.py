import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def tonkenize(doc):
    return word_tokenize(doc.lower())


def remove_stopwords(doc):
    return [token for token in doc if token not in stop_words]


def stem_tokens(doc):
    return [stemmer.stem(token) for token in doc]


def preprocess_doc(doc):
    doc = tonkenize(doc)
    doc = remove_stopwords(doc)
    doc = stem_tokens(doc)
    return doc


def preprocess_documents(documents):
    preprocessed_docs = []
    with open("preprocessed.jsonl", "w", encoding="utf-8") as out_file:
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

            out_file.write(json.dumps(new_doc, ensure_ascii=False))
            out_file.write("\n")

    return preprocessed_docs
