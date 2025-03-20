"""
Module for indexing tokenized documents
:author JunYang Wang

For CSI-4107 Assignment 1
"""
import json


def build_index(documents) -> dict:
    """
    Generate inverted index from given documents

    :param documents: tokenized documents
    :return: a dictionary containing inverted index
    """
    inverted_index = {}
    for document in documents:
        document_id = document["_id"]
        content_tokens = document["text"]

        for token in content_tokens:
            # If the token does not appear before
            if token not in inverted_index:
                inverted_index[token] = {}

            # If token is not discovered in current document
            if document_id not in inverted_index[token]:
                inverted_index[token][document_id] = 0

            # Increment the token count
            inverted_index[token][document_id] += 1

    return inverted_index


def save_index_tofile(index, file_name):
    """
    Save index to JSON file

    :param index: a dictionary containing inverted index
    :param file_name: name of the saved file
    :return: JSON file containing index
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=4)


def load_index_fromfile(file_name) -> dict:
    """
    Load index from JSON file

    :param file_name: name of JSON file containing index
    :return: a dictionary containing inverted index
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)

    return inverted_index
