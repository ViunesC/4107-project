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
    # TODO
    return inverted_index


def save_index_tofile(index):
    """
    Save index to JSON file

    :param index: a dictionary containing inverted index
    :return: text file containing index
    """
    # TODO


def load_index_fromfile(filename):
    """
    Load index from JSON file

    :param filename: name of JSON file containing index
    :return: a dictionary containing inverted index
    """
    # TODO
