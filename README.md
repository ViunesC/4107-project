# 4107-project
Repo for CSI4107 2025W
This project implements an Information Retrieval (IR) system based on the vector space model for a collection of scientific documents. The system processes a corpus and a set of test queries, builds an inverted index, and performs retrieval and ranking using a TF-IDF weighted vector space model with cosine similarity.

Two different runs were executed:

1. Title Only: Documents are represented using only the title field.

2. Title + Full Text: Documents are represented using the concatenation of the title and the full text.

The results of both runs are evaluated with trec_eval.

# Algorithms, Data Structures, and Optimizations Used

**Tokenization and Text Cleaning:** We implemented a preprocessing function (preprocess_text) that converts the text to lowercase and removes punctuation using regular expressions. We also use a helper function (safe_join) to handle both list and string types safely.

**Inverted Index**: Maps each token to a list of document IDs where the token appears, along with its term frequency.

**Dictionary for Documents and Queries:** Each document is stored in a dictionary, mapping IDs to their processed text representation.

**Uniform Preprocessing:** Used hash tables for fast lookups and insertions.Stored TF directly in the index to avoid recalculating it during retrieval. Indexed documents incrementally, processing one document at a time to reduce memory consumption.

**stopword**：Implemented stopword removal to eliminate frequent but non-informative words.

**TF-IDF weighting + Cosine Similarity ranking**：Constructed using TfidfVectorizer from sklearn and Used it for efficient TF-IDF computation. Used to rank documents based on their similarity to the query. Constructed a global TF-IDF matrix to avoid redundant computations. Used NumPy operations for efficient similarity calculations and sorting.

# First 10 Answers for the First 2 Queries

**Query 0: "0-dimension biomateri lack induct properti."**

0 Q0 42421723 1 0.1037 run_time

0 Q0 35008773 2 0.0922 run_time

0 Q0 994800 3 0.0855 run_time

0 Q0 26731863 4 0.0850 run_time

0 Q0 12156187 5 0.0800 run_time

0 Q0 14827874 6 0.0745 run_time

0 Q0 13231899 7 0.0732 run_time

0 Q0 18953920 8 0.0717 run_time

0 Q0 7581911 9 0.0693 run_time

0 Q0 21257564 10 0.0669 run_time

**Query 2: "1 in 5 million in uk have abnorm prp posit."**

2 Q0 13734012 1 0.3156 run_time

2 Q0 17333231 2 0.2836 run_time

2 Q0 42240424 3 0.2822 run_time

2 Q0 13770184 4 0.2405 run_time

2 Q0 18617259 5 0.1510 run_time

2 Q0 695938 6 0.1423 run_time

2 Q0 1292369 7 0.1265 run_time

2 Q0 32481310 8 0.1243 run_time

2 Q0 17415081 9 0.1194 run_time

2 Q0 3716075 10 0.1190 run_time

# MAP

**We evaluated our system using trec_eval for two configurations:**

1. Using only titles (Results_title_only.txt)
   
3. Using both titles and full text (Results_title_full.txt)

**Title Only 0.3867**

**Title + Text 0.5430**

The increase in recall suggests that full-text queries provided more context, enabling better ranking of relevant documents. Precision at top 5 and top 10 documents also increased, meaning the top-ranked documents were more likely to be relevant.
