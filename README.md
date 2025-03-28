# 4107 Assignment 2
This assignment modified solution to previous assignment, and implemented an Information Retrieval (IR) system based on neural models, BERT and Doc2Vec for a collection of scientific documents. The system processes a corpus and a set of test queries, builds an inverted index, and performs retrieval and ranking using two models with cosine similarity.

Two different runs were executed:

1. BERT.

2. Doc2Vec.

The results of both runs are evaluated with *trec_eval*.



## Identification

Group 33: Yucheng Chen (300194614), Junyang Wang (300241369), Danning Chen (300234800)



Task distribution:

| Name         | Responsibility                            |
| ------------ | ----------------------------------------- |
| Danning Chen | Modifying pipeline to adapt to new model, Running experiment on BERT |
| Yucheng Chen | Modifying preprocessing module to adapt to new model, Running experiment on Doc2Vec      |
| Junyang Wang | Comparing results and calculate metrics using trec_eval, Report writing |



## How to run

1. Install [python 3.12 or above](https://www.python.org/downloads/)

2. Install [NLTK](https://www.nltk.org/install.html):

   For Linux/MacOS (might need to install pip first if not already do so):

   ```
   pip install --user -U nltk
   ```

   For Windows: Check out [here](https://pypi.org/project/nltk/)

3. Install Scikit-learn:

   ```
   pip install -U scikit-learn
   ```


4. Install NLTK 'punkt_tab' and 'stopwords' data

   Open python terminal and run following commands:

   ```
   >>> import nltk
   >>> nltk.download('punkt_tab')
   >>> nltk.download('stopwords', quiet=True)
   ```

5. Install SentenceTransformer:

   ```
   pip install -U sentence-transformers
   ```

6. Install Gensim:

   ```
   pip install --upgrade gensim
   ```

7. Compile and run src/main.py



**Note**

The result of computing MAP with trec_eval are stored in *MAP_score.txt*. To compute MAP score yourself, check out this guide for installation and usage: https://aldolipani.com/trec_eval-installation-usage-and-behaviour/



## Algorithms, Data Structures, and Optimizations Used

1. **Tokenization and Text Cleaning:** We implemented a preprocessing function (preprocess_text) that converts the text to lowercase and removes punctuation using regular expressions. We also use a helper function (safe_join) to handle both list and string types safely.

2. **Inverted Index**: Maps each token to a list of document IDs where the token appears, along with its term frequency ($d_{f}$ and $tf_i$).

3. **Dictionary for Documents and Queries:** Each document is stored in a dictionary, mapping IDs to their processed text representation.

4. **Uniform Preprocessing:** Used hash tables for fast lookups and insertions. Stored TF directly in the index to avoid recalculating it during retrieval. Indexed documents incrementally, processing one document at a time to reduce memory consumption.

5. **Stop word:** Implemented stop word removal to eliminate frequent but non-informative words.

6. **TF-IDF weighting + Cosine Similarity ranking:** We constructed our weighting based on following formula while using *TfidfVectorizer* from *Scikit-learn* for efficient TF-IDF computation. 
   $$
   w_{ij} = tf_{ij}*idf_i
   $$
   The weighting was used to rank documents based on their similarity to the query. Constructed a global TF-IDF matrix to avoid redundant computations and utilize *NumPy* operations for efficient similarity calculations and sorting.

7. **Importing new model**: We imported *all-MiniLM-L6-v2*, a variant of BERT model from SentenceTransformer, and *doc2vec* from Gensim. 

8. **Rerank**: Then, we used our system from a1 to produce initial results ("Results_title_full.txt"), then re-ranked them based on a new similarity score between the query and each selected document. The result was stored in "Results_neural_rerank.txt" (for BERT) and "Results_neural_rerank_doc2vec.txt" (for Doc2Vec). 



## First 10 Answers for the Query 1 and 3

neural_run is result of using BERT model and neural_doc2vec is result of using Doc2Vec

#### Query 0: "0-dimension biomateri lack induct properti."

0 Q0 16541762 1 0.3943 neural_run

0 Q0 10786948 2 0.3832 neural_run

0 Q0 8185080 3 0.3413 neural_run

0 Q0 45364685 4 0.3188 neural_run

0 Q0 12156187 5 0.3081 neural_run

0 Q0 18953920 6 0.2934 neural_run

0 Q0 1944452 7 0.2778 neural_run

0 Q0 28249680 8 0.2750 neural_run

0 Q0 994800 9 0.2672 neural_run

0 Q0 12824568 10 0.2533 neural_run

0 Q0 10582939 1 0.7279 neural_doc2vec

0 Q0 19510470 2 0.7070 neural_doc2vec

0 Q0 11335860 3 0.6592 neural_doc2vec

0 Q0 11390393 4 0.6468 neural_doc2vec

0 Q0 32001951 5 0.6395 neural_doc2vec

0 Q0 26886351 6 0.6315 neural_doc2vec

0 Q0 26025820 7 0.6194 neural_doc2vec

0 Q0 16532419 8 0.6123 neural_doc2vec

0 Q0 16541762 9 0.6035 neural_doc2vec

0 Q0 5372773 10 0.6027 neural_doc2vec


#### Query 3: "1 in 5 million in uk have abnorm prp posit."

3 Q0 15153602 1 0.5460 neural_run

3 Q0 3672261 2 0.4873 neural_run

3 Q0 4632921 3 0.4822 neural_run

3 Q0 23389795 4 0.4684 neural_run

3 Q0 1544804 5 0.4473 neural_run

3 Q0 4378885 6 0.4403 neural_run

3 Q0 14729253 7 0.4354 neural_run

3 Q0 14019636 8 0.4282 neural_run

3 Q0 14717500 9 0.4220 neural_run

3 Q0 14376683 10 0.4068 neural_run

3 Q0 13519661 11 0.4010 neural_run

3 Q0 1544804 1 0.8202 neural_doc2vec

3 Q0 15153602 2 0.7784 neural_doc2vec

3 Q0 1067605 3 0.7446 neural_doc2vec

3 Q0 5650232 4 0.7346 neural_doc2vec

3 Q0 13791788 5 0.7321 neural_doc2vec

3 Q0 14717500 6 0.7292 neural_doc2vec

3 Q0 13373629 7 0.7137 neural_doc2vec

3 Q0 1631583 8 0.7063 neural_doc2vec

3 Q0 10944947 9 0.7048 neural_doc2vec

3 Q0 13949015 10 0.6671 neural_doc2vec

3 Q0 2739854 11 0.6654 neural_doc2vec


## Evaluation

**We evaluated our system using trec_eval for two configurations:**

1. BERT model

2. Doc2Vec model



| Configuration | MAP Score | Precison@10 |
| ------------- | ------ |---|
| Title         | 0.3867 |0.0623|
| Title + Text  | 0.5430 |0.0804|
| Doc2Vec | 0.4784 |0.0690|
| BERT | 0.6452 |0.0814|



The increase in MAP suggests that using BERT model to perform the task leads to better result, enabling better ranking of relevant documents. Precision at top 10 documents also increased, meaning the top-ranked documents were more likely to be relevant.

### BERT

```
viunesc@DESKTOP-8H6BMQH:~/projects/4107-project/a2/src$ trec_eval qrels.txt Results_neural_rerank.txt -m map -m P.10
map                     all     0.6452
P_10                    all     0.0814
```

### Doc2Vec

```
viunesc@DESKTOP-8H6BMQH:~/projects/4107-project/a2/src$ trec_eval qrels.txt Results_neural_rerank_doc2vec.txt -m map -m P.10
map                     all     0.4784
P_10                    all     0.0690
```
