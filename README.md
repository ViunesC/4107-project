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
| Danning Chen | Modifying retrieval and indexing module to adapt to new model, Building BERT pipeline |
| Yucheng Chen | Modifying preprocessing module to adapt to new model, Building Doc2Vec pipeline |
| Junyang Wang | Conducting experiments on two models, Calculate metrics using trec_eval, Report writing |



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

7. **Importing new model**: We imported *all-MiniLM-L6-v2*, a variant of BERT model (Based on MiniLM, smaller version of BERT) from SentenceTransformer, and *doc2vec* from Gensim. Then we trained them with exsiting scores by chunking documents, embedding chuncks using all-MiniLM-L6-v2, storing embeddings in vector db and getting top 100 documents based on similarity with query embedding.

8. **Rerank**: Then, we used our system from a1 to produce initial results ("Results_title_full.txt"), then re-ranked them based on a new similarity score between the query and each selected document. The result was stored in "Results_neural_rerank.txt" (for BERT) and "Results_neural_rerank_doc2vec.txt" (for Doc2Vec). 



## First 10 Answers for the Query 1 and 3

neural_run is result of using BERT model and neural_doc2vec is result of using Doc2Vec

#### Query 0: "0-dimension biomateri lack induct properti."

0 Q0 16541762 1 0.3549 neural_run

0 Q0 10786948 2 0.3449 neural_run

0 Q0 8185080 3 0.3072 neural_run

0 Q0 45364685 4 0.2869 neural_run

0 Q0 12156187 5 0.2773 neural_run

0 Q0 18953920 6 0.2641 neural_run

0 Q0 1944452 7 0.2500 neural_run

0 Q0 28249680 8 0.2475 neural_run

0 Q0 994800 9 0.2405 neural_run

0 Q0 12824568 10 0.2280 neural_run

0 Q0 19510470 1 0.5157 neural_doc2vec

0 Q0 10582939 2 0.5020 neural_doc2vec

0 Q0 11335860 3 0.4572 neural_doc2vec

0 Q0 26886351 4 0.4511 neural_doc2vec

0 Q0 35008773 5 0.4444 neural_doc2vec

0 Q0 32001951 6 0.4439 neural_doc2vec

0 Q0 2686003 7 0.4330 neural_doc2vec

0 Q0 16532419 8 0.4292 neural_doc2vec

0 Q0 11390393 9 0.4278 neural_doc2vec

0 Q0 26025820 10 0.4236 neural_doc2vec




#### Query 3: "1 in 5 million in uk have abnorm prp posit."

3 Q0 15153602 1 0.4913 neural_run

3 Q0 14717500 2 0.4726 neural_run

3 Q0 3672261 3 0.4386 neural_run

3 Q0 4632921 4 0.4340 neural_run

3 Q0 23389795 5 0.4216 neural_run

3 Q0 1544804 6 0.4026 neural_run

3 Q0 4378885 7 0.3963 neural_run

3 Q0 14729253 8 0.3919 neural_run

3 Q0 14019636 9 0.3854 neural_run

3 Q0 14376683 10 0.3661 neural_run

3 Q0 13519661 11 0.3609 neural_run

3 Q0 14717500 1 0.8599 neural_doc2vec

3 Q0 1544804 2 0.5859 neural_doc2vec

3 Q0 15153602 3 0.5592 neural_doc2vec

3 Q0 13791788 4 0.5291 neural_doc2vec

3 Q0 13373629 5 0.5198 neural_doc2vec

3 Q0 5650232 6 0.5123 neural_doc2vec

3 Q0 11117679 7 0.5020 neural_doc2vec

3 Q0 10944947 8 0.5004 neural_doc2vec

3 Q0 4414547 9 0.4814 neural_doc2vec

3 Q0 2739854 10 0.4759 neural_doc2vec

3 Q0 1067605 11 0.4718 neural_doc2vec




## Evaluation

**We evaluated our system using trec_eval for two configurations:**

1. BERT model

2. Doc2Vec model



| Configuration | MAP Score | Precison@10 |
| ------------- | ------ | --- |
| Title         | 0.3867 | 0.0623 |
| Title + Text  | 0.5430 | 0.0804 |
| Doc2Vec | 0.5625 | 0.0727 |
| BERT | 0.6329 | 0.0850 |



The increase in MAP suggests that using BERT model to perform the task leads to better result, enabling better ranking of relevant documents. Precision at top 10 documents also increased, meaning the top-ranked documents were more likely to be relevant.



### BERT

```
viunesc@DESKTOP-8H6BMQH:~/projects/4107-project/a2/src$ trec_eval qrels.txt Results_neural_rerank.txt -m map -m P.10
map                     all     0.6329
P_10                    all     0.0850
```



### Doc2Vec

```
viunesc@DESKTOP-8H6BMQH:~/projects/4107-project/a2/src$ trec_eval qrels.txt Results_neural_rerank_doc2vec.txt -m map -m P.10
map                     all     0.5625
P_10                    all     0.0727
```
