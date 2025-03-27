import trectools

# Load qrels (ground truth)
qrels = {
    "1": {"D1": 1, "D2": 0},
    "2": {"D3": 2, "D4": 1},
}

# Load system rankings
bert_run = {
    "1": {"D1": 0.85, "D2": 0.75},
    "2": {"D3": 0.90, "D4": 0.70},
}

doc2vec_run = {
    "1": {"D2": 0.80, "D1": 0.65},
    "2": {"D4": 0.88, "D3": 0.72},
}

# Define evaluation metrics
metrics = {"map", "P_10"}

# Initialize evaluator
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

# Evaluate both systems
bert_results = evaluator.evaluate(bert_run)
doc2vec_results = evaluator.evaluate(doc2vec_run)

# Compute mean scores across all queries
def compute_mean(results, metrics):
    return {metric: sum(results[q][metric] for q in results) / len(results) for metric in metrics}

bert_scores = compute_mean(bert_results, metrics)
doc2vec_scores = compute_mean(doc2vec_results, metrics)

# Print results
print("BERT Scores:", bert_scores)
print("Doc2Vec Scores:", doc2vec_scores)