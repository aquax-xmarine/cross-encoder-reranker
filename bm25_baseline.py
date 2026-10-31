import re
import numpy as np
from rank_bm25 import BM25Okapi
from data import load_corpus, load_queries, load_qrels
from metrics import ndcg_at_k, recall_at_k

def tokenize(text):
    return re.findall(r"\w+", text.lower())

corpus = load_corpus()
queries = load_queries()
qrels = load_qrels()

doc_ids = list(corpus.keys())
bm25 = BM25Okapi([tokenize(corpus[d]) for d in doc_ids])

def bm25_search(query, k=100):
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in top]

if __name__ == "__main__":
    ndcgs, recalls = [], []
    for qid, relevant in qrels.items():
        ranked = bm25_search(queries[qid])
        ndcgs.append(ndcg_at_k(ranked, relevant, 10))
        recalls.append(recall_at_k(ranked, relevant, 50))

    print(f"BM25  nDCG@10:   {np.mean(ndcgs):.4f}")
    print(f"BM25  Recall@50: {np.mean(recalls):.4f}")