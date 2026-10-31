import numpy as np
from sentence_transformers import CrossEncoder
from bm25_baseline import bm25_search, corpus, queries, qrels
from metrics import ndcg_at_k

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidate_ids):
    pairs = [(query, corpus[d]) for d in candidate_ids]
    scores = model.predict(pairs, batch_size=32)
    order = np.argsort(scores)[::-1]
    return [candidate_ids[i] for i in order]

if __name__ == "__main__":
    before, after = [], []
    for n, (qid, relevant) in enumerate(qrels.items(), 1):
        candidates = bm25_search(queries[qid], k=50)
        reranked = rerank(queries[qid], candidates)
        before.append(ndcg_at_k(candidates, relevant, 10))
        after.append(ndcg_at_k(reranked, relevant, 10))
        if n % 50 == 0:
            print(f"{n} queries done...")

    print(f"\nBM25 only       nDCG@10: {np.mean(before):.4f}")
    print(f"BM25 + rerank   nDCG@10: {np.mean(after):.4f}")