import numpy as np

def ndcg_at_k(ranked_ids, relevant, k=10):
    dcg = sum(relevant.get(d, 0) / np.log2(i + 2)
              for i, d in enumerate(ranked_ids[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_ids, relevant, k):
    found = sum(1 for d in ranked_ids[:k] if d in relevant)
    return found / len(relevant)