# Cross-Encoder Reranking on BEIR SciFact

Adds a cross-encoder reranker to BM25 retrieval and measures the improvement with nDCG@10.

## Results

| Method | nDCG@10 |
| --- | --- |
| BM25 | 0.6519 |
| BM25 + cross-encoder rerank (top 50) | 0.6818 |

BM25 Recall@50: 0.8657. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Setup

```bash
pip install -r requirements.txt
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip scifact.zip
python3 bm25_baseline.py
python3 rerank.py
```

## Files

- `data.py`: loads corpus, queries, and qrels
- `metrics.py`: nDCG@k and Recall@k
- `bm25_baseline.py`: BM25 retrieval and baseline scores
- `rerank.py`: reranks BM25's top 50 with a cross-encoder