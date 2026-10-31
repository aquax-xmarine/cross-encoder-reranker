import json
import csv

def load_corpus(path="scifact/corpus.jsonl"):
    corpus = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc["title"] + " " + doc["text"]
    return corpus

def load_queries(path="scifact/queries.jsonl"):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries

def load_qrels(path="scifact/qrels/test.tsv"):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for qid, doc_id, score in reader:
            qrels.setdefault(qid, {})[doc_id] = int(score)
    return qrels

if __name__ == "__main__":
    corpus = load_corpus()
    queries = load_queries()
    qrels = load_qrels()
    print("Documents:", len(corpus))
    print("Queries:", len(queries))
    print("Test queries with answers:", len(qrels))

    qid = next(iter(qrels))
    print("\nExample claim:", queries[qid])
    for doc_id in qrels[qid]:
        print("Correct doc:", corpus[doc_id][:200], "...")