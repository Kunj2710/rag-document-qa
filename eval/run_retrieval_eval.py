"""Retrieval evaluation: hit rate, MRR and latency per vector store and embedding model.

Replaces the qualitative "retrieved more relevant chunks" / "faster" verdicts in
the README with numbers. Each question in questions_sec10k.jsonl has an
`answer_phrase` copied verbatim from the 10-K passage that answers it; a
retrieved passage counts as a hit if it contains that phrase.

Corpus: 1,880 passages from 10 SEC 10-K filings (AAPL, CVX, JPM, PFE, WMT;
two fiscal years each). Fetch it with ./fetch_corpus.sh (SHA-256 verified).

Examples
    python eval/run_retrieval_eval.py --embeddings tfidf          # offline, no keys
    python eval/run_retrieval_eval.py --embeddings minilm         # sentence-transformers
    python eval/run_retrieval_eval.py --embeddings openai         # needs OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).lower()


def load_corpus(path: Path):
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    return [r["content"] for r in rows], [{"id": r["id"], **r["meta"]} for r in rows]


def load_questions(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


class TfidfLsaEmbeddings(Embeddings):
    """Offline baseline: TF-IDF (1-2 grams) reduced to 384 dims with truncated SVD.

    Same dimensionality as MiniLM so vector-store latency is comparable.
    """

    def __init__(self, corpus: list[str], dim: int = 384):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
        X = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=dim, random_state=0).fit(X)

    def _emb(self, texts):
        Z = self.svd.transform(self.vec.transform(texts))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return Z.tolist()

    def embed_documents(self, texts):
        return self._emb(texts)

    def embed_query(self, text):
        return self._emb([text])[0]


def get_embeddings(name: str, corpus: list[str]):
    if name == "tfidf":
        return TfidfLsaEmbeddings(corpus)
    if name == "minilm":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if name == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    raise ValueError(name)


def build_store(store: str, texts, metas, emb):
    if store == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.from_texts(texts, emb, metadatas=metas)
    if store == "chroma":
        from langchain_community.vectorstores import Chroma

        return Chroma.from_texts(texts, emb, metadatas=metas, collection_name=f"eval_{int(time.time()*1e6)}")
    raise ValueError(store)


def evaluate(vs, questions, emb, k: int, repeats: int):
    hits1 = hitsk = rr = 0.0
    lat = []
    per_q = []
    # embed once so latency measures the vector store, not the embedding API
    qvecs = [emb.embed_query(q["question"]) for q in questions]
    for q, v in zip(questions, qvecs):
        for _ in range(repeats):
            t0 = time.perf_counter()
            docs = vs.similarity_search_by_vector(v, k=k)
            lat.append((time.perf_counter() - t0) * 1000)
        target = norm(q["answer_phrase"])
        rank = next((i + 1 for i, d in enumerate(docs) if target in norm(d.page_content)), None)
        hits1 += rank == 1
        hitsk += rank is not None
        rr += 1 / rank if rank else 0
        per_q.append({"id": q["id"], "rank": rank, "top1": docs[0].metadata.get("id")})
    n = len(questions)
    return {
        "hit@1": hits1 / n, f"hit@{k}": hitsk / n, f"MRR@{k}": rr / n,
        "p50 ms": float(np.percentile(lat, 50)), "p95 ms": float(np.percentile(lat, 95)),
    }, per_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data" / "sec-10k-2026.10.chunks.jsonl.gz"))
    ap.add_argument("--questions", default=str(HERE / "questions_sec10k.jsonl"))
    ap.add_argument("--embeddings", default="tfidf", choices=["tfidf", "minilm", "openai"])
    ap.add_argument("--stores", nargs="+", default=["faiss", "chroma"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20, help="timed searches per question")
    args = ap.parse_args()

    texts, metas = load_corpus(Path(args.corpus))
    questions = load_questions(Path(args.questions))
    for q in questions:  # every label must exist in the corpus
        assert any(norm(q["answer_phrase"]) in norm(t) for t in texts), q["id"]

    t0 = time.perf_counter()
    emb = get_embeddings(args.embeddings, texts)
    fit_s = time.perf_counter() - t0
    print(f"{len(texts):,} passages | {len(questions)} questions | embeddings={args.embeddings} (setup {fit_s:.1f}s)")

    rows = []
    for store in args.stores:
        t0 = time.perf_counter()
        vs = build_store(store, texts, metas, emb)
        build_s = time.perf_counter() - t0
        m, per_q = evaluate(vs, questions, emb, args.k, args.repeats)
        rows.append({"store": store, "embeddings": args.embeddings, "index build s": build_s, **m})
        out = HERE / "results"
        out.mkdir(exist_ok=True)
        json.dump(per_q, open(out / f"per_question_{args.embeddings}_{store}.json", "w"), indent=1)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / f"retrieval_{args.embeddings}.csv", index=False)
    print(df.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
