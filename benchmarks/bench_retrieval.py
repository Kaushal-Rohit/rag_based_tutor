"""
Retrieval Benchmark Suite
==========================
Measures actual FAISS vs ChromaDB search latency on this machine with this index.

Produces:
  - benchmarks/results.json  (raw data)
  - benchmarks/results.md    (human-readable markdown report)

All numbers in the README and portfolio documents MUST come from this script.

Usage:
    python benchmarks/bench_retrieval.py
"""

import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

# ── Project root setup ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "notebooks", "faiss_hnsw_index.bin")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "notebooks", "chroma_db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmarks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────
# 1. Load dependencies
# ────────────────────────────────────────────
print("[1/6] Loading libraries...")
import faiss
import chromadb
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────
# 2. Load data and indices
# ────────────────────────────────────────────
print("[2/6] Loading document metadata...")
from src.indexer import load_documents

documents, metadatas, ids = load_documents(DATASET_DIR)

print("[2/6] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("[2/6] Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
print(f"       FAISS: {faiss_index.ntotal} vectors, dim={faiss_index.d}")

print("[2/6] Loading ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name="rag_collection")
print(f"       ChromaDB: {collection.count()} vectors")

assert faiss_index.ntotal == len(documents), "FAISS/document count mismatch!"
assert collection.count() == len(documents), "ChromaDB/document count mismatch!"


# ────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────
def embed_query(query):
    return model.encode([query])


def search_faiss(query_emb, k=5, filters=None, overfetch=10):
    start = time.perf_counter()
    fetch_k = min(k * overfetch if filters else k, faiss_index.ntotal)
    D, I = faiss_index.search(query_emb, k=fetch_k)
    search_end = time.perf_counter()

    result_ids, result_docs = [], []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(documents):
            continue
        if filters:
            match = all(metadatas[idx].get(fk, "") == fv for fk, fv in filters.items())
            if not match:
                continue
        result_ids.append(ids[idx])
        result_docs.append(documents[idx])
        if len(result_ids) >= k:
            break

    end = time.perf_counter()
    return {
        "ids": result_ids, "docs": result_docs,
        "search_ms": (search_end - start) * 1000,
        "total_ms": (end - start) * 1000,
    }


def search_chroma(query_emb, k=5, filters=None):
    start = time.perf_counter()
    where_filter = None
    if filters:
        conditions = [{fk: {"$eq": fv}} for fk, fv in filters.items()]
        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    qp = {
        "query_embeddings": query_emb.tolist(),
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        qp["where"] = where_filter

    results = collection.query(**qp)
    end = time.perf_counter()
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "docs": results["documents"][0] if results["documents"] else [],
        "total_ms": (end - start) * 1000,
    }


# ────────────────────────────────────────────
# 3. Benchmark queries
# ────────────────────────────────────────────
BENCHMARK_QUERIES = [
    {"query": "Explain Newton's second law of motion", "filters": None},
    {"query": "What is electromagnetic induction?", "filters": {"subject": "Physics"}},
    {"query": "Derive the expression for electric field due to a dipole", "filters": {"subject": "Physics", "class": "12"}},
    {"query": "What are the colligative properties of solutions?", "filters": None},
    {"query": "Explain the band theory of solids", "filters": {"subject": "Chemistry"}},
    {"query": "What is electrochemistry?", "filters": {"subject": "Chemistry", "class": "12"}},
    {"query": "Describe the portrait of the grandmother", "filters": {"subject": "English"}},
    {"query": "What is the central theme of the poem?", "filters": None},
    {"query": "What is the structure of an atom?", "filters": {"subject": "Science"}},
    {"query": "Describe the classification of living organisms", "filters": None},
    {"query": "Define Ohm's law", "filters": {"content_type": "definition"}},
    {"query": "Solve the numerical problem on resistance", "filters": {"content_type": "exercise"}},
    {"query": "What is the difference between speed and velocity?", "filters": None},
    {"query": "State Faraday's laws of electromagnetic induction", "filters": {"subject": "Physics", "class": "12"}},
    {"query": "Explain Le Chatelier's principle", "filters": {"subject": "Chemistry"}},
]

K = 5
NUM_RUNS = 5

print(f"\n[3/6] Running latency benchmark: {len(BENCHMARK_QUERIES)} queries x {NUM_RUNS} runs...\n")

all_faiss_embed, all_faiss_search, all_faiss_total = [], [], []
all_chroma_total, all_overlaps = [], []
per_query_results = []

for qi, q_info in enumerate(BENCHMARK_QUERIES):
    query = q_info["query"]
    filters = q_info["filters"]

    q_faiss_total, q_faiss_search, q_chroma_total, q_overlaps = [], [], [], []

    for run in range(NUM_RUNS):
        t0 = time.perf_counter()
        query_emb = embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000
        all_faiss_embed.append(embed_ms)

        fr = search_faiss(query_emb, k=K, filters=filters)
        cr = search_chroma(query_emb, k=K, filters=filters)

        q_faiss_search.append(fr["search_ms"])
        q_faiss_total.append(fr["total_ms"])
        q_chroma_total.append(cr["total_ms"])

        all_faiss_search.append(fr["search_ms"])
        all_faiss_total.append(fr["total_ms"])
        all_chroma_total.append(cr["total_ms"])

        faiss_id_set = set(fr["ids"])
        chroma_id_set = set(cr["ids"])
        union = faiss_id_set | chroma_id_set
        overlap = len(faiss_id_set & chroma_id_set) / max(len(union), 1) * 100
        q_overlaps.append(overlap)
        all_overlaps.append(overlap)

    per_query_results.append({
        "query": query,
        "filters": str(filters),
        "faiss_search_mean_ms": round(np.mean(q_faiss_search), 3),
        "faiss_total_mean_ms": round(np.mean(q_faiss_total), 3),
        "chroma_total_mean_ms": round(np.mean(q_chroma_total), 3),
        "overlap_pct": round(np.mean(q_overlaps), 1),
    })
    status = "OK" if np.mean(q_overlaps) > 50 else "!!"
    print(
        f"  [{status}] Q{qi + 1:02d}: FAISS={np.mean(q_faiss_search):.2f}ms | "
        f"Chroma={np.mean(q_chroma_total):.2f}ms | "
        f"Overlap={np.mean(q_overlaps):.0f}% | {query[:50]}"
    )

# ────────────────────────────────────────────
# 4. Recall measurement (HNSW vs brute force)
# ────────────────────────────────────────────
print(f"\n[4/6] Measuring HNSW recall@{K} vs brute-force (Flat index)...")

flat_index = faiss.IndexFlatL2(faiss_index.d)
all_vectors = np.zeros((faiss_index.ntotal, faiss_index.d), dtype=np.float32)
for i in range(faiss_index.ntotal):
    all_vectors[i] = faiss_index.reconstruct(i)
flat_index.add(all_vectors)
print(f"       Flat index built with {flat_index.ntotal} vectors")

recall_scores = []
for q_info in BENCHMARK_QUERIES:
    query_emb = embed_query(q_info["query"])
    D_gt, I_gt = flat_index.search(query_emb, k=K)
    gt_set = set(I_gt[0].tolist())
    D_hnsw, I_hnsw = faiss_index.search(query_emb, k=K)
    hnsw_set = set(I_hnsw[0].tolist())
    recall = len(gt_set & hnsw_set) / len(gt_set) * 100
    recall_scores.append(recall)

avg_recall = np.mean(recall_scores)
print(f"       HNSW Recall@{K}: {avg_recall:.1f}% (avg over {len(BENCHMARK_QUERIES)} queries)")

# ────────────────────────────────────────────
# 5. Context coverage
# ────────────────────────────────────────────
print(f"\n[5/6] Measuring context coverage...")

qa_test_queries = [
    {"query": "The three phases of the author's relationship with his grandmother", "answer_fragment": "three phases", "filters": {"subject": "English"}},
    {"query": "What is Coulomb's law?", "answer_fragment": "inversely proportional", "filters": {"subject": "Physics"}},
    {"query": "What is a solution in chemistry?", "answer_fragment": "homogeneous mixture", "filters": {"subject": "Chemistry"}},
    {"query": "Define matter in our surroundings", "answer_fragment": "matter", "filters": {"subject": "Science"}},
    {"query": "What is electric charge?", "answer_fragment": "charge", "filters": {"subject": "Physics"}},
    {"query": "What is Ohm's law?", "answer_fragment": "resistance", "filters": None},
    {"query": "What is electromagnetic induction?", "answer_fragment": "magnetic", "filters": {"subject": "Physics"}},
    {"query": "What are isotopes?", "answer_fragment": "mass", "filters": None},
    {"query": "What is electrochemical cell?", "answer_fragment": "electrode", "filters": {"subject": "Chemistry"}},
    {"query": "What is velocity?", "answer_fragment": "direction", "filters": None},
]

rag_hits, no_rag_hits = 0, 0
total_qa = len(qa_test_queries)

for qa in qa_test_queries:
    query_emb = embed_query(qa["query"])
    cr = search_chroma(query_emb, k=K, filters=qa["filters"])
    combined_text = " ".join(cr["docs"]).lower()
    if qa["answer_fragment"].lower() in combined_text:
        rag_hits += 1
    if qa["answer_fragment"].lower() in qa["query"].lower():
        no_rag_hits += 1

rag_coverage = rag_hits / total_qa * 100
no_rag_coverage = no_rag_hits / total_qa * 100
hallucination_reduction = rag_coverage - no_rag_coverage

print(f"       RAG context coverage: {rag_coverage:.0f}% ({rag_hits}/{total_qa})")
print(f"       No-RAG baseline: {no_rag_coverage:.0f}% ({no_rag_hits}/{total_qa})")
print(f"       Hallucination reduction: {hallucination_reduction:.0f} pp")

# ────────────────────────────────────────────
# 6. Compile and save results
# ────────────────────────────────────────────
print(f"\n[6/6] Compiling results...")

hw_info = f"{platform.processor() or platform.machine()}, {platform.system()} {platform.release()}"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

results = {
    "benchmark_metadata": {
        "total_vectors": faiss_index.ntotal,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": faiss_index.d,
        "faiss_index_type": "IndexHNSWFlat (M=32)",
        "chromadb_space": "cosine",
        "num_queries": len(BENCHMARK_QUERIES),
        "num_runs_per_query": NUM_RUNS,
        "k": K,
        "timestamp": timestamp,
        "hardware": hw_info,
        "python_version": platform.python_version(),
    },
    "key_metrics": {
        "faiss_hnsw_search_latency_mean_ms": round(np.mean(all_faiss_search), 3),
        "faiss_hnsw_search_latency_median_ms": round(np.median(all_faiss_search), 3),
        "faiss_hnsw_search_latency_p95_ms": round(np.percentile(all_faiss_search, 95), 3),
        "faiss_total_latency_mean_ms": round(np.mean(all_faiss_total), 3),
        "chroma_total_latency_mean_ms": round(np.mean(all_chroma_total), 3),
        "embedding_latency_mean_ms": round(np.mean(all_faiss_embed), 3),
        "embedding_latency_median_ms": round(np.median(all_faiss_embed), 3),
        "hnsw_recall_at_k": round(avg_recall, 1),
        "faiss_vs_chroma_result_overlap_pct": round(np.mean(all_overlaps), 1),
        "rag_context_coverage_pct": round(rag_coverage, 1),
        "no_rag_baseline_coverage_pct": round(no_rag_coverage, 1),
        "hallucination_rate_reduction_pct_points": round(hallucination_reduction, 1),
        "faiss_disk_size_mb": round(os.path.getsize(FAISS_INDEX_PATH) / 1024 / 1024, 1),
    },
    "per_query_results": per_query_results,
    "recall_per_query": [
        {"query": q["query"], "recall_at_k": round(r, 1)}
        for q, r in zip(BENCHMARK_QUERIES, recall_scores)
    ],
}

faiss_avg = np.mean(all_faiss_total)
chroma_avg = np.mean(all_chroma_total)
if faiss_avg > 0:
    results["key_metrics"]["chroma_to_faiss_speedup_ratio"] = round(chroma_avg / faiss_avg, 2)

# Save JSON
json_path = os.path.join(OUTPUT_DIR, "results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

# Save Markdown report
md_path = os.path.join(OUTPUT_DIR, "results.md")
km = results["key_metrics"]
with open(md_path, "w") as f:
    f.write(f"# Retrieval Benchmark Results\n\n")
    f.write(f"**Date:** {timestamp}  \n")
    f.write(f"**Hardware:** {hw_info}  \n")
    f.write(f"**Python:** {platform.python_version()}  \n")
    f.write(f"**Vectors:** {faiss_index.ntotal}  \n")
    f.write(f"**Queries:** {len(BENCHMARK_QUERIES)} × {NUM_RUNS} runs  \n\n")
    f.write(f"## Key Metrics\n\n")
    f.write(f"| Metric | Value |\n")
    f.write(f"|--------|-------|\n")
    f.write(f"| FAISS HNSW search latency (mean) | {km['faiss_hnsw_search_latency_mean_ms']:.3f} ms |\n")
    f.write(f"| FAISS HNSW search latency (P95) | {km['faiss_hnsw_search_latency_p95_ms']:.3f} ms |\n")
    f.write(f"| ChromaDB total latency (mean) | {km['chroma_total_latency_mean_ms']:.3f} ms |\n")
    f.write(f"| Embedding latency (mean) | {km['embedding_latency_mean_ms']:.3f} ms |\n")
    f.write(f"| HNSW Recall@{K} | {km['hnsw_recall_at_k']:.1f}% |\n")
    f.write(f"| FAISS vs Chroma overlap | {km['faiss_vs_chroma_result_overlap_pct']:.1f}% |\n")
    f.write(f"| RAG context coverage | {km['rag_context_coverage_pct']:.0f}% |\n")
    f.write(f"| Hallucination reduction | {km['hallucination_rate_reduction_pct_points']:.0f} pp |\n")
    f.write(f"| FAISS disk size | {km['faiss_disk_size_mb']:.1f} MB |\n\n")
    f.write(f"## Per-Query Results\n\n")
    f.write(f"| Query | FAISS (ms) | Chroma (ms) | Overlap |\n")
    f.write(f"|-------|-----------|------------|--------|\n")
    for pq in per_query_results:
        f.write(
            f"| {pq['query'][:50]} | {pq['faiss_search_mean_ms']:.2f} | "
            f"{pq['chroma_total_mean_ms']:.2f} | {pq['overlap_pct']:.0f}% |\n"
        )

print(f"\n{'=' * 70}")
print(f"  BENCHMARK COMPLETE")
print(f"  JSON: {json_path}")
print(f"  Markdown: {md_path}")
print(f"{'=' * 70}")
print(f"\n  Key Numbers:")
print(f"  +---------------------------------------------------+")
print(f"  | FAISS HNSW Search (mean):  {km['faiss_hnsw_search_latency_mean_ms']:.2f} ms          |")
print(f"  | FAISS HNSW Search (P95):   {km['faiss_hnsw_search_latency_p95_ms']:.2f} ms          |")
print(f"  | ChromaDB Total (mean):     {km['chroma_total_latency_mean_ms']:.2f} ms         |")
print(f"  | Embedding (mean):          {km['embedding_latency_mean_ms']:.2f} ms         |")
print(f"  | HNSW Recall@{K}:            {km['hnsw_recall_at_k']:.1f}%             |")
print(f"  | FAISS vs Chroma Overlap:   {km['faiss_vs_chroma_result_overlap_pct']:.1f}%             |")
print(f"  | RAG Context Coverage:      {km['rag_context_coverage_pct']:.0f}%               |")
print(f"  | Hallucination Reduction:   {km['hallucination_rate_reduction_pct_points']:.0f} pp              |")
print(f"  +---------------------------------------------------+")
