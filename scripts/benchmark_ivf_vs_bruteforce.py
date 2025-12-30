#!/usr/bin/env python3
"""
Benchmark IVF (approximate) vs Brute Force (exact) search.

Compares:
- Query throughput (QPS)
- Recall (accuracy)
- Latency distribution

Supports different vector types: float32, int8, bit
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXT_PATH = Path(__file__).parent.parent / "dist" / "vec0"


@dataclass
class BenchmarkResult:
    name: str
    n_vectors: int
    dimensions: int
    n_queries: int
    k: int
    total_time: float
    qps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    recall: float = 1.0  # Only meaningful for IVF
    dtype: str = "float"


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(str(EXT_PATH))
    db.enable_load_extension(False)
    return db


def generate_clustered_vectors(n_vectors: int, dimensions: int, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Generate vectors clustered around n_clusters centers."""
    np.random.seed(seed)
    vectors = []
    vectors_per_cluster = n_vectors // n_clusters

    for _ in range(n_clusters):
        center = np.random.randn(dimensions).astype(np.float32) * 10
        cluster_vecs = center + np.random.randn(vectors_per_cluster, dimensions).astype(np.float32) * 0.5
        vectors.append(cluster_vecs)

    # Handle remainder
    remaining = n_vectors - (vectors_per_cluster * n_clusters)
    if remaining > 0:
        center = np.random.randn(dimensions).astype(np.float32) * 10
        cluster_vecs = center + np.random.randn(remaining, dimensions).astype(np.float32) * 0.5
        vectors.append(cluster_vecs)

    return np.vstack(vectors)


def convert_to_int8(vectors: np.ndarray) -> np.ndarray:
    """Convert float32 vectors to int8 by scaling to [-127, 127]."""
    # Normalize to [-1, 1] then scale to int8 range
    min_val = vectors.min()
    max_val = vectors.max()
    if max_val - min_val > 0:
        normalized = 2.0 * (vectors - min_val) / (max_val - min_val) - 1.0
    else:
        normalized = np.zeros_like(vectors)
    return (normalized * 127).astype(np.int8)


def convert_to_bit(vectors: np.ndarray) -> np.ndarray:
    """Convert float32 vectors to bit vectors (packed uint8)."""
    # Binarize based on sign, pack bits into bytes
    binary = (vectors > 0).astype(np.uint8)
    n_vectors, dimensions = vectors.shape
    # Dimensions must be multiple of 8 for bit vectors
    assert dimensions % 8 == 0, "Dimensions must be multiple of 8 for bit vectors"
    packed = np.packbits(binary, axis=1)
    return packed


def get_insert_sql(dtype: str) -> str:
    """Get the INSERT SQL statement for the given dtype."""
    if dtype == "float":
        return "INSERT INTO v(rowid, embedding) VALUES (?, ?)"
    elif dtype == "int8":
        return "INSERT INTO v(rowid, embedding) VALUES (?, vec_int8(?))"
    elif dtype == "bit":
        return "INSERT INTO v(rowid, embedding) VALUES (?, vec_bit(?))"
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_query_sql(dtype: str) -> str:
    """Get the query SQL for the given dtype."""
    if dtype == "float":
        return "SELECT rowid FROM v WHERE embedding MATCH ? AND k=?"
    elif dtype == "int8":
        return "SELECT rowid FROM v WHERE embedding MATCH vec_int8(?) AND k=?"
    elif dtype == "bit":
        return "SELECT rowid FROM v WHERE embedding MATCH vec_bit(?) AND k=?"
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def benchmark_brute_force(
    vectors: np.ndarray, query_vectors: np.ndarray, k: int, dtype: str = "float"
) -> tuple[BenchmarkResult, list[list[int]]]:
    """Benchmark brute force (exact) search."""
    db = connect()
    n_vectors, dimensions = vectors.shape
    n_queries = len(query_vectors)

    # For bit vectors, dimensions is the packed size, but we report the original bit count
    if dtype == "bit":
        reported_dims = dimensions * 8
    else:
        reported_dims = dimensions

    # Create table with appropriate type
    db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding {dtype}[{reported_dims}])")

    # Insert vectors with appropriate SQL
    insert_sql = get_insert_sql(dtype)
    for i, vec in enumerate(vectors):
        db.execute(insert_sql, (i + 1, vec.tobytes()))

    # Get query SQL for this dtype
    query_sql = get_query_sql(dtype)

    # Warm up
    for i in range(min(5, n_queries)):
        db.execute(query_sql, (query_vectors[i].tobytes(), k)).fetchall()

    # Benchmark
    latencies = []
    all_results = []

    for query in query_vectors:
        start = time.perf_counter()
        results = db.execute(query_sql, (query.tobytes(), k)).fetchall()
        latencies.append(time.perf_counter() - start)
        all_results.append([r[0] for r in results])

    total_time = sum(latencies)
    latencies_ms = [lat * 1000 for lat in latencies]

    return BenchmarkResult(
        name=f"Brute Force ({dtype})",
        n_vectors=n_vectors,
        dimensions=reported_dims,
        n_queries=n_queries,
        k=k,
        total_time=total_time,
        qps=n_queries / total_time,
        avg_latency_ms=np.mean(latencies_ms),
        p50_latency_ms=np.percentile(latencies_ms, 50),
        p99_latency_ms=np.percentile(latencies_ms, 99),
        recall=1.0,
        dtype=dtype,
    ), all_results


def benchmark_ivf(
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    nlist: int,
    nprobe: int,
    ground_truth: list[list[int]],
    dtype: str = "float",
) -> BenchmarkResult:
    """Benchmark IVF (approximate) search."""
    db = connect()
    n_vectors, dimensions = vectors.shape
    n_queries = len(query_vectors)

    # For bit vectors, dimensions is the packed size, but we report the original bit count
    if dtype == "bit":
        reported_dims = dimensions * 8
    else:
        reported_dims = dimensions

    # Create IVF table with appropriate type
    db.execute(
        f"CREATE VIRTUAL TABLE v USING vec0(embedding {dtype}[{reported_dims}], ivf_nlist={nlist}, ivf_nprobe={nprobe})"
    )

    # Insert vectors with appropriate SQL
    insert_sql = get_insert_sql(dtype)
    for i, vec in enumerate(vectors):
        db.execute(insert_sql, (i + 1, vec.tobytes()))

    # Train IVF
    db.execute("SELECT vec0_ivf_train('v')")

    # Get query SQL for this dtype
    query_sql = get_query_sql(dtype)

    # Warm up
    for i in range(min(5, n_queries)):
        db.execute(query_sql, (query_vectors[i].tobytes(), k)).fetchall()

    # Benchmark
    latencies = []
    recalls = []

    for i, query in enumerate(query_vectors):
        start = time.perf_counter()
        results = db.execute(query_sql, (query.tobytes(), k)).fetchall()
        latencies.append(time.perf_counter() - start)

        # Calculate recall
        ivf_ids = set(r[0] for r in results)
        gt_ids = set(ground_truth[i][:k])
        recall = len(ivf_ids & gt_ids) / k if k > 0 else 1.0
        recalls.append(recall)

    total_time = sum(latencies)
    latencies_ms = [lat * 1000 for lat in latencies]

    return BenchmarkResult(
        name=f"IVF (nlist={nlist}, nprobe={nprobe})",
        n_vectors=n_vectors,
        dimensions=reported_dims,
        n_queries=n_queries,
        k=k,
        total_time=total_time,
        qps=n_queries / total_time,
        avg_latency_ms=np.mean(latencies_ms),
        p50_latency_ms=np.percentile(latencies_ms, 50),
        p99_latency_ms=np.percentile(latencies_ms, 99),
        recall=np.mean(recalls),
        dtype=dtype,
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result in a formatted way."""
    print(f"\n{'=' * 60}")
    print(f"  {result.name}")
    print(f"{'=' * 60}")
    print(f"  Dataset: {result.n_vectors:,} vectors × {result.dimensions} dimensions ({result.dtype})")
    print(f"  Queries: {result.n_queries} queries, k={result.k}")
    print(f"  {'─' * 56}")
    print(f"  Throughput:    {result.qps:,.1f} QPS")
    print(f"  Latency (avg): {result.avg_latency_ms:.2f} ms")
    print(f"  Latency (p50): {result.p50_latency_ms:.2f} ms")
    print(f"  Latency (p99): {result.p99_latency_ms:.2f} ms")
    if "Brute Force" not in result.name:
        print(f"  Recall:        {result.recall:.1%}")
    print(f"{'=' * 60}")


def run_dtype_benchmark(n_vectors: int, dimensions: int, k: int, dtype: str, n_queries: int = 100):
    """Run benchmark for a specific data type."""
    print(f"\n\n{'#' * 60}")
    print(f"  Config: {n_vectors:,} vectors, {dimensions}D, k={k}, dtype={dtype}")
    print(f"{'#' * 60}")

    # Generate base data (float32)
    n_clusters = min(64, n_vectors // 100)
    print(f"\nGenerating {n_vectors:,} clustered vectors...")
    vectors_float = generate_clustered_vectors(n_vectors, dimensions, n_clusters)

    # Generate query vectors (random subset + some new vectors)
    np.random.seed(123)
    query_indices = np.random.choice(n_vectors, n_queries // 2, replace=False)
    query_vectors_existing = vectors_float[query_indices]
    query_vectors_new = np.random.randn(n_queries // 2, dimensions).astype(np.float32)
    query_vectors_float = np.vstack([query_vectors_existing, query_vectors_new])

    # Convert to target dtype
    if dtype == "float":
        vectors = vectors_float
        query_vectors = query_vectors_float
    elif dtype == "int8":
        # Convert to int8 - need to convert queries using same scale as vectors
        all_data = np.vstack([vectors_float, query_vectors_float])
        min_val = all_data.min()
        max_val = all_data.max()
        if max_val - min_val > 0:
            normalized = 2.0 * (all_data - min_val) / (max_val - min_val) - 1.0
        else:
            normalized = np.zeros_like(all_data)
        all_int8 = (normalized * 127).astype(np.int8)
        vectors = all_int8[:n_vectors]
        query_vectors = all_int8[n_vectors:]
    elif dtype == "bit":
        vectors = convert_to_bit(vectors_float)
        query_vectors = convert_to_bit(query_vectors_float)
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Benchmark brute force (ground truth)
    print(f"\nBenchmarking Brute Force ({dtype})...")
    bf_result, ground_truth = benchmark_brute_force(vectors, query_vectors, k, dtype)
    print_result(bf_result)

    # Benchmark IVF with different nprobe values
    nlist = min(64, n_vectors // 100)  # ~100 vectors per list

    for nprobe in [1, 4, 16]:
        if nprobe > nlist:
            continue
        print(f"\nBenchmarking IVF ({dtype}, nlist={nlist}, nprobe={nprobe})...")
        ivf_result = benchmark_ivf(vectors, query_vectors, k, nlist, nprobe, ground_truth, dtype)
        print_result(ivf_result)

        # Show speedup
        speedup = ivf_result.qps / bf_result.qps
        print(f"  Speedup vs Brute Force: {speedup:.2f}x")

    return bf_result


def main():
    print("=" * 60)
    print("  IVF vs Brute Force Benchmark - Multi-dtype")
    print("=" * 60)

    # Test configurations - reduced for multi-dtype comparison
    n_vectors = 50_000
    dimensions = 128  # Must be multiple of 8 for bit vectors
    k = 10
    n_queries = 100

    # Run benchmarks for each dtype
    results = {}
    for dtype in ["float", "int8", "bit"]:
        results[dtype] = run_dtype_benchmark(n_vectors, dimensions, k, dtype, n_queries)

    # Summary comparison
    print("\n\n" + "=" * 60)
    print("  SUMMARY: Brute Force QPS by Data Type")
    print("=" * 60)
    for dtype, result in results.items():
        print(f"  {dtype:8s}: {result.qps:,.1f} QPS ({result.avg_latency_ms:.2f}ms avg latency)")
    print("=" * 60)
    print("  Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
