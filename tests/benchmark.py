#!/usr/bin/env python3
"""
sqlite-vec Benchmark Suite

Benchmarks brute-force vs IVF approximate nearest neighbor search
with realistic synthetic data that mimics embedding distributions.

Usage:
    python tests/benchmark.py [--quick] [--full]

    --quick: Run minimal benchmarks (default)
    --full:  Run comprehensive benchmarks (slower)
"""

import argparse
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
    k: int
    method: str  # "brute_force" or "ivf"
    ivf_nlist: int = 0
    ivf_nprobe: int = 0
    build_time_ms: float = 0
    train_time_ms: float = 0
    avg_query_time_ms: float = 0
    queries_per_second: float = 0
    recall_at_k: float = 0  # vs brute force ground truth


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(str(EXT_PATH))
    db.enable_load_extension(False)
    return db


def generate_clustered_data(n_vectors: int, dimensions: int, n_clusters: int = 50, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic data that mimics real embedding distributions.
    Creates clustered data which is more realistic than uniform random.
    """
    np.random.seed(seed)

    vectors_per_cluster = n_vectors // n_clusters
    remainder = n_vectors % n_clusters

    vectors = []
    for i in range(n_clusters):
        # Random cluster center
        center = np.random.randn(dimensions).astype(np.float32) * 5

        # Number of vectors in this cluster
        cluster_size = vectors_per_cluster + (1 if i < remainder else 0)

        # Generate vectors around the center with some variance
        cluster_vectors = center + np.random.randn(cluster_size, dimensions).astype(np.float32) * 0.5
        vectors.append(cluster_vectors)

    all_vectors = np.vstack(vectors)
    np.random.shuffle(all_vectors)

    # Normalize to unit length (common for embeddings)
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    all_vectors = all_vectors / norms

    return all_vectors


def generate_queries(data: np.ndarray, n_queries: int, seed: int = 123) -> np.ndarray:
    """Generate query vectors - mix of existing vectors and nearby points."""
    np.random.seed(seed)

    n_exact = n_queries // 3
    n_nearby = n_queries // 3
    n_random = n_queries - n_exact - n_nearby

    queries = []

    # Some queries are exact vectors from the dataset
    indices = np.random.choice(len(data), n_exact, replace=False)
    queries.append(data[indices])

    # Some queries are nearby existing vectors
    indices = np.random.choice(len(data), n_nearby, replace=False)
    nearby = data[indices] + np.random.randn(n_nearby, data.shape[1]).astype(np.float32) * 0.1
    nearby = nearby / np.linalg.norm(nearby, axis=1, keepdims=True)
    queries.append(nearby)

    # Some queries are random
    random_queries = np.random.randn(n_random, data.shape[1]).astype(np.float32)
    random_queries = random_queries / np.linalg.norm(random_queries, axis=1, keepdims=True)
    queries.append(random_queries)

    return np.vstack(queries)


def get_brute_force_results(db: sqlite3.Connection, query: np.ndarray, k: int) -> list[int]:
    """Get ground truth results using brute force."""
    results = db.execute("SELECT rowid FROM v_bf WHERE embedding MATCH ? AND k = ?", (query.tobytes(), k)).fetchall()
    return [r[0] for r in results]


def calculate_recall(retrieved: list[int], ground_truth: list[int]) -> float:
    """Calculate recall@k."""
    if not ground_truth:
        return 1.0
    return len(set(retrieved) & set(ground_truth)) / len(ground_truth)


def benchmark_brute_force(data: np.ndarray, queries: np.ndarray, k: int, name: str = "brute_force") -> BenchmarkResult:
    """Benchmark brute-force KNN search."""
    db = connect()
    dimensions = data.shape[1]

    # Build index
    start = time.perf_counter()
    db.execute(f"CREATE VIRTUAL TABLE v_bf USING vec0(embedding float[{dimensions}])")
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
    build_time = (time.perf_counter() - start) * 1000

    # Run queries
    query_times = []
    for q in queries:
        start = time.perf_counter()
        db.execute("SELECT rowid, distance FROM v_bf WHERE embedding MATCH ? AND k = ?", (q.tobytes(), k)).fetchall()
        query_times.append((time.perf_counter() - start) * 1000)

    avg_query_time = np.mean(query_times)
    qps = 1000 / avg_query_time if avg_query_time > 0 else 0

    db.close()

    return BenchmarkResult(
        name=name,
        n_vectors=len(data),
        dimensions=dimensions,
        k=k,
        method="brute_force",
        build_time_ms=build_time,
        avg_query_time_ms=avg_query_time,
        queries_per_second=qps,
        recall_at_k=1.0,  # Brute force is ground truth
    )


def benchmark_ivf(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    nlist: int,
    nprobe: int,
    ground_truth: list[list[int]] = None,
    name: str = "ivf",
) -> BenchmarkResult:
    """Benchmark IVF approximate KNN search."""
    db = connect()
    dimensions = data.shape[1]

    # Build index
    start = time.perf_counter()
    db.execute(
        f"CREATE VIRTUAL TABLE v_ivf USING vec0(embedding float[{dimensions}], ivf_nlist={nlist}, ivf_nprobe={nprobe})"
    )
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_ivf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
    build_time = (time.perf_counter() - start) * 1000

    # Train IVF
    start = time.perf_counter()
    db.execute("SELECT vec0_ivf_train('v_ivf')")
    train_time = (time.perf_counter() - start) * 1000

    # Run queries and calculate recall
    query_times = []
    recalls = []

    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = db.execute(
            "SELECT rowid, distance FROM v_ivf WHERE embedding MATCH ? AND k = ?", (q.tobytes(), k)
        ).fetchall()
        query_times.append((time.perf_counter() - start) * 1000)

        if ground_truth:
            retrieved = [r[0] for r in results]
            recalls.append(calculate_recall(retrieved, ground_truth[i]))

    avg_query_time = np.mean(query_times)
    qps = 1000 / avg_query_time if avg_query_time > 0 else 0
    avg_recall = np.mean(recalls) if recalls else 0

    db.close()

    return BenchmarkResult(
        name=name,
        n_vectors=len(data),
        dimensions=dimensions,
        k=k,
        method="ivf",
        ivf_nlist=nlist,
        ivf_nprobe=nprobe,
        build_time_ms=build_time,
        train_time_ms=train_time,
        avg_query_time_ms=avg_query_time,
        queries_per_second=qps,
        recall_at_k=avg_recall,
    )


def get_ground_truth(data: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    """Get brute-force ground truth for all queries."""
    db = connect()
    dimensions = data.shape[1]

    db.execute(f"CREATE VIRTUAL TABLE v_gt USING vec0(embedding float[{dimensions}])")
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_gt(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

    ground_truth = []
    for q in queries:
        results = db.execute("SELECT rowid FROM v_gt WHERE embedding MATCH ? AND k = ?", (q.tobytes(), k)).fetchall()
        ground_truth.append([r[0] for r in results])

    db.close()
    return ground_truth


def print_result(result: BenchmarkResult):
    """Print a benchmark result."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {result.name}")
    print(f"{'=' * 60}")
    print(f"  Dataset: {result.n_vectors:,} vectors x {result.dimensions}D")
    print(f"  Method:  {result.method}")
    if result.method == "ivf":
        print(f"  IVF:     nlist={result.ivf_nlist}, nprobe={result.ivf_nprobe}")
    print(f"  k:       {result.k}")
    print(f"  Build:   {result.build_time_ms:.1f} ms")
    if result.train_time_ms > 0:
        print(f"  Train:   {result.train_time_ms:.1f} ms")
    print(f"  Query:   {result.avg_query_time_ms:.3f} ms avg")
    print(f"  QPS:     {result.queries_per_second:.1f} queries/sec")
    if result.method == "ivf":
        print(f"  Recall:  {result.recall_at_k:.1%}")


def run_quick_benchmark():
    """Run a quick benchmark for fast feedback."""
    print("\n" + "=" * 60)
    print("QUICK BENCHMARK")
    print("=" * 60)

    # Parameters
    n_vectors = 10000
    dimensions = 128
    n_queries = 100
    k = 10

    print(f"\nGenerating {n_vectors:,} clustered vectors ({dimensions}D)...")
    data = generate_clustered_data(n_vectors, dimensions)
    queries = generate_queries(data, n_queries)

    print("Computing ground truth...")
    ground_truth = get_ground_truth(data, queries, k)

    # Brute force
    print("\nRunning brute-force benchmark...")
    bf_result = benchmark_brute_force(data, queries, k, "Brute Force (10K vectors)")
    print_result(bf_result)

    # IVF with different nprobe values
    nlist = 100
    for nprobe in [1, 5, 10, 20]:
        print(f"\nRunning IVF benchmark (nprobe={nprobe})...")
        ivf_result = benchmark_ivf(data, queries, k, nlist, nprobe, ground_truth, f"IVF nlist={nlist} nprobe={nprobe}")
        print_result(ivf_result)

    print("\n" + "=" * 60)
    print("QUICK BENCHMARK COMPLETE")
    print("=" * 60)


def run_full_benchmark():
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 60)
    print("FULL BENCHMARK SUITE")
    print("=" * 60)

    results = []

    # Test different dataset sizes
    for n_vectors in [5000, 20000, 50000]:
        dimensions = 128
        n_queries = 100
        k = 10

        print(f"\n{'=' * 60}")
        print(f"Dataset: {n_vectors:,} vectors x {dimensions}D")
        print(f"{'=' * 60}")

        print("Generating data...")
        data = generate_clustered_data(n_vectors, dimensions)
        queries = generate_queries(data, n_queries)

        print("Computing ground truth...")
        ground_truth = get_ground_truth(data, queries, k)

        # Brute force
        print("Running brute-force...")
        bf_result = benchmark_brute_force(data, queries, k, f"BF {n_vectors // 1000}K")
        print_result(bf_result)
        results.append(bf_result)

        # IVF with optimal nlist (sqrt(n) is a common heuristic)
        nlist = max(int(np.sqrt(n_vectors)), 10)

        for nprobe in [1, nlist // 10, nlist // 4]:
            nprobe = max(1, nprobe)
            print(f"Running IVF (nlist={nlist}, nprobe={nprobe})...")
            ivf_result = benchmark_ivf(
                data,
                queries,
                k,
                nlist,
                nprobe,
                ground_truth,
                f"IVF {n_vectors // 1000}K (nlist={nlist}, nprobe={nprobe})",
            )
            print_result(ivf_result)
            results.append(ivf_result)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Name':<40} {'QPS':>10} {'Recall':>10} {'Query ms':>10}")
    print("-" * 70)
    for r in results:
        recall_str = f"{r.recall_at_k:.1%}" if r.method == "ivf" else "100%"
        print(f"{r.name:<40} {r.queries_per_second:>10.1f} {recall_str:>10} {r.avg_query_time_ms:>10.3f}")

    print("\n" + "=" * 60)
    print("FULL BENCHMARK COMPLETE")
    print("=" * 60)


def run_dimension_benchmark():
    """Benchmark different dimension sizes."""
    print("\n" + "=" * 60)
    print("DIMENSION BENCHMARK")
    print("=" * 60)

    n_vectors = 10000
    n_queries = 50
    k = 10

    results = []

    for dimensions in [32, 128, 384, 768]:
        print(f"\n{'=' * 60}")
        print(f"Dimensions: {dimensions}")
        print(f"{'=' * 60}")

        data = generate_clustered_data(n_vectors, dimensions)
        queries = generate_queries(data, n_queries)
        ground_truth = get_ground_truth(data, queries, k)

        # Brute force
        bf_result = benchmark_brute_force(data, queries, k, f"BF {dimensions}D")
        print_result(bf_result)
        results.append(bf_result)

        # IVF
        nlist = 100
        nprobe = 10
        ivf_result = benchmark_ivf(data, queries, k, nlist, nprobe, ground_truth, f"IVF {dimensions}D")
        print_result(ivf_result)
        results.append(ivf_result)

    # Summary
    print("\n" + "=" * 60)
    print("DIMENSION COMPARISON")
    print("=" * 60)
    print(f"{'Dimensions':<15} {'BF QPS':>10} {'IVF QPS':>10} {'Speedup':>10} {'Recall':>10}")
    print("-" * 55)

    for i in range(0, len(results), 2):
        bf = results[i]
        ivf = results[i + 1]
        dims = bf.dimensions
        speedup = ivf.queries_per_second / bf.queries_per_second if bf.queries_per_second > 0 else 0
        print(
            f"{dims:<15} {bf.queries_per_second:>10.1f} {ivf.queries_per_second:>10.1f} {speedup:>10.2f}x {ivf.recall_at_k:>10.1%}"
        )


def generate_int8_data(n_vectors: int, dimensions: int, seed: int = 42) -> np.ndarray:
    """Generate int8 vector data."""
    np.random.seed(seed)
    # Generate clustered int8 vectors (-128 to 127)
    n_clusters = min(50, n_vectors // 10)
    vectors_per_cluster = n_vectors // n_clusters
    remainder = n_vectors % n_clusters

    vectors = []
    for i in range(n_clusters):
        # Random cluster center in int8 range
        center = np.random.randint(-100, 100, dimensions)
        cluster_size = vectors_per_cluster + (1 if i < remainder else 0)

        # Generate vectors around the center with variance
        noise = np.random.randint(-30, 30, (cluster_size, dimensions))
        cluster_vectors = np.clip(center + noise, -128, 127).astype(np.int8)
        vectors.append(cluster_vectors)

    all_vectors = np.vstack(vectors)
    np.random.shuffle(all_vectors)
    return all_vectors


def generate_bit_data(n_vectors: int, dimensions: int, seed: int = 42) -> np.ndarray:
    """Generate bit vector data as packed bytes."""
    np.random.seed(seed)
    # Dimensions must be divisible by 8 for bit vectors
    assert dimensions % 8 == 0, "Bit vector dimensions must be divisible by 8"

    n_bytes = dimensions // 8

    # Generate clustered bit vectors
    n_clusters = min(20, n_vectors // 5)
    vectors_per_cluster = n_vectors // n_clusters
    remainder = n_vectors % n_clusters

    vectors = []
    for i in range(n_clusters):
        # Random cluster center
        center = np.random.randint(0, 256, n_bytes, dtype=np.uint8)
        cluster_size = vectors_per_cluster + (1 if i < remainder else 0)

        # Generate vectors by flipping random bits from center
        for _ in range(cluster_size):
            vec = center.copy()
            # Flip ~10% of bits randomly
            n_flips = max(1, n_bytes // 10)
            flip_bytes = np.random.choice(n_bytes, n_flips, replace=False)
            flip_bits = np.random.randint(0, 8, n_flips)
            for byte_idx, bit_idx in zip(flip_bytes, flip_bits):
                vec[byte_idx] ^= 1 << bit_idx
            vectors.append(vec)

    all_vectors = np.array(vectors, dtype=np.uint8)
    np.random.shuffle(all_vectors)
    return all_vectors


def benchmark_int8_brute_force(
    data: np.ndarray, queries: np.ndarray, k: int, name: str = "int8_brute_force"
) -> BenchmarkResult:
    """Benchmark int8 brute-force KNN search."""
    db = connect()
    dimensions = data.shape[1]

    # Build index
    start = time.perf_counter()
    db.execute(f"CREATE VIRTUAL TABLE v_bf USING vec0(embedding int8[{dimensions}])")
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))
    build_time = (time.perf_counter() - start) * 1000

    # Run queries
    query_times = []
    for q in queries:
        start = time.perf_counter()
        db.execute(
            "SELECT rowid, distance FROM v_bf WHERE embedding MATCH vec_int8(?) AND k = ?", (q.tobytes(), k)
        ).fetchall()
        query_times.append((time.perf_counter() - start) * 1000)

    avg_query_time = np.mean(query_times)
    qps = 1000 / avg_query_time if avg_query_time > 0 else 0

    db.close()

    return BenchmarkResult(
        name=name,
        n_vectors=len(data),
        dimensions=dimensions,
        k=k,
        method="brute_force",
        build_time_ms=build_time,
        avg_query_time_ms=avg_query_time,
        queries_per_second=qps,
        recall_at_k=1.0,
    )


def benchmark_int8_ivf(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    nlist: int,
    nprobe: int,
    ground_truth: list[list[int]] = None,
    name: str = "int8_ivf",
) -> BenchmarkResult:
    """Benchmark int8 IVF approximate KNN search."""
    db = connect()
    dimensions = data.shape[1]

    # Build index
    start = time.perf_counter()
    db.execute(
        f"CREATE VIRTUAL TABLE v_ivf USING vec0(embedding int8[{dimensions}], ivf_nlist={nlist}, ivf_nprobe={nprobe})"
    )
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_ivf(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))
    build_time = (time.perf_counter() - start) * 1000

    # Train IVF
    start = time.perf_counter()
    db.execute("SELECT vec0_ivf_train('v_ivf')")
    train_time = (time.perf_counter() - start) * 1000

    # Run queries and calculate recall
    query_times = []
    recalls = []

    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = db.execute(
            "SELECT rowid, distance FROM v_ivf WHERE embedding MATCH vec_int8(?) AND k = ?", (q.tobytes(), k)
        ).fetchall()
        query_times.append((time.perf_counter() - start) * 1000)

        if ground_truth:
            retrieved = [r[0] for r in results]
            recalls.append(calculate_recall(retrieved, ground_truth[i]))

    avg_query_time = np.mean(query_times)
    qps = 1000 / avg_query_time if avg_query_time > 0 else 0
    avg_recall = np.mean(recalls) if recalls else 0

    db.close()

    return BenchmarkResult(
        name=name,
        n_vectors=len(data),
        dimensions=dimensions,
        k=k,
        method="ivf",
        ivf_nlist=nlist,
        ivf_nprobe=nprobe,
        build_time_ms=build_time,
        train_time_ms=train_time,
        avg_query_time_ms=avg_query_time,
        queries_per_second=qps,
        recall_at_k=avg_recall,
    )


def benchmark_bit_brute_force(
    data: np.ndarray, queries: np.ndarray, k: int, name: str = "bit_brute_force"
) -> BenchmarkResult:
    """Benchmark bit vector brute-force KNN search."""
    db = connect()
    # For bit vectors, dimensions is in bits, but data is packed bytes
    dimensions = data.shape[1] * 8

    # Build index
    start = time.perf_counter()
    db.execute(f"CREATE VIRTUAL TABLE v_bf USING vec0(embedding bit[{dimensions}])")
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec.tobytes()))
    build_time = (time.perf_counter() - start) * 1000

    # Run queries
    query_times = []
    for q in queries:
        start = time.perf_counter()
        db.execute(
            "SELECT rowid, distance FROM v_bf WHERE embedding MATCH vec_bit(?) AND k = ?", (q.tobytes(), k)
        ).fetchall()
        query_times.append((time.perf_counter() - start) * 1000)

    avg_query_time = np.mean(query_times)
    qps = 1000 / avg_query_time if avg_query_time > 0 else 0

    db.close()

    return BenchmarkResult(
        name=name,
        n_vectors=len(data),
        dimensions=dimensions,
        k=k,
        method="brute_force",
        build_time_ms=build_time,
        avg_query_time_ms=avg_query_time,
        queries_per_second=qps,
        recall_at_k=1.0,
    )


def benchmark_bit_ivf(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    nlist: int,
    nprobe: int,
    ground_truth: list[list[int]] = None,
    name: str = "bit_ivf",
) -> BenchmarkResult:
    """Benchmark bit vector IVF approximate KNN search."""
    db = connect()
    dimensions = data.shape[1] * 8

    # Build index
    start = time.perf_counter()
    db.execute(
        f"CREATE VIRTUAL TABLE v_ivf USING vec0(embedding bit[{dimensions}], ivf_nlist={nlist}, ivf_nprobe={nprobe})"
    )
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_ivf(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec.tobytes()))
    build_time = (time.perf_counter() - start) * 1000

    # Train IVF
    start = time.perf_counter()
    db.execute("SELECT vec0_ivf_train('v_ivf')")
    train_time = (time.perf_counter() - start) * 1000

    # Run queries and calculate recall
    query_times = []
    recalls = []

    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = db.execute(
            "SELECT rowid, distance FROM v_ivf WHERE embedding MATCH vec_bit(?) AND k = ?", (q.tobytes(), k)
        ).fetchall()
        query_times.append((time.perf_counter() - start) * 1000)

        if ground_truth:
            retrieved = [r[0] for r in results]
            recalls.append(calculate_recall(retrieved, ground_truth[i]))

    avg_query_time = np.mean(query_times)
    qps = 1000 / avg_query_time if avg_query_time > 0 else 0
    avg_recall = np.mean(recalls) if recalls else 0

    db.close()

    return BenchmarkResult(
        name=name,
        n_vectors=len(data),
        dimensions=dimensions,
        k=k,
        method="ivf",
        ivf_nlist=nlist,
        ivf_nprobe=nprobe,
        build_time_ms=build_time,
        train_time_ms=train_time,
        avg_query_time_ms=avg_query_time,
        queries_per_second=qps,
        recall_at_k=avg_recall,
    )


def get_int8_ground_truth(data: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    """Get brute-force ground truth for int8 queries."""
    db = connect()
    dimensions = data.shape[1]

    db.execute(f"CREATE VIRTUAL TABLE v_gt USING vec0(embedding int8[{dimensions}])")
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_gt(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))

    ground_truth = []
    for q in queries:
        results = db.execute(
            "SELECT rowid FROM v_gt WHERE embedding MATCH vec_int8(?) AND k = ?", (q.tobytes(), k)
        ).fetchall()
        ground_truth.append([r[0] for r in results])

    db.close()
    return ground_truth


def get_bit_ground_truth(data: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    """Get brute-force ground truth for bit queries."""
    db = connect()
    dimensions = data.shape[1] * 8

    db.execute(f"CREATE VIRTUAL TABLE v_gt USING vec0(embedding bit[{dimensions}])")
    for i, vec in enumerate(data):
        db.execute("INSERT INTO v_gt(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec.tobytes()))

    ground_truth = []
    for q in queries:
        results = db.execute(
            "SELECT rowid FROM v_gt WHERE embedding MATCH vec_bit(?) AND k = ?", (q.tobytes(), k)
        ).fetchall()
        ground_truth.append([r[0] for r in results])

    db.close()
    return ground_truth


def run_int8_benchmark():
    """Benchmark int8 vector performance."""
    print("\n" + "=" * 60)
    print("INT8 VECTOR BENCHMARK")
    print("=" * 60)

    n_vectors = 10000
    dimensions = 128
    n_queries = 100
    k = 10

    print(f"\nGenerating {n_vectors:,} int8 vectors ({dimensions}D)...")
    data = generate_int8_data(n_vectors, dimensions)
    # Use same vectors as queries for int8
    query_indices = np.random.choice(len(data), n_queries, replace=False)
    queries = data[query_indices]

    print("Computing ground truth...")
    ground_truth = get_int8_ground_truth(data, queries, k)

    # Brute force
    print("\nRunning int8 brute-force benchmark...")
    bf_result = benchmark_int8_brute_force(data, queries, k, "Int8 Brute Force")
    print_result(bf_result)

    # IVF with different nprobe values
    nlist = 100
    results = [bf_result]
    for nprobe in [1, 5, 10, 20]:
        print(f"\nRunning int8 IVF benchmark (nprobe={nprobe})...")
        ivf_result = benchmark_int8_ivf(
            data, queries, k, nlist, nprobe, ground_truth, f"Int8 IVF nlist={nlist} nprobe={nprobe}"
        )
        print_result(ivf_result)
        results.append(ivf_result)

    # Summary
    print("\n" + "=" * 60)
    print("INT8 SUMMARY")
    print("=" * 60)
    print(f"{'Name':<40} {'QPS':>10} {'Recall':>10} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        recall_str = f"{r.recall_at_k:.1%}" if r.method == "ivf" else "100%"
        speedup = r.queries_per_second / bf_result.queries_per_second if bf_result.queries_per_second > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r.method == "ivf" else "-"
        print(f"{r.name:<40} {r.queries_per_second:>10.1f} {recall_str:>10} {speedup_str:>10}")


def run_bit_benchmark():
    """Benchmark bit vector performance."""
    print("\n" + "=" * 60)
    print("BIT VECTOR BENCHMARK")
    print("=" * 60)

    n_vectors = 10000
    dimensions = 256  # bits (32 bytes)
    n_queries = 100
    k = 10

    print(f"\nGenerating {n_vectors:,} bit vectors ({dimensions} bits)...")
    data = generate_bit_data(n_vectors, dimensions)
    # Use same vectors as queries
    query_indices = np.random.choice(len(data), n_queries, replace=False)
    queries = data[query_indices]

    print("Computing ground truth...")
    ground_truth = get_bit_ground_truth(data, queries, k)

    # Brute force
    print("\nRunning bit brute-force benchmark...")
    bf_result = benchmark_bit_brute_force(data, queries, k, "Bit Brute Force")
    print_result(bf_result)

    # IVF with different nprobe values
    nlist = 50
    results = [bf_result]
    for nprobe in [1, 5, 10, 20]:
        print(f"\nRunning bit IVF benchmark (nprobe={nprobe})...")
        ivf_result = benchmark_bit_ivf(
            data, queries, k, nlist, nprobe, ground_truth, f"Bit IVF nlist={nlist} nprobe={nprobe}"
        )
        print_result(ivf_result)
        results.append(ivf_result)

    # Summary
    print("\n" + "=" * 60)
    print("BIT SUMMARY")
    print("=" * 60)
    print(f"{'Name':<40} {'QPS':>10} {'Recall':>10} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        recall_str = f"{r.recall_at_k:.1%}" if r.method == "ivf" else "100%"
        speedup = r.queries_per_second / bf_result.queries_per_second if bf_result.queries_per_second > 0 else 0
        speedup_str = f"{speedup:.2f}x" if r.method == "ivf" else "-"
        print(f"{r.name:<40} {r.queries_per_second:>10.1f} {recall_str:>10} {speedup_str:>10}")


def run_vector_type_comparison():
    """Compare float32, int8, and bit vector performance."""
    print("\n" + "=" * 60)
    print("VECTOR TYPE COMPARISON")
    print("=" * 60)

    n_vectors = 10000
    dimensions = 128
    n_queries = 100
    k = 10
    nlist = 100
    nprobe = 10

    results = []

    # Float32
    print(f"\n--- Float32 ({dimensions}D) ---")
    float_data = generate_clustered_data(n_vectors, dimensions)
    float_queries = generate_queries(float_data, n_queries)
    float_gt = get_ground_truth(float_data, float_queries, k)

    print("Brute force...")
    bf_float = benchmark_brute_force(float_data, float_queries, k, "Float32 BF")
    results.append(("Float32", "BF", bf_float))

    print("IVF...")
    ivf_float = benchmark_ivf(float_data, float_queries, k, nlist, nprobe, float_gt, "Float32 IVF")
    results.append(("Float32", "IVF", ivf_float))

    # Int8
    print(f"\n--- Int8 ({dimensions}D) ---")
    int8_data = generate_int8_data(n_vectors, dimensions)
    int8_queries = int8_data[np.random.choice(len(int8_data), n_queries, replace=False)]
    int8_gt = get_int8_ground_truth(int8_data, int8_queries, k)

    print("Brute force...")
    bf_int8 = benchmark_int8_brute_force(int8_data, int8_queries, k, "Int8 BF")
    results.append(("Int8", "BF", bf_int8))

    print("IVF...")
    ivf_int8 = benchmark_int8_ivf(int8_data, int8_queries, k, nlist, nprobe, int8_gt, "Int8 IVF")
    results.append(("Int8", "IVF", ivf_int8))

    # Bit (use 256 bits for comparison - 32 bytes)
    bit_dims = 256
    print(f"\n--- Bit ({bit_dims} bits) ---")
    bit_data = generate_bit_data(n_vectors, bit_dims)
    bit_queries = bit_data[np.random.choice(len(bit_data), n_queries, replace=False)]
    bit_gt = get_bit_ground_truth(bit_data, bit_queries, k)

    print("Brute force...")
    bf_bit = benchmark_bit_brute_force(bit_data, bit_queries, k, "Bit BF")
    results.append(("Bit", "BF", bf_bit))

    print("IVF...")
    ivf_bit = benchmark_bit_ivf(bit_data, bit_queries, k, 50, nprobe, bit_gt, "Bit IVF")
    results.append(("Bit", "IVF", ivf_bit))

    # Summary table
    print("\n" + "=" * 60)
    print("VECTOR TYPE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Type':<10} {'Method':<6} {'Build ms':>10} {'Train ms':>10} {'Query ms':>10} {'QPS':>10} {'Recall':>8}")
    print("-" * 74)
    for type_name, method, r in results:
        train_str = f"{r.train_time_ms:.1f}" if r.train_time_ms > 0 else "-"
        recall_str = f"{r.recall_at_k:.1%}" if r.method == "ivf" else "100%"
        print(
            f"{type_name:<10} {method:<6} {r.build_time_ms:>10.1f} {train_str:>10} {r.avg_query_time_ms:>10.3f} {r.queries_per_second:>10.1f} {recall_str:>8}"
        )

    # Storage comparison
    print("\n" + "=" * 60)
    print("STORAGE COMPARISON (per vector)")
    print("=" * 60)
    print(f"{'Type':<10} {'Dimensions':>12} {'Bytes/vec':>12} {'Relative':>10}")
    print("-" * 46)
    print(f"{'Float32':<10} {dimensions:>12} {dimensions * 4:>12} {'1.00x':>10}")
    print(f"{'Int8':<10} {dimensions:>12} {dimensions:>12} {'0.25x':>10}")
    print(f"{'Bit':<10} {bit_dims:>12} {bit_dims // 8:>12} {f'{(bit_dims // 8) / (dimensions * 4):.2f}x':>10}")


def run_memory_benchmark():
    """Benchmark memory usage and training time for different dataset sizes."""
    print("\n" + "=" * 60)
    print("MEMORY AND TRAINING TIME BENCHMARK")
    print("=" * 60)

    import tracemalloc

    dimensions = 128
    k = 10
    n_queries = 50
    nlist = 100
    nprobe = 10

    results = []

    for n_vectors in [1000, 5000, 10000, 25000]:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {n_vectors:,} vectors x {dimensions}D")
        print(f"{'=' * 60}")

        data = generate_clustered_data(n_vectors, dimensions)
        queries = generate_queries(data, n_queries)

        # Track memory for brute force
        tracemalloc.start()
        db_bf = connect()
        db_bf.execute(f"CREATE VIRTUAL TABLE v_bf USING vec0(embedding float[{dimensions}])")
        for i, vec in enumerate(data):
            db_bf.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
        bf_current, bf_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        db_bf.close()

        # Track memory and time for IVF
        tracemalloc.start()
        db_ivf = connect()
        db_ivf.execute(
            f"CREATE VIRTUAL TABLE v_ivf USING vec0(embedding float[{dimensions}], ivf_nlist={nlist}, ivf_nprobe={nprobe})"
        )
        for i, vec in enumerate(data):
            db_ivf.execute("INSERT INTO v_ivf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        start = time.perf_counter()
        db_ivf.execute("SELECT vec0_ivf_train('v_ivf')")
        train_time = (time.perf_counter() - start) * 1000

        ivf_current, ivf_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        db_ivf.close()

        print("  Brute Force:")
        print(f"    Peak memory: {bf_peak / (1024 * 1024):.2f} MB")
        print("  IVF:")
        print(f"    Train time:  {train_time:.1f} ms")
        print(f"    Peak memory: {ivf_peak / (1024 * 1024):.2f} MB")
        print(f"    Overhead:    {(ivf_peak - bf_peak) / (1024 * 1024):.2f} MB")

        results.append(
            {
                "n_vectors": n_vectors,
                "bf_peak_mb": bf_peak / (1024 * 1024),
                "ivf_peak_mb": ivf_peak / (1024 * 1024),
                "train_time_ms": train_time,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("MEMORY AND TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Vectors':>10} {'BF Memory':>12} {'IVF Memory':>12} {'Overhead':>10} {'Train ms':>10}")
    print("-" * 54)
    for r in results:
        overhead = r["ivf_peak_mb"] - r["bf_peak_mb"]
        print(
            f"{r['n_vectors']:>10,} {r['bf_peak_mb']:>11.2f}M {r['ivf_peak_mb']:>11.2f}M {overhead:>9.2f}M {r['train_time_ms']:>10.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sqlite-vec benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (default)")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--dimensions", action="store_true", help="Run dimension comparison")
    parser.add_argument("--int8", action="store_true", help="Run int8 vector benchmark")
    parser.add_argument("--bit", action="store_true", help="Run bit vector benchmark")
    parser.add_argument("--types", action="store_true", help="Compare all vector types")
    parser.add_argument("--memory", action="store_true", help="Run memory and training benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all:
        run_quick_benchmark()
        run_int8_benchmark()
        run_bit_benchmark()
        run_vector_type_comparison()
        run_memory_benchmark()
    elif args.full:
        run_full_benchmark()
    elif args.dimensions:
        run_dimension_benchmark()
    elif args.int8:
        run_int8_benchmark()
    elif args.bit:
        run_bit_benchmark()
    elif args.types:
        run_vector_type_comparison()
    elif args.memory:
        run_memory_benchmark()
    else:
        run_quick_benchmark()
