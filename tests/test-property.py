"""
Property-based tests for sqlite-vec using Hypothesis.

These tests verify invariants that should hold for any valid input,
helping discover edge cases that may not be covered by example-based tests.
"""

import sqlite3
from pathlib import Path

import numpy as np
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

EXT_PATH = Path(__file__).parent.parent / "dist" / "vec0"


def connect():
    """Create a connection with sqlite-vec loaded."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(str(EXT_PATH))
    db.enable_load_extension(False)
    return db


def to_blob(arr):
    """Convert numpy array to blob for sqlite-vec."""
    return arr.astype(np.float32).tobytes()


# ============================================================================
# Strategy definitions
# ============================================================================

# Reasonable dimension range (not too small, not too large)
dimensions = st.integers(min_value=2, max_value=128)

# Finite floats only (no inf/nan for most tests)
finite_floats = st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False)


def vector_of_dim(dim):
    """Generate a random vector of given dimension."""
    return st.lists(finite_floats, min_size=dim, max_size=dim).map(lambda v: np.array(v, dtype=np.float32))


# ============================================================================
# Distance function properties
# ============================================================================


class TestDistanceProperties:
    """Property tests for distance calculations."""

    @given(dim=dimensions)
    @settings(max_examples=20, deadline=5000)
    def test_distance_to_self_is_zero(self, dim):
        """Distance from any vector to itself should be 0."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        vec = np.random.randn(dim).astype(np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (to_blob(vec),))

        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=1", (to_blob(vec),)
        ).fetchall()

        assert len(results) == 1
        assert results[0][0] == 1
        # Distance to self should be 0 (or very close due to float precision)
        assert results[0][1] < 1e-6

    @given(dim=dimensions)
    @settings(max_examples=20, deadline=5000)
    def test_distance_is_nonnegative(self, dim):
        """Squared L2 distance is always non-negative."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        for i in range(5):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)
        ).fetchall()

        for rowid, distance in results:
            assert distance >= 0, f"Distance {distance} is negative"

    @given(dim=dimensions)
    @settings(max_examples=20, deadline=5000)
    def test_distance_symmetry(self, dim):
        """Distance(A, B) should equal Distance(B, A)."""
        db = connect()

        vec_a = np.random.randn(dim).astype(np.float32)
        vec_b = np.random.randn(dim).astype(np.float32)

        # Query A with B
        db.execute(f"CREATE VIRTUAL TABLE v1 USING vec0(embedding float[{dim}])")
        db.execute("INSERT INTO v1(rowid, embedding) VALUES (1, ?)", (to_blob(vec_a),))
        dist_ab = db.execute("SELECT distance FROM v1 WHERE embedding MATCH ? AND k=1", (to_blob(vec_b),)).fetchone()[0]

        # Query B with A
        db.execute(f"CREATE VIRTUAL TABLE v2 USING vec0(embedding float[{dim}])")
        db.execute("INSERT INTO v2(rowid, embedding) VALUES (1, ?)", (to_blob(vec_b),))
        dist_ba = db.execute("SELECT distance FROM v2 WHERE embedding MATCH ? AND k=1", (to_blob(vec_a),)).fetchone()[0]

        # Should be equal (within floating point tolerance)
        assert abs(dist_ab - dist_ba) < 1e-5, f"Asymmetric: {dist_ab} vs {dist_ba}"


# ============================================================================
# KNN result properties
# ============================================================================


class TestKNNProperties:
    """Property tests for KNN query results."""

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_vectors=st.integers(min_value=5, max_value=50),
        k=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30, deadline=10000)
    def test_knn_returns_at_most_k(self, dim, n_vectors, k):
        """KNN should return at most k results."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(f"SELECT rowid FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)).fetchall()

        expected_count = min(k, n_vectors)
        assert len(results) == expected_count

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_vectors=st.integers(min_value=10, max_value=50),
        k=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=10000)
    def test_knn_results_are_sorted(self, dim, n_vectors, k):
        """KNN results should be sorted by distance (ascending)."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        distances = [r[1] for r in results]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1], f"Results not sorted: {distances[i]} > {distances[i + 1]}"

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=10, max_value=50))
    @settings(max_examples=20, deadline=10000)
    def test_knn_rowids_are_unique(self, dim, n_vectors):
        """KNN should not return duplicate rowids."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        query = np.random.randn(dim).astype(np.float32)
        k = min(10, n_vectors)
        results = db.execute(f"SELECT rowid FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)).fetchall()

        rowids = [r[0] for r in results]
        assert len(rowids) == len(set(rowids)), f"Duplicate rowids: {rowids}"

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=10, max_value=50))
    @settings(max_examples=20, deadline=10000)
    def test_knn_rowids_exist(self, dim, n_vectors):
        """All returned rowids should exist in the table."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        inserted_rowids = set()
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))
            inserted_rowids.add(i + 1)

        query = np.random.randn(dim).astype(np.float32)
        k = min(10, n_vectors)
        results = db.execute(f"SELECT rowid FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)).fetchall()

        for (rowid,) in results:
            assert rowid in inserted_rowids, f"Rowid {rowid} not in table"


# ============================================================================
# Vector function properties
# ============================================================================


class TestVectorFunctionProperties:
    """Property tests for vector helper functions."""

    @given(dim=st.integers(min_value=1, max_value=256))
    @settings(max_examples=30, deadline=5000)
    def test_vec_length_correct(self, dim):
        """vec_length should return the correct dimension."""
        db = connect()
        vec = np.random.randn(dim).astype(np.float32)

        result = db.execute("SELECT vec_length(?)", (to_blob(vec),)).fetchone()[0]
        assert result == dim

    @given(dim=st.integers(min_value=1, max_value=128))
    @settings(max_examples=20, deadline=5000)
    def test_vec_normalize_unit_length(self, dim):
        """Normalized vector should have approximately unit length (L2 norm)."""
        db = connect()

        # Avoid zero vectors
        vec = np.random.randn(dim).astype(np.float32)
        if np.linalg.norm(vec) < 1e-6:
            vec = np.ones(dim, dtype=np.float32)

        normalized = db.execute("SELECT vec_normalize(?)", (to_blob(vec),)).fetchone()[0]
        norm_arr = np.frombuffer(normalized, dtype=np.float32)

        # Check L2 norm is ~1
        norm = np.linalg.norm(norm_arr)
        assert abs(norm - 1.0) < 1e-5, f"Norm is {norm}, expected ~1"

    @given(
        dim=st.integers(min_value=4, max_value=64),
        start=st.integers(min_value=0, max_value=10),
        length=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=5000)
    def test_vec_slice_length(self, dim, start, length):
        """vec_slice should return correct number of elements."""
        assume(start + length <= dim)  # Valid slice bounds

        db = connect()
        vec = np.random.randn(dim).astype(np.float32)

        end = start + length
        sliced = db.execute("SELECT vec_slice(?, ?, ?)", (to_blob(vec), start, end)).fetchone()[0]

        sliced_arr = np.frombuffer(sliced, dtype=np.float32)
        assert len(sliced_arr) == length

    @given(
        dim=st.integers(min_value=2, max_value=64),
        start=st.integers(min_value=0, max_value=10),
        length=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=5000)
    def test_vec_slice_values(self, dim, start, length):
        """vec_slice should return correct element values."""
        assume(start + length <= dim)

        db = connect()
        vec = np.random.randn(dim).astype(np.float32)

        end = start + length
        sliced = db.execute("SELECT vec_slice(?, ?, ?)", (to_blob(vec), start, end)).fetchone()[0]

        sliced_arr = np.frombuffer(sliced, dtype=np.float32)
        expected = vec[start:end]

        np.testing.assert_array_almost_equal(sliced_arr, expected, decimal=5)


# ============================================================================
# IVF vs Brute Force consistency
# ============================================================================


class TestIVFConsistency:
    """Property tests verifying IVF matches brute force results."""

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_vectors=st.integers(min_value=20, max_value=100),
        k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=15, deadline=30000)
    def test_ivf_exhaustive_matches_bruteforce(self, dim, n_vectors, k):
        """IVF with nprobe=nlist should return same results as brute force."""
        assume(k <= n_vectors)

        nlist = 4  # Small nlist for testing

        # Generate deterministic vectors
        np.random.seed(42)
        vectors = [np.random.randn(dim).astype(np.float32) for _ in range(n_vectors)]
        query = np.random.randn(dim).astype(np.float32)

        # Brute force
        db_bf = connect()
        db_bf.execute(f"CREATE VIRTUAL TABLE bf USING vec0(embedding float[{dim}])")
        for i, vec in enumerate(vectors):
            db_bf.execute("INSERT INTO bf(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        bf_results = db_bf.execute(
            f"SELECT rowid, distance FROM bf WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()
        bf_rowids = set(r[0] for r in bf_results)

        # IVF with exhaustive search
        db_ivf = connect()
        db_ivf.execute(
            f"CREATE VIRTUAL TABLE ivf USING vec0(embedding float[{dim}], ivf_nlist={nlist}, ivf_nprobe={nlist})"
        )
        for i, vec in enumerate(vectors):
            db_ivf.execute("INSERT INTO ivf(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db_ivf.execute("SELECT vec0_ivf_train('ivf')")

        ivf_results = db_ivf.execute(
            f"SELECT rowid, distance FROM ivf WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()
        ivf_rowids = set(r[0] for r in ivf_results)

        # With nprobe=nlist, IVF should match brute force exactly
        note(f"BF rowids: {bf_rowids}, IVF rowids: {ivf_rowids}")
        assert bf_rowids == ivf_rowids, f"IVF differs from brute force: bf={bf_rowids}, ivf={ivf_rowids}"


# ============================================================================
# Insert/Delete invariants
# ============================================================================


class TestInsertDeleteProperties:
    """Property tests for insert/delete operations."""

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_inserts=st.integers(min_value=5, max_value=30),
        n_deletes=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=10000)
    def test_delete_removes_from_results(self, dim, n_inserts, n_deletes):
        """Deleted rowids should not appear in KNN results."""
        assume(n_deletes < n_inserts)

        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        for i in range(n_inserts):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Delete some rowids
        deleted = set(range(1, n_deletes + 1))
        for rowid in deleted:
            db.execute("DELETE FROM v WHERE rowid = ?", (rowid,))

        query = np.random.randn(dim).astype(np.float32)
        k = min(10, n_inserts - n_deletes)
        results = db.execute(f"SELECT rowid FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)).fetchall()

        result_rowids = set(r[0] for r in results)
        overlap = result_rowids & deleted
        assert len(overlap) == 0, f"Deleted rowids in results: {overlap}"

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=5, max_value=30))
    @settings(max_examples=20, deadline=10000)
    def test_point_query_returns_correct_rowid(self, dim, n_vectors):
        """Point query should return the exact rowid requested."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Query a specific rowid
        target_rowid = np.random.randint(1, n_vectors + 1)
        result = db.execute("SELECT rowid FROM v WHERE rowid = ?", (target_rowid,)).fetchone()

        assert result is not None
        assert result[0] == target_rowid


# ============================================================================
# IVF Edge Case Fuzzing
# ============================================================================


class TestIVFEdgeCaseFuzzing:
    """Fuzz testing for IVF edge cases - empty clusters, extreme ratios, etc."""

    @given(
        dim=st.integers(min_value=4, max_value=64),
        n_vectors=st.integers(min_value=1, max_value=20),
        nlist=st.integers(min_value=1, max_value=50),
        k=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=30000)
    def test_ivf_nlist_vs_vectors_ratio(self, dim, n_vectors, nlist, k):
        """Test IVF with various nlist/n_vectors ratios including nlist > n_vectors."""
        # nprobe should be at least nlist to search everything, capped at nlist
        nprobe = min(nlist, max(1, nlist))

        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist={nlist},
                ivf_nprobe={nprobe}
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Training should not crash regardless of nlist/n_vectors ratio
        try:
            db.execute("SELECT vec0_ivf_train('v')")
        except Exception:
            # Empty table training should fail gracefully
            if n_vectors == 0:
                return
            raise

        query = np.random.randn(dim).astype(np.float32)
        expected_k = min(k, n_vectors)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        # Should return at most min(k, n_vectors) results
        assert len(results) <= expected_k, f"Got {len(results)} results, expected <= {expected_k}"

        # Results should be sorted by distance
        distances = [r[1] for r in results]
        assert distances == sorted(distances), "Results not sorted"

        # All distances should be non-negative
        assert all(d >= 0 for d in distances), "Negative distances found"

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_vectors=st.integers(min_value=10, max_value=50),
        nprobe=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=30000)
    def test_ivf_varying_nprobe(self, dim, n_vectors, nprobe):
        """Test that varying nprobe doesn't crash and produces valid results."""
        nlist = 10

        db = connect()
        # nprobe must be <= nlist
        actual_nprobe = min(nprobe, nlist)
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist={nlist},
                ivf_nprobe={actual_nprobe}
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.random.randn(dim).astype(np.float32)
        k = 5
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        # With low nprobe, we may get fewer than k results if clusters are sparse
        # But we should get at least 1 and at most k
        assert 1 <= len(results) <= min(k, n_vectors), f"Got {len(results)} results, expected 1-{min(k, n_vectors)}"
        rowids = [r[0] for r in results]
        assert len(set(rowids)) == len(rowids), "Duplicate rowids"

        # Results should be sorted by distance
        distances = [r[1] for r in results]
        assert distances == sorted(distances), "Results not sorted"

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=20, max_value=100))
    @settings(max_examples=20, deadline=30000)
    def test_ivf_all_identical_vectors(self, dim, n_vectors):
        """Test IVF when all vectors are identical (edge case for clustering)."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=5,
                ivf_nprobe=5
            )
        """)

        # Insert identical vectors
        identical_vec = np.ones(dim, dtype=np.float32)
        for i in range(n_vectors):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(identical_vec)))

        # Training should handle this edge case
        db.execute("SELECT vec0_ivf_train('v')")

        # Query with the same vector
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=10", (to_blob(identical_vec),)
        ).fetchall()

        assert len(results) == 10
        # All distances should be 0 (or very close)
        for rowid, dist in results:
            assert dist < 1e-5, f"Distance {dist} too large for identical vectors"

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=10, max_value=50))
    @settings(max_examples=20, deadline=30000)
    def test_ivf_all_zero_vectors(self, dim, n_vectors):
        """Test IVF when all vectors are zero."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        zero_vec = np.zeros(dim, dtype=np.float32)
        for i in range(n_vectors):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(zero_vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (to_blob(zero_vec),)
        ).fetchall()

        assert len(results) == 5
        # All distances should be 0
        for rowid, dist in results:
            assert dist < 1e-6

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_initial=st.integers(min_value=20, max_value=50),
        n_insert_after=st.integers(min_value=1, max_value=20),
        n_delete=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=30000)
    def test_ivf_dynamic_insert_delete(self, dim, n_initial, n_insert_after, n_delete):
        """Test insert and delete operations after IVF training."""
        assume(n_delete < n_initial)

        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=5,
                ivf_nprobe=5
            )
        """)

        np.random.seed(42)
        for i in range(n_initial):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        # Insert more vectors after training
        for i in range(n_insert_after):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (n_initial + i + 1, to_blob(vec)))

        # Delete some vectors
        deleted = set(range(1, n_delete + 1))
        for rowid in deleted:
            db.execute("DELETE FROM v WHERE rowid = ?", (rowid,))

        # Query should still work
        query = np.random.randn(dim).astype(np.float32)
        total_remaining = n_initial + n_insert_after - n_delete
        k = min(10, total_remaining)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        # Deleted rowids should not appear
        result_rowids = set(r[0] for r in results)
        assert not (result_rowids & deleted), "Deleted rowids in results"

        # Results should be sorted
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=20, max_value=50))
    @settings(max_examples=15, deadline=60000)
    def test_ivf_retrain_consistency(self, dim, n_vectors):
        """Test that retraining produces consistent (valid) results."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=5,
                ivf_nprobe=5
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        query = np.random.randn(dim).astype(np.float32)

        # Train and query
        db.execute("SELECT vec0_ivf_train('v')")
        results1 = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)).fetchall()

        # Retrain
        db.execute("SELECT vec0_ivf_train('v')")
        results2 = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)).fetchall()

        # Both should return 5 results
        assert len(results1) == 5
        assert len(results2) == 5

        # Results might differ due to k-means randomness, but both should be valid rowids
        valid_rowids = set(range(1, n_vectors + 1))
        for (rowid,) in results1:
            assert rowid in valid_rowids
        for (rowid,) in results2:
            assert rowid in valid_rowids

    @given(
        dim=st.sampled_from([8, 16, 32, 64]),  # Must be multiple of 8 for bit vectors
        n_vectors=st.integers(min_value=20, max_value=100),
    )
    @settings(max_examples=20, deadline=30000)
    def test_ivf_int8_edge_cases(self, dim, n_vectors):
        """Test IVF with int8 vectors including extreme values."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding int8[{dim}],
                ivf_nlist=5,
                ivf_nprobe=5
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            # Include extreme values
            if i % 3 == 0:
                vec = np.full(dim, 127, dtype=np.int8)  # Max int8
            elif i % 3 == 1:
                vec = np.full(dim, -128, dtype=np.int8)  # Min int8
            else:
                vec = np.random.randint(-128, 128, dim, dtype=np.int8)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(dim, dtype=np.int8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_int8(?) AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    @given(
        n_bits=st.sampled_from([64, 128, 256]),  # Must be multiple of 8
        n_vectors=st.integers(min_value=20, max_value=100),
    )
    @settings(max_examples=20, deadline=30000)
    def test_ivf_bit_edge_cases(self, n_bits, n_vectors):
        """Test IVF with bit vectors including all-zeros and all-ones."""
        n_bytes = n_bits // 8

        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding bit[{n_bits}],
                ivf_nlist=5,
                ivf_nprobe=5
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            if i % 3 == 0:
                vec = np.zeros(n_bytes, dtype=np.uint8)  # All zeros
            elif i % 3 == 1:
                vec = np.full(n_bytes, 255, dtype=np.uint8)  # All ones
            else:
                vec = np.random.randint(0, 256, n_bytes, dtype=np.uint8)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(n_bytes, dtype=np.uint8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_bit(?) AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # Hamming distances should be non-negative integers
        for rowid, dist in results:
            assert dist >= 0
            assert float(dist).is_integer() or dist == int(dist)

    @given(
        dim=st.integers(min_value=4, max_value=32),
        n_vectors=st.integers(min_value=5, max_value=30),
        nlist=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=30000)
    def test_ivf_sparse_clusters(self, dim, n_vectors, nlist):
        """Test IVF when vectors are sparsely distributed (many empty clusters possible)."""
        # Use sparse vectors (mostly zeros with few non-zero elements)
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist={nlist},
                ivf_nprobe={nlist}
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.zeros(dim, dtype=np.float32)
            # Only set 1-2 dimensions to non-zero
            n_nonzero = min(2, dim)
            indices = np.random.choice(dim, n_nonzero, replace=False)
            vec[indices] = np.random.randn(n_nonzero)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(dim, dtype=np.float32)
        query[0] = 1.0
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)
        ).fetchall()

        expected_k = min(5, n_vectors)
        assert len(results) == expected_k


class TestIVFStressEdgeCases:
    """Stress tests for IVF edge cases that might cause crashes or corruption."""

    @given(st.data())
    @settings(max_examples=30, deadline=60000)
    def test_ivf_random_operations_sequence(self, data):
        """Random sequence of operations on IVF table."""
        dim = data.draw(st.integers(min_value=4, max_value=16))
        nlist = data.draw(st.integers(min_value=2, max_value=8))

        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist={nlist},
                ivf_nprobe={nlist}
            )
        """)

        np.random.seed(42)
        next_rowid = 1
        active_rowids = set()
        trained = False

        # Perform random sequence of operations
        n_ops = data.draw(st.integers(min_value=10, max_value=50))
        for _ in range(n_ops):
            op = data.draw(st.sampled_from(["insert", "delete", "query", "train"]))

            if op == "insert":
                vec = np.random.randn(dim).astype(np.float32)
                db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (next_rowid, to_blob(vec)))
                active_rowids.add(next_rowid)
                next_rowid += 1

            elif op == "delete" and active_rowids:
                rowid = data.draw(st.sampled_from(list(active_rowids)))
                db.execute("DELETE FROM v WHERE rowid = ?", (rowid,))
                active_rowids.discard(rowid)

            elif op == "train" and active_rowids:
                db.execute("SELECT vec0_ivf_train('v')")
                trained = True

            elif op == "query" and active_rowids:
                query = np.random.randn(dim).astype(np.float32)
                k = min(5, len(active_rowids))
                results = db.execute(
                    f"SELECT rowid FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
                ).fetchall()

                # Verify results are valid
                for (rowid,) in results:
                    assert rowid in active_rowids, f"Rowid {rowid} not in active set"

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=10, max_value=30))
    @settings(max_examples=20, deadline=30000)
    def test_ivf_extreme_values(self, dim, n_vectors):
        """Test IVF with extreme float values (very large, very small)."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            if i % 4 == 0:
                vec = np.full(dim, 1e30, dtype=np.float32)  # Very large
            elif i % 4 == 1:
                vec = np.full(dim, 1e-30, dtype=np.float32)  # Very small
            elif i % 4 == 2:
                vec = np.full(dim, -1e30, dtype=np.float32)  # Very large negative
            else:
                vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(dim, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)
        ).fetchall()

        assert len(results) == 5
        # Distances should be non-negative (might be inf for extreme vectors)
        for rowid, dist in results:
            assert dist >= 0 or np.isinf(dist)

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=5, max_value=20))
    @settings(max_examples=20, deadline=30000)
    def test_ivf_single_cluster_all_nprobe(self, dim, n_vectors):
        """Test IVF with nlist=1 (everything in one cluster)."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=1,
                ivf_nprobe=1
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        # Should behave like brute force since there's only one cluster
        query = np.random.randn(dim).astype(np.float32)
        k = min(5, n_vectors)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        assert len(results) == k

        # Verify against brute force
        db.execute(f"CREATE VIRTUAL TABLE bf USING vec0(embedding float[{dim}])")
        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO bf(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        bf_results = db.execute(f"SELECT rowid FROM bf WHERE embedding MATCH ? AND k={k}", (to_blob(query),)).fetchall()

        ivf_rowids = set(r[0] for r in results)
        bf_rowids = set(r[0] for r in bf_results)
        assert ivf_rowids == bf_rowids, "nlist=1 should match brute force exactly"


# ============================================================================
# Chunk Boundary and Blob Edge Cases
# ============================================================================


class TestChunkBoundaryFuzzing:
    """Fuzz testing for chunk boundaries and blob operations."""

    @given(
        dim=st.integers(min_value=4, max_value=32),
        chunk_size_mult=st.integers(min_value=1, max_value=8),  # 8, 16, 24, ..., 64
        n_vectors=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30, deadline=60000)
    def test_chunk_fill_patterns(self, dim, chunk_size_mult, n_vectors):
        """Test various chunk fill patterns - partial, exact, overflow."""
        chunk_size = chunk_size_mult * 8  # Must be divisible by 8
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                chunk_size={chunk_size}
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Query should work regardless of chunk patterns
        query = np.random.randn(dim).astype(np.float32)
        k = min(5, n_vectors)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        assert len(results) == k
        # All rowids should be valid
        for rowid, dist in results:
            assert 1 <= rowid <= n_vectors
            assert dist >= 0

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=10, max_value=50))
    @settings(max_examples=20, deadline=30000)
    def test_chunk_size_minimal(self, dim, n_vectors):
        """Test minimal chunk_size=8 (smallest valid chunk size)."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                chunk_size=8
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Query should span all chunks
        query = np.random.randn(dim).astype(np.float32)
        k = min(10, n_vectors)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        assert len(results) == k
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    @given(
        dim=st.integers(min_value=4, max_value=16),
        n_vectors=st.integers(min_value=50, max_value=100),
        delete_count=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=20, deadline=30000)
    def test_chunk_sparse_after_deletes(self, dim, n_vectors, delete_count):
        """Test queries on chunks with many deleted vectors (sparse validity bitmaps)."""
        assume(delete_count < n_vectors)

        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                chunk_size=16
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Delete vectors to create sparse chunks
        deleted = set()
        delete_indices = np.random.choice(n_vectors, delete_count, replace=False)
        for idx in delete_indices:
            rowid = int(idx) + 1  # Convert np.int64 to Python int
            db.execute("DELETE FROM v WHERE rowid = ?", (rowid,))
            deleted.add(rowid)

        # Query should only return non-deleted vectors
        query = np.random.randn(dim).astype(np.float32)
        remaining = n_vectors - delete_count
        k = min(10, remaining)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        result_rowids = set(r[0] for r in results)
        overlap = result_rowids & deleted
        assert not overlap, f"Deleted rowids in results: {overlap}"
        assert len(results) == k


# ============================================================================
# Distance Function Edge Cases
# ============================================================================


class TestDistanceEdgeCases:
    """Fuzz testing for distance function edge cases with special values."""

    @given(dim=st.integers(min_value=4, max_value=32))
    @settings(max_examples=20, deadline=10000)
    def test_distance_very_small_values(self, dim):
        """Test distance with very small (near-zero) values."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        # Insert vectors with very small values
        small_vec = np.full(dim, 1e-38, dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (to_blob(small_vec),))

        normal_vec = np.ones(dim, dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (2, ?)", (to_blob(normal_vec),))

        zero_vec = np.zeros(dim, dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (3, ?)", (to_blob(zero_vec),))

        # Query should handle small values
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=3", (to_blob(zero_vec),)
        ).fetchall()

        assert len(results) == 3
        for rowid, dist in results:
            assert dist >= 0
            assert not np.isnan(dist)

    @given(dim=st.integers(min_value=4, max_value=32))
    @settings(max_examples=20, deadline=10000)
    def test_distance_very_large_values(self, dim):
        """Test distance with very large values."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        # Insert vectors with large values (but not overflow-inducing)
        large_vec = np.full(dim, 1e10, dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (to_blob(large_vec),))

        neg_large_vec = np.full(dim, -1e10, dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (2, ?)", (to_blob(neg_large_vec),))

        normal_vec = np.ones(dim, dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (3, ?)", (to_blob(normal_vec),))

        # Query with zero
        query = np.zeros(dim, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=3", (to_blob(query),)
        ).fetchall()

        assert len(results) == 3
        # Normal vector should be closest to zero
        assert results[0][0] == 3

    @given(dim=st.integers(min_value=4, max_value=32), n_vectors=st.integers(min_value=10, max_value=50))
    @settings(max_examples=15, deadline=10000)
    def test_distance_mixed_magnitude_vectors(self, dim, n_vectors):
        """Test with vectors of wildly different magnitudes."""
        db = connect()
        db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dim}])")

        np.random.seed(42)
        for i in range(n_vectors):
            # Randomly scale each vector by different magnitude
            scale = 10 ** np.random.uniform(-5, 5)
            vec = (np.random.randn(dim) * scale).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)
        ).fetchall()

        assert len(results) == 5
        distances = [r[1] for r in results]
        assert distances == sorted(distances)


# ============================================================================
# IVF Assignment and Cache Consistency
# ============================================================================


class TestIVFCacheConsistency:
    """Fuzz testing for IVF assignment and inverted list cache consistency."""

    @given(
        dim=st.integers(min_value=4, max_value=16),
        n_initial=st.integers(min_value=30, max_value=60),
        n_insert=st.integers(min_value=5, max_value=20),
        n_delete=st.integers(min_value=3, max_value=15),
    )
    @settings(max_examples=20, deadline=60000)
    def test_ivf_assignment_after_modifications(self, dim, n_initial, n_insert, n_delete):
        """Test IVF assignments remain consistent after insert/delete."""
        assume(n_delete < n_initial)

        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=5,
                ivf_nprobe=5
            )
        """)

        np.random.seed(42)
        for i in range(n_initial):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        # Insert new vectors after training
        for i in range(n_insert):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (n_initial + i + 1, to_blob(vec)))

        # Delete some original vectors
        delete_indices = np.random.choice(n_initial, n_delete, replace=False)
        deleted = set()
        for idx in delete_indices:
            rowid = int(idx) + 1  # Convert np.int64 to Python int
            db.execute("DELETE FROM v WHERE rowid = ?", (rowid,))
            deleted.add(rowid)

        # Query should work and return only valid rowids
        query = np.random.randn(dim).astype(np.float32)
        total = n_initial + n_insert - n_delete
        k = min(10, total)
        results = db.execute(
            f"SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k={k}", (to_blob(query),)
        ).fetchall()

        result_rowids = set(r[0] for r in results)
        overlap = result_rowids & deleted
        assert not overlap, f"Deleted rowids in results: {overlap}"

        # All result rowids should be valid
        valid_rowids = set(range(1, n_initial + n_insert + 1)) - deleted
        for rowid in result_rowids:
            assert rowid in valid_rowids

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=20, max_value=50))
    @settings(max_examples=15, deadline=60000)
    def test_ivf_update_vector_assignment(self, dim, n_vectors):
        """Test that updating a vector updates its IVF assignment."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        db.execute("SELECT vec0_ivf_train('v')")

        # Update a vector to a very different value
        update_rowid = n_vectors // 2
        new_vec = np.full(dim, 100.0, dtype=np.float32)  # Very different from random
        db.execute("UPDATE v SET embedding = ? WHERE rowid = ?", (to_blob(new_vec), update_rowid))

        # Query with the new vector should find the updated row
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=1", (to_blob(new_vec),)
        ).fetchall()

        assert len(results) == 1
        assert results[0][0] == update_rowid
        assert results[0][1] < 1e-5  # Distance should be ~0


# ============================================================================
# Metadata and Auxiliary Column Edge Cases
# ============================================================================


class TestMetadataAuxiliaryFuzzing:
    """Fuzz testing for metadata and auxiliary column edge cases."""

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=10, max_value=30))
    @settings(max_examples=20, deadline=30000)
    def test_metadata_filter_combinations(self, dim, n_vectors):
        """Test various metadata filter combinations."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                category integer,
                score float
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            category = i % 5
            score = np.random.uniform(0, 100)
            db.execute(
                "INSERT INTO v(rowid, embedding, category, score) VALUES (?, ?, ?, ?)",
                (i + 1, to_blob(vec), category, score),
            )

        query = np.random.randn(dim).astype(np.float32)

        # Test with category filter
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5 AND category = 2", (to_blob(query),)
        ).fetchall()

        # All results should have category=2
        for rowid, _ in results:
            cat = db.execute("SELECT category FROM v WHERE rowid = ?", (rowid,)).fetchone()[0]
            assert cat == 2

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=10, max_value=30))
    @settings(max_examples=20, deadline=30000)
    def test_auxiliary_column_retrieval(self, dim, n_vectors):
        """Test auxiliary column data is retrieved correctly with KNN."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                +label text,
                +value integer
            )
        """)

        np.random.seed(42)
        labels = []
        values = []
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            label = f"item_{i}"
            value = i * 10
            labels.append(label)
            values.append(value)
            db.execute(
                "INSERT INTO v(rowid, embedding, label, value) VALUES (?, ?, ?, ?)", (i + 1, to_blob(vec), label, value)
            )

        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(
            "SELECT rowid, distance, label, value FROM v WHERE embedding MATCH ? AND k=5", (to_blob(query),)
        ).fetchall()

        assert len(results) == 5
        for rowid, dist, label, value in results:
            expected_label = f"item_{rowid - 1}"
            expected_value = (rowid - 1) * 10
            assert label == expected_label
            assert value == expected_value

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=10, max_value=30))
    @settings(max_examples=15, deadline=30000)
    def test_auxiliary_null_values(self, dim, n_vectors):
        """Test auxiliary columns with NULL values."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dim}],
                +label text
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            # Every other row has NULL label
            label = f"item_{i}" if i % 2 == 0 else None
            db.execute("INSERT INTO v(rowid, embedding, label) VALUES (?, ?, ?)", (i + 1, to_blob(vec), label))

        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(
            "SELECT rowid, label FROM v WHERE embedding MATCH ? AND k=10", (to_blob(query),)
        ).fetchall()

        for rowid, label in results:
            if (rowid - 1) % 2 == 0:
                assert label == f"item_{rowid - 1}"
            else:
                assert label is None


# ============================================================================
# Virtual Table Lifecycle Edge Cases
# ============================================================================


class TestVTableLifecycleFuzzing:
    """Fuzz testing for virtual table lifecycle edge cases."""

    @given(dim=st.integers(min_value=4, max_value=16), n_tables=st.integers(min_value=2, max_value=5))
    @settings(max_examples=15, deadline=30000)
    def test_create_drop_recreate_cycle(self, dim, n_tables):
        """Test creating, dropping, and recreating tables with same name."""
        db = connect()

        for cycle in range(n_tables):
            # Create table
            db.execute(f"CREATE VIRTUAL TABLE test_table USING vec0(embedding float[{dim}])")

            # Insert some data
            np.random.seed(42 + cycle)
            n_vectors = 10 + cycle * 5
            for i in range(n_vectors):
                vec = np.random.randn(dim).astype(np.float32)
                db.execute("INSERT INTO test_table(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

            # Query
            query = np.random.randn(dim).astype(np.float32)
            results = db.execute(
                "SELECT rowid FROM test_table WHERE embedding MATCH ? AND k=5", (to_blob(query),)
            ).fetchall()
            assert len(results) == 5

            # Drop table
            db.execute("DROP TABLE test_table")

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=10, max_value=30))
    @settings(max_examples=15, deadline=30000)
    def test_multiple_tables_same_db(self, dim, n_vectors):
        """Test multiple vec0 tables in the same database."""
        db = connect()

        table_names = ["vectors_a", "vectors_b", "vectors_c"]
        for name in table_names:
            db.execute(f"CREATE VIRTUAL TABLE {name} USING vec0(embedding float[{dim}])")

        np.random.seed(42)
        # Insert different data into each table
        for table_idx, name in enumerate(table_names):
            for i in range(n_vectors):
                vec = np.random.randn(dim).astype(np.float32) * (table_idx + 1)
                db.execute(f"INSERT INTO {name}(rowid, embedding) VALUES (?, ?)", (i + 1, to_blob(vec)))

        # Query each table
        query = np.random.randn(dim).astype(np.float32)
        for name in table_names:
            results = db.execute(
                f"SELECT rowid FROM {name} WHERE embedding MATCH ? AND k=5", (to_blob(query),)
            ).fetchall()
            assert len(results) == 5

    @given(dim=st.integers(min_value=4, max_value=16), n_vectors=st.integers(min_value=20, max_value=50))
    @settings(max_examples=10, deadline=60000)
    def test_table_with_all_options(self, dim, n_vectors):
        """Test table with all options combined."""
        db = connect()
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                user_id text primary key,
                embedding float[{dim}],
                category integer,
                +label text,
                chunk_size=16,
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(n_vectors):
            vec = np.random.randn(dim).astype(np.float32)
            user_id = f"user_{i:04d}"
            category = i % 3
            label = f"label_{i}"
            db.execute(
                "INSERT INTO v(user_id, embedding, category, label) VALUES (?, ?, ?, ?)",
                (user_id, to_blob(vec), category, label),
            )

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with all features
        query = np.random.randn(dim).astype(np.float32)
        results = db.execute(
            "SELECT user_id, label FROM v WHERE embedding MATCH ? AND k=5 AND category = 1", (to_blob(query),)
        ).fetchall()

        # Verify results are from category 1
        for user_id, label in results:
            # Extract index from user_id
            idx = int(user_id.split("_")[1])
            assert idx % 3 == 1


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
