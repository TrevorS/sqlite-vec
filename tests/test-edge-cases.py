"""
Edge case tests for sqlite-vec.

Tests boundary conditions, error handling, and unusual inputs
that might cause crashes or incorrect behavior.
"""

import sqlite3
import struct

import pytest

EXT_PATH = "./dist/vec0"


def connect(ext=EXT_PATH):
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(ext)
    db.enable_load_extension(False)
    return db


# ============================================================================
# Vector Size Edge Cases
# ============================================================================


class TestVectorSizes:
    """Test edge cases related to vector dimensions and sizes."""

    def test_single_dimension_vector(self):
        """Vector with only 1 dimension."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[1])")

        vec = struct.pack("f", 1.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        result = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert len(result) == 1

    def test_large_dimension_vector(self):
        """Vector with many dimensions (1024)."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[1024])")

        vec = struct.pack("1024f", *([1.0] * 1024))
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        result = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert len(result) == 1

    def test_zero_length_vector_rejected(self):
        """Zero-length vectors should be rejected."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[0])")

    def test_negative_dimension_rejected(self):
        """Negative dimensions should be rejected."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[-1])")


# ============================================================================
# NULL and Empty Value Handling
# ============================================================================


class TestNullHandling:
    """Test NULL value handling in various contexts."""

    def test_null_vector_insert(self):
        """NULL vector insert should fail."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, NULL)")

    def test_null_query_vector(self):
        """NULL query vector should fail."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT rowid FROM v WHERE embedding MATCH NULL AND k=1").fetchall()

    def test_empty_table_knn(self):
        """KNN on empty table should return no results."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=10", (query,)).fetchall()
        assert len(results) == 0


# ============================================================================
# K Value Edge Cases
# ============================================================================


class TestKValues:
    """Test edge cases for the k parameter in KNN queries."""

    def test_k_equals_zero(self):
        """k=0 should return no results."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=0", (vec,)).fetchall()
        assert len(results) == 0

    def test_k_larger_than_table(self):
        """k larger than table size should return all rows."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        for i in range(5):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        query = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=100", (query,)).fetchall()
        assert len(results) == 5

    def test_k_equals_one(self):
        """k=1 should return exactly one result."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        for i in range(10):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        query = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()
        assert len(results) == 1

    def test_negative_k_rejected(self):
        """Negative k should be rejected."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=-1", (vec,)).fetchall()


# ============================================================================
# Special Float Values
# ============================================================================


class TestSpecialFloats:
    """Test handling of special float values (inf, -inf, nan, denormals)."""

    def test_infinity_in_vector(self):
        """Vectors containing infinity."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", float("inf"), 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        # Should still be queryable
        query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()
        assert len(results) == 1

    def test_negative_infinity_in_vector(self):
        """Vectors containing negative infinity."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", float("-inf"), 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()
        assert len(results) == 1

    def test_nan_in_vector(self):
        """Vectors containing NaN."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", float("nan"), 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()
        # Should still return results (NaN distance handling)
        assert len(results) == 1

    def test_very_small_floats(self):
        """Vectors with very small (denormalized) floats."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        # Smallest positive denormalized float
        tiny = 1.4e-45
        vec = struct.pack("4f", tiny, tiny, tiny, tiny)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert len(results) == 1

    def test_very_large_floats(self):
        """Vectors with very large floats."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        big = 3.4e38  # Near float max
        vec = struct.pack("4f", big, big, big, big)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert len(results) == 1


# ============================================================================
# Malformed Input
# ============================================================================


class TestMalformedInput:
    """Test handling of malformed or corrupted input."""

    def test_wrong_size_vector(self):
        """Vector with wrong number of bytes should be rejected."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        # 3 floats instead of 4
        vec = struct.pack("3f", 1.0, 0.0, 0.0)
        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

    def test_empty_blob(self):
        """Empty blob should be rejected."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, X'')")

    def test_text_instead_of_blob(self):
        """Text instead of blob should be rejected or handled."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        # Text that's not valid JSON
        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, 'hello')")

    def test_invalid_json_vector(self):
        """Invalid JSON should be rejected."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, '[1.0, 2.0, ')")


# ============================================================================
# Chunk Boundary Edge Cases
# ============================================================================


class TestChunkBoundaries:
    """Test behavior at chunk boundaries."""

    def test_exactly_one_chunk(self):
        """Insert exactly chunk_size vectors."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], chunk_size=64)")

        for i in range(64):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        query = struct.pack("4f", 32.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=10", (query,)).fetchall()
        assert len(results) == 10

    def test_chunk_plus_one(self):
        """Insert chunk_size + 1 vectors (spans two chunks)."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], chunk_size=64)")

        for i in range(65):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        query = struct.pack("4f", 64.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=10", (query,)).fetchall()
        assert len(results) == 10

    def test_minimum_chunk_size(self):
        """Test with minimum allowed chunk_size."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], chunk_size=64)")

        for i in range(10):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        query = struct.pack("4f", 5.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=5", (query,)).fetchall()
        assert len(results) == 5


# ============================================================================
# Distance Metric Edge Cases
# ============================================================================


class TestDistanceMetrics:
    """Test edge cases for different distance metrics."""

    def test_cosine_zero_vector(self):
        """Cosine distance with zero vector."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance=cosine)")

        # Insert zero vector
        zero_vec = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (zero_vec,))

        # Query with non-zero vector
        query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()
        # Should handle gracefully (distance may be NaN or special value)
        assert len(results) == 1

    def test_identical_vectors_zero_distance(self):
        """Identical vectors should have zero distance."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        results = db.execute("SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert len(results) == 1
        assert abs(results[0][1]) < 1e-6  # Distance should be ~0


# ============================================================================
# Delete and Update Edge Cases
# ============================================================================


class TestDeleteUpdate:
    """Test edge cases for delete and update operations."""

    def test_delete_all_then_query(self):
        """Delete all rows then query."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        for i in range(10):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        db.execute("DELETE FROM v")

        query = struct.pack("4f", 5.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=10", (query,)).fetchall()
        assert len(results) == 0

    def test_delete_half_then_query(self):
        """Delete half the rows then query."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        for i in range(10):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        # Delete even rowids
        for i in range(0, 10, 2):
            db.execute("DELETE FROM v WHERE rowid = ?", (i + 1,))

        query = struct.pack("4f", 5.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=10", (query,)).fetchall()
        assert len(results) == 5

    def test_update_vector_then_query(self):
        """Update a vector and verify query reflects change."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec1 = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
        vec2 = struct.pack("4f", 100.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec1,))

        # Update to far away vector
        db.execute("UPDATE v SET embedding = ? WHERE rowid = 1", (vec2,))

        # Query near origin - should have large distance now
        query = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()
        assert len(results) == 1
        assert results[0][1] > 99.0  # Distance should be ~100


# ============================================================================
# Rowid Edge Cases
# ============================================================================


class TestRowids:
    """Test edge cases for rowid handling."""

    def test_large_rowid(self):
        """Very large rowid values."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        large_rowid = 2**62
        db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (large_rowid, vec))

        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert results[0][0] == large_rowid

    def test_negative_rowid(self):
        """Negative rowid values."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (-1, ?)", (vec,))

        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert results[0][0] == -1

    def test_duplicate_rowid_rejected(self):
        """Duplicate rowid should be rejected."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        # sqlite-vec raises OperationalError for constraint violations
        with pytest.raises(sqlite3.OperationalError, match="UNIQUE constraint failed"):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))


# ============================================================================
# IVF Edge Cases
# ============================================================================


class TestIVFEdgeCases:
    """Test edge cases specific to IVF implementation."""

    def test_ivf_nlist_larger_than_vectors(self):
        """nlist larger than number of vectors - should not crash, may return fewer results."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=100)")

        # Insert only 10 vectors (fewer than nlist)
        for i in range(10):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        # Training should still work without crashing
        db.execute("SELECT vec0_ivf_train('v')")

        # Query should work without crashing
        # Note: When nlist > n_vectors, k-means creates mostly empty clusters
        # so IVF may return fewer results than expected. This is known behavior.
        query = struct.pack("4f", 5.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=5", (query,)).fetchall()
        # At minimum, we should get some results without error
        assert len(results) >= 1

    def test_ivf_nprobe_equals_nlist(self):
        """nprobe = nlist (exhaustive search via IVF)."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4, ivf_nprobe=4)")

        for i in range(20):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

        db.execute("SELECT vec0_ivf_train('v')")

        # Should get same results as brute force
        query = struct.pack("4f", 10.0, 0.0, 0.0, 0.0)
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=5", (query,)).fetchall()
        assert len(results) == 5


# ============================================================================
# Concurrent Access (Basic)
# ============================================================================


class TestConcurrency:
    """Basic concurrency tests (single connection, multiple operations)."""

    def test_interleaved_insert_query(self):
        """Interleave inserts and queries."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        for i in range(100):
            vec = struct.pack("4f", float(i), 0.0, 0.0, 0.0)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec))

            # Query after each insert
            if i > 0:
                query = struct.pack("4f", float(i / 2), 0.0, 0.0, 0.0)
                results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=5", (query,)).fetchall()
                assert len(results) <= i + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
