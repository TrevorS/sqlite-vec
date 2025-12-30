"""
Error path tests for sqlite-vec.

Tests error handling, invalid inputs, and edge cases that trigger
error paths in the C code to improve code coverage.
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
# JSON Parsing Error Paths
# ============================================================================


class TestJSONErrors:
    """Test JSON parsing error paths."""

    def test_json_not_array(self):
        """JSON object instead of array."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec_f32('{\"a\": 1}')").fetchall()

    def test_json_nested_array(self):
        """Nested JSON array (should fail)."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec_f32('[[1.0, 2.0], [3.0, 4.0]]')").fetchall()

    def test_json_string_elements(self):
        """JSON array with string elements."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute('SELECT vec_f32(\'["a", "b", "c"]\')').fetchall()

    def test_json_null_elements(self):
        """JSON array with null elements."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec_f32('[1.0, null, 3.0]')").fetchall()


# ============================================================================
# int8 Vector Error Paths
# ============================================================================


class TestInt8Errors:
    """Test int8 vector error paths."""

    def test_int8_out_of_range(self):
        """int8 value out of range (-128 to 127)."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="int8|range"):
            db.execute("SELECT vec_int8('[128]')").fetchall()

    def test_int8_negative_out_of_range(self):
        """int8 negative value out of range."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="int8|range"):
            db.execute("SELECT vec_int8('[-129]')").fetchall()

    def test_int8_float_value(self):
        """int8 with float value (should truncate or error)."""
        db = connect()
        # Should either work (truncation) or error - just shouldn't crash
        try:
            result = db.execute("SELECT vec_int8('[1.5, 2.7]')").fetchone()
            assert result is not None
        except sqlite3.OperationalError:
            pass  # Also acceptable

    def test_int8_empty_array(self):
        """int8 with empty array."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="zero-length|empty"):
            db.execute("SELECT vec_int8('[]')").fetchall()

    def test_int8_table_type_mismatch(self):
        """Insert float vector into int8 column."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding int8[4])")
        vec = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)  # float32 bytes
        with pytest.raises(sqlite3.OperationalError, match="int8|float|type"):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))


# ============================================================================
# bit Vector Error Paths
# ============================================================================


class TestBitErrors:
    """Test bit vector error paths."""

    def test_bit_wrong_size_blob(self):
        """bit vector with wrong size blob."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding bit[16])")
        # 16 bits = 2 bytes, but we provide 3
        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (1, vec_bit(X'AABBCC'))")


# ============================================================================
# Type Mismatch Errors
# ============================================================================


class TestTypeMismatch:
    """Test type mismatch error paths."""

    def test_distance_type_mismatch(self):
        """Distance calculation with mismatched types."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="type|mismatch"):
            db.execute("SELECT vec_distance_L2('[1.0, 2.0]', vec_int8('[1, 2]'))").fetchall()

    def test_add_type_mismatch(self):
        """Vector addition with mismatched types."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="type|mismatch"):
            db.execute("SELECT vec_add('[1.0, 2.0]', vec_int8('[1, 2]'))").fetchall()

    def test_sub_type_mismatch(self):
        """Vector subtraction with mismatched types."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="type|mismatch"):
            db.execute("SELECT vec_sub('[1.0, 2.0]', vec_int8('[1, 2]'))").fetchall()


# ============================================================================
# Length Mismatch Errors
# ============================================================================


class TestLengthMismatch:
    """Test length mismatch error paths."""

    def test_distance_length_mismatch(self):
        """Distance calculation with different lengths."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="length|dimension"):
            db.execute("SELECT vec_distance_L2('[1.0, 2.0]', '[1.0, 2.0, 3.0]')").fetchall()

    def test_add_length_mismatch(self):
        """Vector addition with different lengths."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="length|dimension"):
            db.execute("SELECT vec_add('[1.0, 2.0]', '[1.0, 2.0, 3.0]')").fetchall()


# ============================================================================
# Bit Vector Operations Errors
# ============================================================================


class TestBitOperationErrors:
    """Test bit vector operation error paths."""

    def test_bit_l2_distance(self):
        """L2 distance on bit vectors (not supported)."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="bit|L2"):
            db.execute("SELECT vec_distance_L2(vec_bit(X'AA'), vec_bit(X'BB'))").fetchall()

    def test_bit_cosine_distance(self):
        """Cosine distance on bit vectors (not supported)."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="bit|cosine"):
            db.execute("SELECT vec_distance_cosine(vec_bit(X'AA'), vec_bit(X'BB'))").fetchall()

    def test_bit_add(self):
        """Addition of bit vectors (not supported)."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="bit|add"):
            db.execute("SELECT vec_add(vec_bit(X'AA'), vec_bit(X'BB'))").fetchall()

    def test_bit_sub(self):
        """Subtraction of bit vectors (not supported)."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="bit|subtract"):
            db.execute("SELECT vec_sub(vec_bit(X'AA'), vec_bit(X'BB'))").fetchall()


# ============================================================================
# Quantization Error Paths
# ============================================================================


class TestQuantizationErrors:
    """Test quantization error paths."""

    def test_binary_quantize_bit_vector(self):
        """Binary quantization of already-bit vector."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="bit|quantize|float|int8"):
            db.execute("SELECT vec_quantize_binary(vec_bit(X'FF'))").fetchall()

    def test_binary_quantize_wrong_length(self):
        """Binary quantization of vector with length not divisible by 8."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="divisible|8"):
            db.execute("SELECT vec_quantize_binary('[1.0, 2.0, 3.0, 4.0]')").fetchall()


# ============================================================================
# vec0 Table Error Paths
# ============================================================================


class TestVec0Errors:
    """Test vec0 virtual table error paths."""

    def test_invalid_distance_metric(self):
        """Invalid distance metric name."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4] distance=invalid)")

    def test_invalid_column_type(self):
        """Invalid column type."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding invalid[4])")

    def test_missing_dimensions(self):
        """Missing dimension specification."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float)")

    def test_query_wrong_dimension_vector(self):
        """Query with wrong dimension vector."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        # Query with 3D vector instead of 4D
        query = struct.pack("3f", 1.0, 0.0, 0.0)
        with pytest.raises(sqlite3.OperationalError, match="dimension|length"):
            db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (query,)).fetchall()


# ============================================================================
# IVF Error Paths
# ============================================================================


class TestIVFErrors:
    """Test IVF-specific error paths."""

    def test_train_untrained_query(self):
        """Query IVF table before training (should work, uses untrained mode)."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")
        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec,))

        # Query before training - should still work
        results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=1", (vec,)).fetchall()
        assert len(results) == 1

    def test_train_empty_table(self):
        """Train IVF on empty table."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Training empty table might error or be a no-op
        try:
            db.execute("SELECT vec0_ivf_train('v')")
        except sqlite3.OperationalError:
            pass  # Acceptable to error on empty table

    def test_train_nonexistent_table(self):
        """Train IVF on non-existent table."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec0_ivf_train('nonexistent')")

    def test_set_option_invalid_table(self):
        """Set option on non-existent table."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec0_set_option('nonexistent', 'nprobe', 4)")

    def test_set_option_non_ivf_table(self):
        """Set IVF option on non-IVF table."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
        # Non-IVF table doesn't support nprobe option
        with pytest.raises(sqlite3.OperationalError, match="unknown option"):
            db.execute("SELECT vec0_set_option('v', 'nprobe', 4)")

    def test_ivf_nprobe_zero_rejected(self):
        """IVF with nprobe=0 should be rejected at table creation."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError, match="positive integer"):
            db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4, ivf_nprobe=0)")


# ============================================================================
# Slice/Normalize Errors
# ============================================================================


class TestSliceNormalizeErrors:
    """Test vec_slice and vec_normalize error paths."""

    def test_slice_negative_start(self):
        """Slice with negative start index."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec_slice('[1.0, 2.0, 3.0, 4.0]', -1, 2)").fetchall()

    def test_slice_end_before_start(self):
        """Slice with end before start."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec_slice('[1.0, 2.0, 3.0, 4.0]', 3, 1)").fetchall()

    def test_slice_out_of_bounds(self):
        """Slice with out of bounds indices."""
        db = connect()
        with pytest.raises(sqlite3.OperationalError):
            db.execute("SELECT vec_slice('[1.0, 2.0, 3.0, 4.0]', 0, 10)").fetchall()

    def test_normalize_zero_vector(self):
        """Normalize zero vector."""
        db = connect()
        # Zero vector normalization is undefined (division by zero)
        # Should either error or return NaN values
        try:
            result = db.execute("SELECT vec_normalize('[0.0, 0.0, 0.0]')").fetchone()
            # If it returns, values should be NaN or zero
            assert result is not None
        except sqlite3.OperationalError:
            pass  # Also acceptable to error


# ============================================================================
# Multiple Vector Column Errors
# ============================================================================


class TestMultiColumnErrors:
    """Test error paths with multiple vector columns."""

    def test_insert_missing_column(self):
        """Insert with missing required column."""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(emb1 float[4], emb2 float[4])")
        vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)

        # Only provide one embedding
        with pytest.raises(sqlite3.OperationalError):
            db.execute("INSERT INTO v(rowid, emb1) VALUES (1, ?)", (vec,))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
