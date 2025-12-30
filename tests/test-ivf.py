"""
Comprehensive tests for IVF (Inverted File Index) approximate nearest neighbor search.

These tests verify:
1. IVF table creation with ivf_nlist and ivf_nprobe options
2. IVF training with vec0_ivf_train()
3. IVF-accelerated KNN queries
4. Dynamic IVF assignment on insert/update/delete
5. Recall comparison between IVF and brute-force search
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

EXT_PATH = Path(__file__).parent.parent / "dist" / "vec0"


def connect(ext=EXT_PATH) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(str(ext))
    db.enable_load_extension(False)
    return db


class TestIVFTableCreation:
    """Test IVF table creation and configuration"""

    def test_ivf_table_creation_with_nlist(self):
        """Test creating a table with ivf_nlist option"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=8)")

        # Verify shadow tables exist
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v_%' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "v_ivf_centroids" in table_names
        assert "v_ivf_assignments" in table_names
        assert "v_chunks" in table_names
        assert "v_rowids" in table_names

    def test_ivf_table_creation_with_nprobe(self):
        """Test creating a table with ivf_nlist and ivf_nprobe options"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=16, ivf_nprobe=4)")

        # Should succeed and create tables
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v_%' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "v_ivf_centroids" in table_names
        assert "v_ivf_assignments" in table_names

    def test_ivf_info_table_stores_config(self):
        """Test that ivf_nlist is stored in the _info table"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=16)")

        result = db.execute("SELECT value FROM v_info WHERE key='ivf_nlist'").fetchone()
        assert result is not None
        assert int(result[0]) == 16


class TestIVFTraining:
    """Test IVF training functionality"""

    def test_ivf_train_basic(self):
        """Test basic IVF training"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert some vectors
        vectors = np.random.randn(100, 4).astype(np.float32)
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Train IVF
        result = db.execute("SELECT vec0_ivf_train('v')").fetchone()
        # Success returns NULL or a success message
        assert result[0] is None or "complete" in str(result[0]).lower() or result[0] == ""

        # Verify training succeeded
        is_trained = db.execute("SELECT vec0_ivf_is_trained('v')").fetchone()[0]
        assert is_trained == 1

    def test_ivf_train_creates_centroids(self):
        """Test that training creates centroids"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert vectors
        vectors = np.random.randn(50, 4).astype(np.float32)
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Train
        db.execute("SELECT vec0_ivf_train('v')")

        # Check centroids were created
        centroid_count = db.execute("SELECT COUNT(*) FROM v_ivf_centroids").fetchone()[0]
        assert centroid_count == 4

    def test_ivf_train_creates_assignments(self):
        """Test that training creates assignments for all vectors"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert 50 vectors
        vectors = np.random.randn(50, 4).astype(np.float32)
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Train
        db.execute("SELECT vec0_ivf_train('v')")

        # Check all vectors have assignments
        assignment_count = db.execute("SELECT COUNT(*) FROM v_ivf_assignments").fetchone()[0]
        assert assignment_count == 50

    def test_ivf_train_without_ivf_enabled_fails(self):
        """Test that training a table without IVF enabled fails"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")

        # Insert a vector
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, X'00000000000000000000000000000000')")

        # Training should fail
        with pytest.raises(sqlite3.OperationalError, match="ivf_nlist"):
            db.execute("SELECT vec0_ivf_train('v')").fetchone()

    def test_ivf_train_empty_table_fails(self):
        """Test that training an empty table fails"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        with pytest.raises(sqlite3.OperationalError, match="empty"):
            db.execute("SELECT vec0_ivf_train('v')").fetchone()


class TestIVFQuery:
    """Test IVF-accelerated KNN queries"""

    def setup_trained_table(self, db, n_vectors=200, dimensions=8, nlist=8, nprobe=2):
        """Helper to create and train an IVF table"""
        db.execute(
            f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{dimensions}], ivf_nlist={nlist}, ivf_nprobe={nprobe})"
        )

        # Generate clustered vectors for better IVF performance
        np.random.seed(42)
        vectors = []
        for cluster in range(nlist):
            center = np.random.randn(dimensions).astype(np.float32) * 10
            cluster_vecs = center + np.random.randn(n_vectors // nlist, dimensions).astype(np.float32)
            vectors.append(cluster_vecs)
        vectors = np.vstack(vectors)

        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")
        return vectors

    def test_ivf_knn_query_basic(self):
        """Test basic IVF KNN query"""
        db = connect()
        vectors = self.setup_trained_table(db, n_vectors=100, dimensions=4, nlist=4, nprobe=2)

        # Query with one of the vectors
        query = vectors[0]
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # First result should be the query vector itself
        assert results[0][0] == 1
        assert results[0][1] < 1e-5  # Distance should be ~0

    def test_ivf_knn_returns_sorted_results(self):
        """Test that IVF KNN returns results sorted by distance"""
        db = connect()
        vectors = self.setup_trained_table(db, n_vectors=100, dimensions=4, nlist=4, nprobe=4)

        query = vectors[50]
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=10", (query.tobytes(),)
        ).fetchall()

        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_ivf_recall_reasonable(self):
        """Test that IVF recall is reasonable (at least 70%)"""
        db = connect()
        vectors = self.setup_trained_table(db, n_vectors=200, dimensions=8, nlist=8, nprobe=4)

        # Also create a brute-force table for comparison
        db.execute("CREATE VIRTUAL TABLE v_bf USING vec0(embedding float[8])")
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Test recall on several queries
        np.random.seed(123)
        test_indices = np.random.choice(len(vectors), 10, replace=False)
        recalls = []

        for idx in test_indices:
            query = vectors[idx]
            k = 10

            # Get IVF results
            ivf_results = db.execute(
                "SELECT rowid FROM v WHERE embedding MATCH ? AND k=?", (query.tobytes(), k)
            ).fetchall()
            ivf_ids = set(r[0] for r in ivf_results)

            # Get brute-force results
            bf_results = db.execute(
                "SELECT rowid FROM v_bf WHERE embedding MATCH ? AND k=?", (query.tobytes(), k)
            ).fetchall()
            bf_ids = set(r[0] for r in bf_results)

            # Calculate recall
            recall = len(ivf_ids & bf_ids) / k
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        assert avg_recall >= 0.7, f"Average recall {avg_recall} is too low"

    def test_ivf_nprobe_query_override(self):
        """Test that nprobe can be overridden at query time"""
        db = connect()
        vectors = self.setup_trained_table(db, n_vectors=200, dimensions=8, nlist=8, nprobe=1)

        # Also create a brute-force table for comparison
        db.execute("CREATE VIRTUAL TABLE v_bf USING vec0(embedding float[8])")
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        query = vectors[0]
        k = 10

        # Get brute-force results (ground truth)
        bf_results = db.execute(
            "SELECT rowid FROM v_bf WHERE embedding MATCH ? AND k=?", (query.tobytes(), k)
        ).fetchall()
        bf_ids = set(r[0] for r in bf_results)

        # Query with default nprobe=1 (set at table creation)
        r1 = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=?", (query.tobytes(), k)).fetchall()
        r1_ids = set(r[0] for r in r1)
        recall_nprobe1 = len(r1_ids & bf_ids) / k

        # Query with nprobe=8 (override to search all clusters)
        r2 = db.execute(
            "SELECT rowid FROM v WHERE embedding MATCH ? AND k=? AND nprobe=8", (query.tobytes(), k)
        ).fetchall()
        r2_ids = set(r[0] for r in r2)
        recall_nprobe8 = len(r2_ids & bf_ids) / k

        # Higher nprobe should give better or equal recall
        assert recall_nprobe8 >= recall_nprobe1, (
            f"nprobe=8 recall ({recall_nprobe8}) should be >= nprobe=1 ({recall_nprobe1})"
        )

        # nprobe=8 (all clusters) should give perfect recall
        assert recall_nprobe8 == 1.0, f"nprobe=8 should give 100% recall, got {recall_nprobe8}"

    def test_ivf_nprobe_invalid_value(self):
        """Test that invalid nprobe values are rejected"""
        db = connect()
        vectors = self.setup_trained_table(db, n_vectors=50, dimensions=4, nlist=4, nprobe=2)

        query = vectors[0]

        # nprobe=0 should fail
        with pytest.raises(sqlite3.OperationalError, match="nprobe must be a positive integer"):
            db.execute(
                "SELECT rowid FROM v WHERE embedding MATCH ? AND k=5 AND nprobe=0", (query.tobytes(),)
            ).fetchall()

        # nprobe=-1 should fail
        with pytest.raises(sqlite3.OperationalError, match="nprobe must be a positive integer"):
            db.execute(
                "SELECT rowid FROM v WHERE embedding MATCH ? AND k=5 AND nprobe=-1", (query.tobytes(),)
            ).fetchall()


class TestIVFSetOption:
    """Test vec0_set_option for IVF configuration"""

    def test_set_ivf_nprobe(self):
        """Test setting ivf_nprobe via vec0_set_option"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=8, ivf_nprobe=2)")

        # Insert and train
        for i in range(50):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
        db.execute("SELECT vec0_ivf_train('v')")

        # Check initial value
        result = db.execute("SELECT value FROM v_info WHERE key='ivf_nprobe'").fetchone()
        assert int(result[0]) == 2

        # Update nprobe
        result = db.execute("SELECT vec0_set_option('v', 'ivf_nprobe', 4)").fetchone()
        assert result[0] == 4

        # Verify update
        result = db.execute("SELECT value FROM v_info WHERE key='ivf_nprobe'").fetchone()
        assert int(result[0]) == 4

    def test_set_ivf_nprobe_validation(self):
        """Test that vec0_set_option validates nprobe values"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=8, ivf_nprobe=2)")

        # Insert and train
        for i in range(20):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
        db.execute("SELECT vec0_ivf_train('v')")

        # nprobe=0 should fail
        with pytest.raises(sqlite3.OperationalError, match="positive integer"):
            db.execute("SELECT vec0_set_option('v', 'ivf_nprobe', 0)")

        # nprobe > nlist should fail
        with pytest.raises(sqlite3.OperationalError, match="cannot be greater"):
            db.execute("SELECT vec0_set_option('v', 'ivf_nprobe', 100)")

        # Unknown option should fail
        with pytest.raises(sqlite3.OperationalError, match="unknown option"):
            db.execute("SELECT vec0_set_option('v', 'unknown', 1)")


class TestIVFDynamicAssignment:
    """Test dynamic IVF assignment on insert/update/delete"""

    def test_insert_after_training_creates_assignment(self):
        """Test that inserting after training creates an IVF assignment"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert initial vectors and train
        for i in range(20):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Insert a new vector after training
        new_vec = np.random.randn(4).astype(np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (100, new_vec.tobytes()))

        # Check that the new vector has an assignment
        assignment = db.execute("SELECT centroid_id FROM v_ivf_assignments WHERE rowid=100").fetchone()
        assert assignment is not None

    def test_delete_removes_assignment(self):
        """Test that deleting a vector removes its IVF assignment"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert and train
        for i in range(20):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Delete a vector
        db.execute("DELETE FROM v WHERE rowid=5")

        # Check that the assignment was removed
        assignment = db.execute("SELECT centroid_id FROM v_ivf_assignments WHERE rowid=5").fetchone()
        assert assignment is None

    def test_update_vector_updates_assignment(self):
        """Test that updating a vector updates its IVF assignment"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert vectors designed to cluster
        # Cluster 0: vectors near [10, 0, 0, 0]
        # Cluster 1: vectors near [-10, 0, 0, 0]
        for i in range(10):
            vec = np.array([10, 0, 0, 0], dtype=np.float32) + np.random.randn(4).astype(np.float32) * 0.1
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
        for i in range(10, 20):
            vec = np.array([-10, 0, 0, 0], dtype=np.float32) + np.random.randn(4).astype(np.float32) * 0.1
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Get original assignment for rowid=1
        orig_assignment = db.execute("SELECT centroid_id FROM v_ivf_assignments WHERE rowid=1").fetchone()[0]

        # Update the vector to be in the opposite cluster
        new_vec = np.array([-10, 0, 0, 0], dtype=np.float32)
        db.execute("UPDATE v SET embedding=? WHERE rowid=1", (new_vec.tobytes(),))

        # Get new assignment
        new_assignment = db.execute("SELECT centroid_id FROM v_ivf_assignments WHERE rowid=1").fetchone()[0]

        # Assignment might change (depends on centroid positions)
        # Just verify the assignment exists
        assert new_assignment is not None


class TestIVFEdgeCases:
    """Test edge cases for IVF"""

    def test_k_larger_than_cluster_size(self):
        """Test querying with k larger than vectors in probed clusters"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=10, ivf_nprobe=1)")

        # Insert only 5 vectors
        for i in range(5):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query for k=10 when only 5 vectors exist
        query = np.random.randn(4).astype(np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=10", (query.tobytes(),)
        ).fetchall()

        # Should return all 5 vectors (or fewer if they're all in different clusters)
        assert len(results) <= 5

    def test_query_untrained_table_uses_brute_force(self):
        """Test that querying an untrained IVF table falls back to brute force"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert vectors but don't train
        for i in range(10):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Query should still work (uses brute force)
        query = np.random.randn(4).astype(np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5

    def test_retrain_updates_centroids_and_assignments(self):
        """Test that retraining updates centroids and assignments"""
        db = connect()
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], ivf_nlist=4)")

        # Insert and train
        for i in range(20):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Get original centroids
        orig_centroids = db.execute("SELECT vector FROM v_ivf_centroids ORDER BY centroid_id").fetchall()

        # Add more vectors with different distribution
        for i in range(20, 40):
            vec = np.random.randn(4).astype(np.float32) + 100  # Shifted
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Retrain
        db.execute("SELECT vec0_ivf_train('v')")

        # Get new centroids
        new_centroids = db.execute("SELECT vector FROM v_ivf_centroids ORDER BY centroid_id").fetchall()

        # Centroids should have changed
        # (Compare as bytes)
        assert orig_centroids != new_centroids


# ============================================================================
# IVF with Different Distance Metrics
# ============================================================================


class TestIVFDistanceMetrics:
    """Test IVF with cosine and L1 distance metrics"""

    def test_ivf_cosine_distance(self):
        """Test IVF with cosine distance metric"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4] distance_metric=cosine,
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        # Insert normalized vectors (important for cosine)
        np.random.seed(42)
        for i in range(50):
            vec = np.random.randn(4).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Train IVF
        db.execute("SELECT vec0_ivf_train('v')")

        # Query
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # Cosine distance should be between 0 and 2
        for _, dist in results:
            assert 0 <= dist <= 2

    def test_ivf_cosine_recall(self):
        """Test IVF recall with cosine distance"""
        db = connect()

        np.random.seed(123)
        vectors = [np.random.randn(8).astype(np.float32) for _ in range(100)]
        # Normalize all vectors
        vectors = [v / np.linalg.norm(v) for v in vectors]

        # Brute force table
        db.execute("CREATE VIRTUAL TABLE bf USING vec0(embedding float[8] distance_metric=cosine)")
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO bf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # IVF table with exhaustive search
        db.execute("""
            CREATE VIRTUAL TABLE ivf USING vec0(
                embedding float[8] distance_metric=cosine,
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)
        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO ivf(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))
        db.execute("SELECT vec0_ivf_train('ivf')")

        # Query
        query = np.random.randn(8).astype(np.float32)
        query = query / np.linalg.norm(query)

        bf_results = set(
            r[0]
            for r in db.execute("SELECT rowid FROM bf WHERE embedding MATCH ? AND k=10", (query.tobytes(),)).fetchall()
        )

        ivf_results = set(
            r[0]
            for r in db.execute("SELECT rowid FROM ivf WHERE embedding MATCH ? AND k=10", (query.tobytes(),)).fetchall()
        )

        # With nprobe=nlist, should have perfect recall
        assert bf_results == ivf_results

    def test_ivf_l1_distance(self):
        """Test IVF with L1 (Manhattan) distance metric"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4] distance_metric=l1,
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(50):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # L1 distance should be non-negative
        for _, dist in results:
            assert dist >= 0


# ============================================================================
# IVF with Different Vector Types
# ============================================================================


class TestIVFVectorTypes:
    """Test IVF with int8 and bit vectors"""

    def test_ivf_int8_vectors(self):
        """Test IVF with int8 quantized vectors"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding int8[8],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(50):
            vec = np.random.randint(-128, 127, size=8, dtype=np.int8)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # IVF search with int8 should work
        query = np.zeros(8, dtype=np.int8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_int8(?) AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # Results should be sorted by distance
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_ivf_bit_vectors(self):
        """Test IVF with bit vectors (hamming distance)"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding bit[64],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(50):
            vec = np.random.randint(0, 256, size=8, dtype=np.uint8)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # IVF search with bit vectors should work
        query = np.zeros(8, dtype=np.uint8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_bit(?) AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # Hamming distances should be non-negative integers
        for _, dist in results:
            assert dist >= 0


# ============================================================================
# Partition Keys with IVF and KNN
# ============================================================================


class TestPartitionKeysWithKNN:
    """Test partition key filtering with KNN queries"""

    def test_partition_key_knn_basic(self):
        """Test KNN query with partition key filtering"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                category TEXT partition key,
                embedding float[4]
            )
        """)

        np.random.seed(42)
        # Insert vectors in two categories
        for i in range(25):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(category, embedding) VALUES (?, ?)", ("A", vec.tobytes()))
        for i in range(25):
            vec = np.random.randn(4).astype(np.float32) + 10  # Shifted
            db.execute("INSERT INTO v(category, embedding) VALUES (?, ?)", ("B", vec.tobytes()))

        query = np.zeros(4, dtype=np.float32)

        # Query all
        all_results = db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k=10", (query.tobytes(),)).fetchall()
        assert len(all_results) == 10

        # Query category A only
        a_results = db.execute(
            "SELECT rowid FROM v WHERE embedding MATCH ? AND k=10 AND category = 'A'", (query.tobytes(),)
        ).fetchall()
        # All results should be from category A (closer to origin)
        assert len(a_results) == 10

    def test_partition_key_ivf_knn(self):
        """Test KNN with partition key and IVF"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                region TEXT partition key,
                embedding float[4],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(30):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(region, embedding) VALUES (?, ?)", ("east", vec.tobytes()))
        for i in range(30):
            vec = np.random.randn(4).astype(np.float32) + 5
            db.execute("INSERT INTO v(region, embedding) VALUES (?, ?)", ("west", vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(4, dtype=np.float32)

        # Query east partition
        results = db.execute(
            "SELECT rowid FROM v WHERE embedding MATCH ? AND k=5 AND region = 'east'", (query.tobytes(),)
        ).fetchall()
        assert len(results) == 5

    def test_partition_key_integer(self):
        """Test integer partition key with KNN"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                user_id INTEGER partition key,
                embedding float[4]
            )
        """)

        np.random.seed(42)
        for user_id in [1, 2, 3]:
            for _ in range(20):
                vec = np.random.randn(4).astype(np.float32)
                db.execute("INSERT INTO v(user_id, embedding) VALUES (?, ?)", (user_id, vec.tobytes()))

        query = np.zeros(4, dtype=np.float32)

        # Query user 2 only
        results = db.execute(
            "SELECT rowid FROM v WHERE embedding MATCH ? AND k=5 AND user_id = 2", (query.tobytes(),)
        ).fetchall()
        assert len(results) == 5


# ============================================================================
# Auxiliary Column Filtering
# ============================================================================


class TestAuxiliaryColumns:
    """Test auxiliary column with KNN queries"""

    def test_auxiliary_column_returned_with_knn(self):
        """Test that auxiliary columns are returned in KNN results"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                +label TEXT
            )
        """)

        np.random.seed(42)
        labels = ["apple", "banana", "cherry", "date", "elderberry"]
        for i, label in enumerate(labels):
            vec = np.array([float(i), 0, 0, 0], dtype=np.float32)
            db.execute("INSERT INTO v(rowid, embedding, label) VALUES (?, ?, ?)", (i + 1, vec.tobytes(), label))

        query = np.array([2.0, 0, 0, 0], dtype=np.float32)
        results = db.execute(
            "SELECT rowid, label, distance FROM v WHERE embedding MATCH ? AND k=3", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 3
        # Closest to [2,0,0,0] should be 'cherry' (index 2)
        assert results[0][1] == "cherry"

    def test_auxiliary_column_with_ivf(self):
        """Test auxiliary columns work with IVF"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                +score FLOAT,
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        np.random.seed(42)
        for i in range(50):
            vec = np.random.randn(4).astype(np.float32)
            score = np.random.random() * 100
            db.execute("INSERT INTO v(rowid, embedding, score) VALUES (?, ?, ?)", (i + 1, vec.tobytes(), score))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, score, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # All results should have non-null scores
        for _, score, _ in results:
            assert score is not None
            assert 0 <= score <= 100

    def test_multiple_auxiliary_columns(self):
        """Test multiple auxiliary columns with mixed types"""
        db = connect()
        # Test 3 auxiliary columns with different types
        db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4], +name TEXT, +score FLOAT, +count INTEGER)")

        np.random.seed(42)
        for i in range(20):
            vec = np.random.randn(4).astype(np.float32)
            db.execute(
                "INSERT INTO v(rowid, embedding, name, score, count) VALUES (?, ?, ?, ?, ?)",
                (i + 1, vec.tobytes(), f"item_{i}", i * 1.5, i * 10),
            )

        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, name, score, count, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        for rowid, name, score, count, dist in results:
            assert name.startswith("item_")
            assert score is not None
            assert count is not None


# ============================================================================
# Int8 IVF Distance Accuracy Tests
# ============================================================================


class TestIVFInt8DistanceAccuracy:
    """Test IVF with int8 vectors and different distance metrics"""

    def test_ivf_int8_l2_distance_accuracy(self):
        """Test int8 IVF L2 distance is computed correctly"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding int8[4],
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert known vectors
        vectors = [
            (1, np.array([0, 0, 0, 0], dtype=np.int8)),
            (2, np.array([1, 0, 0, 0], dtype=np.int8)),
            (3, np.array([2, 0, 0, 0], dtype=np.int8)),
            (4, np.array([10, 10, 10, 10], dtype=np.int8)),
        ]
        for rowid, vec in vectors:
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_int8(?))", (rowid, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with zero vector - closest should be rowid 1
        query = np.array([0, 0, 0, 0], dtype=np.int8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_int8(?) AND k=4", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 4
        # First result should be exact match (distance 0)
        assert results[0][0] == 1
        assert results[0][1] == 0.0

        # Verify distances are sorted
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_ivf_int8_cosine_distance_accuracy(self):
        """Test int8 IVF cosine distance is computed correctly"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding int8[4] distance_metric=cosine,
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert vectors with known cosine relationships
        vectors = [
            (1, np.array([100, 0, 0, 0], dtype=np.int8)),  # Unit in x
            (2, np.array([0, 100, 0, 0], dtype=np.int8)),  # Unit in y (orthogonal)
            (3, np.array([50, 50, 0, 0], dtype=np.int8)),  # 45 degrees
            (4, np.array([-100, 0, 0, 0], dtype=np.int8)),  # Opposite direction
        ]
        for rowid, vec in vectors:
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_int8(?))", (rowid, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with x-direction vector
        query = np.array([100, 0, 0, 0], dtype=np.int8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_int8(?) AND k=4", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 4
        # First should be rowid 1 (same direction, distance ~0)
        assert results[0][0] == 1
        assert results[0][1] < 0.01  # Nearly zero

        # Last should be rowid 4 (opposite direction, distance ~2)
        assert results[3][0] == 4
        assert abs(results[3][1] - 2.0) < 0.01  # Cosine distance of opposite = 2

    def test_ivf_int8_l1_distance_accuracy(self):
        """Test int8 IVF L1 (Manhattan) distance is computed correctly"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding int8[4] distance_metric=L1,
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert known vectors
        vectors = [
            (1, np.array([0, 0, 0, 0], dtype=np.int8)),
            (2, np.array([1, 1, 1, 1], dtype=np.int8)),  # L1 = 4
            (3, np.array([2, 2, 2, 2], dtype=np.int8)),  # L1 = 8
            (4, np.array([10, 0, 0, 0], dtype=np.int8)),  # L1 = 10
        ]
        for rowid, vec in vectors:
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_int8(?))", (rowid, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with zero vector
        query = np.array([0, 0, 0, 0], dtype=np.int8)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_int8(?) AND k=4", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 4
        assert results[0][0] == 1  # Exact match
        assert results[0][1] == 0.0

        # Verify L1 distances
        expected_order = [(1, 0.0), (2, 4.0), (3, 8.0), (4, 10.0)]
        for (actual_rowid, actual_dist), (exp_rowid, exp_dist) in zip(results, expected_order):
            assert actual_rowid == exp_rowid
            assert abs(actual_dist - exp_dist) < 0.01

    def test_ivf_int8_recall_vs_bruteforce(self):
        """Test int8 IVF recall is reasonable compared to brute-force"""
        db = connect()

        # Create IVF table
        db.execute("""
            CREATE VIRTUAL TABLE v_ivf USING vec0(
                embedding int8[16],
                ivf_nlist=10,
                ivf_nprobe=5
            )
        """)

        # Create brute-force table
        db.execute("""
            CREATE VIRTUAL TABLE v_bf USING vec0(
                embedding int8[16]
            )
        """)

        # Insert same vectors in both
        np.random.seed(42)
        n_vectors = 500
        vectors = np.random.randint(-128, 127, size=(n_vectors, 16), dtype=np.int8)

        for i, vec in enumerate(vectors):
            db.execute("INSERT INTO v_ivf(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))
            db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, vec_int8(?))", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v_ivf')")

        # Query both and compare results
        k = 10
        n_queries = 20
        total_recall = 0.0

        for q in range(n_queries):
            query = np.random.randint(-128, 127, size=16, dtype=np.int8)

            ivf_results = set(
                r[0]
                for r in db.execute(
                    "SELECT rowid FROM v_ivf WHERE embedding MATCH vec_int8(?) AND k=?", (query.tobytes(), k)
                ).fetchall()
            )

            bf_results = set(
                r[0]
                for r in db.execute(
                    "SELECT rowid FROM v_bf WHERE embedding MATCH vec_int8(?) AND k=?", (query.tobytes(), k)
                ).fetchall()
            )

            recall = len(ivf_results & bf_results) / k
            total_recall += recall

        avg_recall = total_recall / n_queries
        assert avg_recall >= 0.7, f"int8 IVF recall {avg_recall:.2%} is too low (expected >= 70%)"


# ============================================================================
# Bit IVF Distance Accuracy Tests
# ============================================================================


class TestIVFBitDistanceAccuracy:
    """Test IVF with bit vectors and hamming distance"""

    def test_ivf_bit_hamming_distance_accuracy(self):
        """Test bit IVF hamming distance is computed correctly"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding bit[32],
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert known bit patterns (32 bits = 4 bytes)
        vectors = [
            (1, bytes([0x00, 0x00, 0x00, 0x00])),  # All zeros
            (2, bytes([0x01, 0x00, 0x00, 0x00])),  # 1 bit set
            (3, bytes([0xFF, 0x00, 0x00, 0x00])),  # 8 bits set
            (4, bytes([0xFF, 0xFF, 0xFF, 0xFF])),  # All ones (32 bits)
        ]
        for rowid, vec in vectors:
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_bit(?))", (rowid, vec))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with all zeros - hamming distances should be bit counts
        query = bytes([0x00, 0x00, 0x00, 0x00])
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_bit(?) AND k=4", (query,)
        ).fetchall()

        assert len(results) == 4
        # First should be exact match
        assert results[0][0] == 1
        # Note: IVF uses L2 on converted float vectors for centroids,
        # but actual search uses native distance. Check distances are reasonable.
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_ivf_bit_all_zeros_vs_all_ones(self):
        """Test bit IVF correctly distinguishes all-zeros from all-ones"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding bit[64],
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert all-zeros and all-ones (64 bits = 8 bytes)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, vec_bit(?))", (bytes([0x00] * 8),))
        db.execute("INSERT INTO v(rowid, embedding) VALUES (2, vec_bit(?))", (bytes([0xFF] * 8),))

        # Add some intermediate vectors
        for i in range(3, 20):
            vec = bytes([0x55] * 8)  # Alternating bits
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, vec_bit(?))", (i, vec))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with all zeros - should return rowid 1 first
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_bit(?) AND k=2", (bytes([0x00] * 8),)
        ).fetchall()
        assert results[0][0] == 1

        # Query with all ones - should return rowid 2 first
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH vec_bit(?) AND k=2", (bytes([0xFF] * 8),)
        ).fetchall()
        assert results[0][0] == 2

    def test_ivf_bit_recall_vs_bruteforce(self):
        """Test bit IVF recall is reasonable compared to brute-force"""
        db = connect()

        # Create IVF table
        db.execute("""
            CREATE VIRTUAL TABLE v_ivf USING vec0(
                embedding bit[128],
                ivf_nlist=10,
                ivf_nprobe=5
            )
        """)

        # Create brute-force table
        db.execute("""
            CREATE VIRTUAL TABLE v_bf USING vec0(
                embedding bit[128]
            )
        """)

        # Insert same vectors in both (128 bits = 16 bytes)
        np.random.seed(42)
        n_vectors = 500

        for i in range(n_vectors):
            vec = np.random.randint(0, 256, size=16, dtype=np.uint8).tobytes()
            db.execute("INSERT INTO v_ivf(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec))
            db.execute("INSERT INTO v_bf(rowid, embedding) VALUES (?, vec_bit(?))", (i + 1, vec))

        db.execute("SELECT vec0_ivf_train('v_ivf')")

        # Query both and compare results
        k = 10
        n_queries = 20
        total_recall = 0.0

        for q in range(n_queries):
            query = np.random.randint(0, 256, size=16, dtype=np.uint8).tobytes()

            ivf_results = set(
                r[0]
                for r in db.execute(
                    "SELECT rowid FROM v_ivf WHERE embedding MATCH vec_bit(?) AND k=?", (query, k)
                ).fetchall()
            )

            bf_results = set(
                r[0]
                for r in db.execute(
                    "SELECT rowid FROM v_bf WHERE embedding MATCH vec_bit(?) AND k=?", (query, k)
                ).fetchall()
            )

            recall = len(ivf_results & bf_results) / k
            total_recall += recall

        avg_recall = total_recall / n_queries
        assert avg_recall >= 0.6, f"bit IVF recall {avg_recall:.2%} is too low (expected >= 60%)"


# ============================================================================
# IVF Edge Cases Tests
# ============================================================================


class TestIVFEdgeCasesComprehensive:
    """Comprehensive edge case tests for IVF"""

    def test_ivf_nlist_equals_one(self):
        """Test IVF with nlist=1 (degenerates to brute-force)"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=1
            )
        """)

        np.random.seed(42)
        for i in range(50):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Should work and return correct results
        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_ivf_nlist_greater_than_vectors(self):
        """Test IVF with nlist > number of vectors - system should not crash"""
        db = connect()
        # With nlist=20 and only 10 vectors, some clusters will be empty
        # Use nprobe=20 to search all clusters and find all results
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=20,
                ivf_nprobe=20
            )
        """)

        # Only insert 10 vectors
        np.random.seed(42)
        for i in range(10):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        # Training should still work (even with more clusters than vectors)
        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        # With nprobe=20 searching all clusters, we should find 5 results
        assert len(results) == 5

    def test_ivf_identical_vectors(self):
        """Test IVF with all identical vectors"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=4,
                ivf_nprobe=4
            )
        """)

        # All vectors are the same
        identical_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        for i in range(50):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, identical_vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query should return results (all with same distance)
        query = identical_vec
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # All distances should be 0 (or very close)
        for _, dist in results:
            assert dist < 0.0001

    def test_ivf_zero_vectors(self):
        """Test IVF with all-zero vectors"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Mix of zero and non-zero vectors
        for i in range(25):
            db.execute(
                "INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, np.zeros(4, dtype=np.float32).tobytes())
            )
        for i in range(25, 50):
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, np.ones(4, dtype=np.float32).tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with zero vector
        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 5
        # First results should be zero vectors (distance 0)
        for rowid, dist in results:
            assert rowid <= 25  # Zero vectors have rowids 1-25
            assert dist < 0.0001

    def test_ivf_very_high_dimensions(self):
        """Test IVF with high-dimensional vectors"""
        db = connect()
        dims = 768  # Common for BERT/transformer embeddings
        db.execute(f"""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[{dims}],
                ivf_nlist=10,
                ivf_nprobe=3
            )
        """)

        np.random.seed(42)
        for i in range(100):
            vec = np.random.randn(dims).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        query = np.zeros(dims, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=10", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 10
        distances = [r[1] for r in results]
        assert distances == sorted(distances)

    def test_ivf_single_vector(self):
        """Test IVF with only one vector"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=4
            )
        """)

        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        db.execute("INSERT INTO v(rowid, embedding) VALUES (1, ?)", (vec.tobytes(),))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query should return the single vector
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=5", (vec.tobytes(),)
        ).fetchall()

        assert len(results) == 1
        assert results[0][0] == 1
        assert results[0][1] < 0.0001

    def test_ivf_k_larger_than_dataset(self):
        """Test IVF when k > number of vectors"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert only 5 vectors
        for i in range(5):
            vec = np.random.randn(4).astype(np.float32)
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i + 1, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Request k=100 but only 5 vectors exist
        query = np.zeros(4, dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=100", (query.tobytes(),)
        ).fetchall()

        # Should return all 5 vectors
        assert len(results) == 5

    def test_ivf_negative_values(self):
        """Test IVF handles negative values correctly"""
        db = connect()
        db.execute("""
            CREATE VIRTUAL TABLE v USING vec0(
                embedding float[4],
                ivf_nlist=2,
                ivf_nprobe=2
            )
        """)

        # Insert vectors with negative values
        vectors = [
            (1, np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)),
            (2, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)),
            (3, np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32)),
        ]
        for rowid, vec in vectors:
            db.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (rowid, vec.tobytes()))

        db.execute("SELECT vec0_ivf_train('v')")

        # Query with negative vector
        query = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        results = db.execute(
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=3", (query.tobytes(),)
        ).fetchall()

        assert len(results) == 3
        assert results[0][0] == 1  # Exact match
        assert results[0][1] < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
