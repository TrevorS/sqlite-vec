/**
 * Fuzz harness for KNN query execution.
 *
 * Tests:
 * - vec0 table creation with various parameters
 * - Vector insertion with arbitrary data
 * - KNN queries with arbitrary query vectors
 * - Edge cases in k values
 *
 * Build: make fuzz-build
 * Run: make fuzz-knn
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "sqlite3.h"
#include "sqlite-vec.h"

#ifdef __cplusplus
extern "C" {
#endif
int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);
#ifdef __cplusplus
}
#endif

// Helper to extract parameters from fuzz input
typedef struct {
    uint8_t dimensions;      // 1-255 dimensions
    uint8_t num_vectors;     // number of vectors to insert
    uint8_t k;               // k for KNN query
    uint8_t chunk_size;      // chunk_size parameter
    uint8_t flags;           // misc flags
} FuzzParams;

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < sizeof(FuzzParams) + 4) return 0;

    // Extract parameters from input
    FuzzParams params;
    memcpy(&params, data, sizeof(FuzzParams));
    data += sizeof(FuzzParams);
    size -= sizeof(FuzzParams);

    // Sanitize parameters
    int dimensions = (params.dimensions % 64) + 1;  // 1-64 dimensions
    int num_vectors = (params.num_vectors % 50) + 1; // 1-50 vectors
    int k = (params.k % 20) + 1;  // 1-20
    int chunk_size = ((params.chunk_size % 7) + 1) * 64;  // 64-512

    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    int rc;
    char sql[512];

    // Open database
    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        sqlite3_close(db);
        return 0;
    }

    // Initialize extension
    char *errmsg = NULL;
    rc = sqlite3_vec_init(db, &errmsg, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return 0;
    }

    // Create vec0 table with fuzzed parameters
    snprintf(sql, sizeof(sql),
        "CREATE VIRTUAL TABLE v USING vec0(embedding float[%d], chunk_size=%d)",
        dimensions, chunk_size);
    rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return 0;
    }

    // Prepare insert statement
    rc = sqlite3_prepare_v2(db, "INSERT INTO v(rowid, embedding) VALUES (?, ?)", -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_close(db);
        return 0;
    }

    // Insert vectors from fuzz data
    size_t vec_size = dimensions * sizeof(float);
    for (int i = 0; i < num_vectors && size >= vec_size; i++) {
        sqlite3_bind_int64(stmt, 1, i + 1);
        sqlite3_bind_blob(stmt, 2, data, vec_size, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
        data += vec_size;
        size -= vec_size;
    }
    sqlite3_finalize(stmt);

    // Perform KNN query with remaining data as query vector
    if (size >= vec_size) {
        snprintf(sql, sizeof(sql),
            "SELECT rowid, distance FROM v WHERE embedding MATCH ? AND k=%d",
            k > num_vectors ? num_vectors : k);
        rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
        if (rc == SQLITE_OK) {
            sqlite3_bind_blob(stmt, 1, data, vec_size, SQLITE_STATIC);
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                // Consume results
                (void)sqlite3_column_int64(stmt, 0);
                (void)sqlite3_column_double(stmt, 1);
            }
            sqlite3_finalize(stmt);
        }
    }

    // Test with different distance metrics if we have flags
    if (params.flags & 0x01) {
        snprintf(sql, sizeof(sql),
            "CREATE VIRTUAL TABLE v_cos USING vec0(embedding float[%d] distance=cosine)",
            dimensions);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
    }

    if (params.flags & 0x02) {
        snprintf(sql, sizeof(sql),
            "CREATE VIRTUAL TABLE v_l1 USING vec0(embedding float[%d] distance=l1)",
            dimensions);
        sqlite3_exec(db, sql, NULL, NULL, NULL);
    }

    // Test IVF if flag is set
    if (params.flags & 0x04 && num_vectors >= 10) {
        snprintf(sql, sizeof(sql),
            "CREATE VIRTUAL TABLE v_ivf USING vec0(embedding float[%d], ivf_nlist=4)",
            dimensions);
        if (sqlite3_exec(db, sql, NULL, NULL, NULL) == SQLITE_OK) {
            // Insert some vectors
            rc = sqlite3_prepare_v2(db, "INSERT INTO v_ivf(rowid, embedding) VALUES (?, ?)", -1, &stmt, NULL);
            if (rc == SQLITE_OK) {
                // Rewind data pointer (reuse vectors)
                const uint8_t *vec_data = data - (num_vectors * vec_size);
                for (int i = 0; i < num_vectors && i < 10; i++) {
                    sqlite3_bind_int64(stmt, 1, i + 1);
                    sqlite3_bind_blob(stmt, 2, vec_data + (i * vec_size), vec_size, SQLITE_STATIC);
                    sqlite3_step(stmt);
                    sqlite3_reset(stmt);
                }
                sqlite3_finalize(stmt);

                // Train IVF
                sqlite3_exec(db, "SELECT vec0_ivf_train('v_ivf')", NULL, NULL, NULL);
            }
        }
    }

    sqlite3_close(db);
    return 0;
}
