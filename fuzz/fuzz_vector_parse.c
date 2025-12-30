/**
 * Fuzz harness for vector parsing functions.
 *
 * Tests:
 * - vec_f32() with arbitrary blob data
 * - vec_int8() with arbitrary blob data
 * - vec_bit() with arbitrary blob data
 * - JSON vector parsing
 * - Vector length calculation
 *
 * Build: make fuzz-build
 * Run: make fuzz-vector
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "sqlite3.h"
#include "sqlite-vec.h"

// External declaration for sqlite3_vec_init
#ifdef __cplusplus
extern "C" {
#endif
int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);
#ifdef __cplusplus
}
#endif

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0) return 0;

    sqlite3 *db = NULL;
    sqlite3_stmt *stmt = NULL;
    int rc;

    // Open in-memory database
    rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        sqlite3_close(db);
        return 0;
    }

    // Initialize sqlite-vec
    char *errmsg = NULL;
    rc = sqlite3_vec_init(db, &errmsg, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_free(errmsg);
        sqlite3_close(db);
        return 0;
    }

    // Test vec_length with arbitrary blob
    rc = sqlite3_prepare_v2(db, "SELECT vec_length(?)", -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        sqlite3_bind_blob(stmt, 1, data, size, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // Test vec_f32 with arbitrary blob (should validate or reject)
    rc = sqlite3_prepare_v2(db, "SELECT vec_f32(?)", -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        sqlite3_bind_blob(stmt, 1, data, size, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // Test vec_int8 with arbitrary blob
    rc = sqlite3_prepare_v2(db, "SELECT vec_int8(?)", -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        sqlite3_bind_blob(stmt, 1, data, size, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // Test vec_bit with arbitrary blob
    rc = sqlite3_prepare_v2(db, "SELECT vec_bit(?)", -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        sqlite3_bind_blob(stmt, 1, data, size, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // Test JSON parsing (interpret data as string if it looks like JSON)
    if (size > 0 && (data[0] == '[' || data[0] == '{')) {
        // Create null-terminated string
        char *json = malloc(size + 1);
        if (json) {
            memcpy(json, data, size);
            json[size] = '\0';

            rc = sqlite3_prepare_v2(db, "SELECT vec_f32(?)", -1, &stmt, NULL);
            if (rc == SQLITE_OK) {
                sqlite3_bind_text(stmt, 1, json, size, SQLITE_STATIC);
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);
            }
            free(json);
        }
    }

    // Test vec_normalize
    rc = sqlite3_prepare_v2(db, "SELECT vec_normalize(?)", -1, &stmt, NULL);
    if (rc == SQLITE_OK) {
        sqlite3_bind_blob(stmt, 1, data, size, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // Test vec_slice with various parameters
    if (size >= 8) {
        int start = data[0] % 100;
        int end = data[1] % 100;
        char sql[128];
        snprintf(sql, sizeof(sql), "SELECT vec_slice(?, %d, %d)", start, end);
        rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
        if (rc == SQLITE_OK) {
            sqlite3_bind_blob(stmt, 1, data + 2, size - 2, SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    sqlite3_close(db);
    return 0;
}
