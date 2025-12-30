# sqlite-vec Development Guide

sqlite-vec is a SQLite extension for vector search, written in C.

## Quick Start

```bash
make help          # Show all available targets
make setup         # Install Python dependencies (uses uv)
make loadable      # Build the extension
make test          # Run tests
```

## Project Structure

- `sqlite-vec.c` - Main implementation (~10K lines)
- `sqlite-vec.h` - Public API header
- `tests/` - Python test suite (pytest + hypothesis)
- `fuzz/` - C fuzzing harnesses

## Key Make Targets

| Target | Description |
|--------|-------------|
| `make loadable` | Build shared library (vec0.so/dylib) |
| `make test` | Run Python tests |
| `make test-unit` | Run C unit tests |
| `make test-property` | Run property-based tests |
| `make format` | Format C and Python code |
| `make lint` | Run linters |

## Testing

```bash
make test              # Core tests
make test-property     # Hypothesis property tests
make test-all          # All tests
make test-unit         # C unit tests
```

## Architecture

### vec0 Virtual Table

The `vec0` virtual table stores vectors and supports KNN queries:

```sql
CREATE VIRTUAL TABLE movies USING vec0(embedding float[128]);
INSERT INTO movies(rowid, embedding) VALUES (1, ?);
SELECT * FROM movies WHERE embedding MATCH ? AND k = 10;
```

### Shadow Tables

For a table named `movies`:
- `movies_chunks` - Chunk metadata
- `movies_rowids` - Rowid mapping
- `movies_vector_chunks00` - Vector data

### Query Plans

- `FULLSCAN` - SELECT * FROM table
- `POINT` - WHERE rowid = ?
- `KNN` - WHERE col MATCH ? AND k = ?

## Adding Features

### New SQL Function

1. Implement in sqlite-vec.c
2. Register in `sqlite3_vec_init()` via `sqlite3_create_function_v2()`
3. Add tests in `tests/test-loadable.py`

### New Distance Metric

1. Implement `distance_<metric>_float()` and `distance_<metric>_int8()`
2. Add to `enum Vec0DistanceMetrics`
3. Update switch in `vec0Filter_knn_chunks_iter()`
