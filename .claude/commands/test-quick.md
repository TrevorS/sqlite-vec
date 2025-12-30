---
description: Run a quick subset of tests for fast iteration
---

Run a quick subset of critical tests for fast development iteration:

```bash
python3 -m pytest -vv tests/test-loadable.py -k "test_vec0_knn or test_vec0_inserts or test_vec_distance" --tb=short
```

This runs:
- test_vec0_knn - Core KNN search functionality
- test_vec0_inserts - Vector insertion
- test_vec_distance_* - Distance calculations

For full test coverage, use `/test` instead.
