---
description: Explore the KNN search implementation for understanding or modification
---

Explore the core KNN search implementation in sqlite-vec.c:

Key functions to read:

1. **vec0Filter_knn_chunks_iter()** (lines 6451-6802)
   - Main brute-force search loop
   - Iterates chunks, computes distances, maintains top-K

2. **vec0Filter_knn()** (lines 6804-6900)
   - Query entry point
   - Validates query vector, calls chunks_iter

3. **min_idx()** (lines 5774-5814)
   - Top-K selection within a chunk
   - O(k×n) selection algorithm

4. **merge_sorted_lists()** (lines 5660-5721)
   - Merges chunk results with global top-K
   - Standard two-pointer merge

5. **Distance functions** (lines 361-600)
   - distance_l2_sqr_float(), distance_cosine_float(), etc.
   - SIMD-optimized versions (AVX/NEON)

Read these sections to understand how KNN currently works and where ANN would integrate.
