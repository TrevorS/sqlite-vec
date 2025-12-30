# IVF (Approximate Nearest Neighbor Search)

By default, `sqlite-vec` uses brute-force search which scans every vector in the table. For large datasets, you can enable **IVF (Inverted File Index)** for approximate nearest neighbor (ANN) search, which is significantly faster at the cost of some recall accuracy.

## How IVF Works

IVF partitions vectors into clusters using k-means clustering. During a query, only a subset of clusters (controlled by `nprobe`) are searched instead of the entire dataset.

- **Training**: Clusters vectors into `nlist` centroids using k-means
- **Querying**: Finds the `nprobe` nearest centroids, then searches only those clusters
- **Trade-off**: Higher `nprobe` = better recall but slower queries

## Creating an IVF-enabled Table

Add `ivf_nlist` and optionally `ivf_nprobe` to your table definition:

```sql
create virtual table vec_documents using vec0(
  document_id integer primary key,
  contents_embedding float[768],
  ivf_nlist=100,      -- Number of clusters
  ivf_nprobe=10       -- Clusters to search (default)
);
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `ivf_nlist` | Number of clusters to create | Required for IVF |
| `ivf_nprobe` | Number of clusters to search per query | 10 |

**Guidelines for `nlist`:**
- Rule of thumb: `sqrt(n)` where `n` is the number of vectors
- For 1M vectors: ~1000 clusters
- For 100K vectors: ~300 clusters

## Training the Index

After inserting vectors, train the IVF index:

```sql
-- Insert your vectors first
insert into vec_documents(document_id, contents_embedding)
  select id, embedding from documents;

-- Train the IVF index
select vec0_train('vec_documents');
```

The training step:
1. Runs k-means clustering on all vectors
2. Creates centroids for each cluster
3. Assigns each vector to its nearest centroid

### Check Training Status

```sql
select vec0_is_trained('vec_documents');
-- Returns 1 if trained, 0 if not
```

## Querying with IVF

Once trained, KNN queries automatically use the IVF index:

```sql
select document_id, distance
from vec_documents
where contents_embedding match :query
  and k = 10;
```

### Override nprobe at Query Time

For individual queries, you can override the default `nprobe`:

```sql
select document_id, distance
from vec_documents
where contents_embedding match :query
  and k = 10
  and ivf_nprobe = 20;  -- Search more clusters for better recall
```

### Set nprobe for All Queries

Use `vec0_set_option` to change the default `nprobe`:

```sql
select vec0_set_option('vec_documents', 'ivf_nprobe', 50);
```

## Retraining

If you've added many new vectors, retrain to update cluster assignments:

```sql
-- Retrain after significant data changes
select vec0_train('vec_documents');
```

New vectors inserted after training are automatically assigned to the nearest existing centroid.

## Performance Comparison

| Dataset Size | Brute Force | IVF (nprobe=10) | Recall |
|--------------|-------------|-----------------|--------|
| 10K vectors  | ~5ms        | ~2ms            | ~95%   |
| 100K vectors | ~50ms       | ~5ms            | ~92%   |
| 1M vectors   | ~500ms      | ~15ms           | ~90%   |

*Times are approximate and depend on vector dimensions and hardware.*

## Supported Vector Types

IVF works with all vector types:

```sql
-- float32 vectors
create virtual table v1 using vec0(
  embedding float[768], ivf_nlist=100
);

-- int8 vectors
create virtual table v2 using vec0(
  embedding int8[768], ivf_nlist=100
);

-- bit vectors
create virtual table v3 using vec0(
  embedding bit[768], ivf_nlist=100
);
```

## Supported Distance Metrics

IVF works with all distance metrics:

```sql
-- L2 distance (default)
create virtual table v1 using vec0(
  embedding float[768], ivf_nlist=100
);

-- Cosine distance
create virtual table v2 using vec0(
  embedding float[768] distance_metric=cosine, ivf_nlist=100
);

-- L1 distance
create virtual table v3 using vec0(
  embedding float[768] distance_metric=L1, ivf_nlist=100
);
```

## Best Practices

1. **Choose `nlist` based on dataset size**: Use `sqrt(n)` as a starting point
2. **Start with low `nprobe`**: Begin with 10 and increase if recall is too low
3. **Retrain periodically**: If >20% new data is added, consider retraining
4. **Test recall**: Compare IVF results against brute-force to verify acceptable recall
5. **Use with partition keys**: IVF combines well with partition keys for even faster queries

## Limitations

- Training requires all vectors to be inserted first
- Deleting large portions of data may degrade cluster quality (retrain if needed)
- Memory usage increases with `nlist` (centroids are stored in memory)
