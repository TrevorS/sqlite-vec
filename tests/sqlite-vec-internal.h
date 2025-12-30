#include <stdlib.h>

int min_idx(
  // list of distances, size n
  const float *distances,
  // number of entries in distances
  int32_t n,
  // output array of size k, the indicies of the lowest k values in distances
  int32_t *out,
  // output number of elements
  int32_t k
);

// Parse a partition key column definition like "user_id integer partition key"
int vec0_parse_partition_key_definition(
  const char *source,
  int source_length,
  const char **out_column_name,
  int *out_column_name_length,
  int *out_column_type
);
