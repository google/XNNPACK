// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/hashtable.h>

#include <assert.h> // For assert.
#include <stdbool.h> // For bool.
#include <stddef.h> // For size_t.

#include <xnnpack/allocator.h> // For xnn_allocate_zero_memory and xnn_release_memory.
#include <xnnpack/math.h>      // For is_po2.

#define XNN_HASH_TABLE_INITIAL_BUCKETS 32
#define XNN_HASH_TABLE_MAX_LOAD 0.75
#define XNN_HASH_TABLE_GROWTH_FACTOR 2

static bool hash_table_init(struct xnn_hash_table* table, size_t n)
{
  table->buckets = (struct xnn_hash_bucket*) xnn_allocate_zero_memory(n * sizeof(struct xnn_hash_bucket));
  if (table->buckets == NULL) {
    return false;
  }

  table->num_buckets = n;
  table->num_entries = 0;
  return true;
}

bool xnn_hash_table_init(struct xnn_hash_table* table)
{
  return hash_table_init(table, XNN_HASH_TABLE_INITIAL_BUCKETS);
}

// Look up hash in table, returning the bucket index where hash would go.
// Output param found is set to true if we found the entry, false if we did not
// find it. Returning the bucket index allows us to easily insert an entry into
// that location without probing again.
static size_t lookup(struct xnn_hash_table* table, size_t hash, bool* found)
{
  assert(is_po2(table->num_buckets));
  const size_t mask = table->num_buckets - 1;
  size_t idx = hash & mask;
  const struct xnn_hash_bucket* buckets = table->buckets;
  // Linear probing.
  while (buckets[idx].hash != hash && buckets[idx].value != NULL) {
    idx = (idx + 1) & mask;
  }
  *found = buckets[idx].value == NULL ? false : true;
  return idx;
}

// Forward declare circular usages, grow uses insert.
static void hash_table_grow(struct xnn_hash_table* table);

bool xnn_hash_table_insert(struct xnn_hash_table* table, size_t hash, void* entry)
{
  if (((float) table->num_entries / table->num_buckets) > XNN_HASH_TABLE_MAX_LOAD) {
    hash_table_grow(table);
  }

  bool found;
  const size_t idx = lookup(table, hash, &found);
  if (found) {
    return false;
  }
  else {
    table->buckets[idx].hash = hash;
    table->buckets[idx].value = entry;
    table->num_entries++;
    return true;
  }
}

void* xnn_hash_table_lookup(struct xnn_hash_table* table, size_t hash)
{
  bool found;
  const size_t bucket_idx = lookup(table, hash, &found);
  return table->buckets[bucket_idx].value;
}

void xnn_hash_table_release(struct xnn_hash_table* table)
{
  xnn_release_memory(table->buckets);
}

static void hash_table_grow(struct xnn_hash_table* table)
{
  struct xnn_hash_table tmp;
  const size_t new_num_buckets = table->num_buckets * XNN_HASH_TABLE_GROWTH_FACTOR;
  hash_table_init(&tmp, new_num_buckets);

  for (size_t i = 0; i < table->num_buckets; i++) {
    struct xnn_hash_bucket b = table->buckets[i];
    if (b.value != NULL) {
      xnn_hash_table_insert(&tmp, b.hash, b.value);
    }
  }

  xnn_hash_table_release(table);

  table->buckets = tmp.buckets;
  table->num_buckets = tmp.num_buckets;
  table->num_entries = tmp.num_entries;
}
