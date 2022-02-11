// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdbool.h> // For bool.
#include <stddef.h>  // For size_t.

#ifdef __cplusplus
extern "C" {
#endif

// Hash table implementation using open addressing with linear probing.
// See https://en.wikipedia.org/wiki/Hash_table.
//
// This doesn't actually hash the key, the user should hash their key and provide
// the hash to this table.
//
// Values are pointers, they are not copied into hash table, users must ensure
// that they exist for the lifetime of the hash table they are inserted into.
//
// An NULL value is indistinguishable from a missing entry.

// Bucket (or slots) in the hash table.
struct xnn_hash_bucket {
  size_t hash;
  void* value;
};

// A hash table is a list of `num_buckets` buckets. We record the actual number
// of entries in the table. When the load (num_entries/num_buckets) exceed a
// certain number (XNN_HASH_TABLE_MAX_LOW), we grow the table (by XNN_HASH_TABLE_GROWH_FACTOR).
struct xnn_hash_table {
  struct xnn_hash_bucket* buckets;
  size_t num_buckets;
  size_t num_entries;
};

// Initializes a xnn_hash_table. Returns false if initialization failed, true otherwise.
bool xnn_hash_table_init(struct xnn_hash_table* table);

// Inserts entry, keyed by hash, into table. Returns true if insertion succeeded
// (i.e. entry did not exists), false if insertion failed (entry with same hash
// exists).
bool xnn_hash_table_insert(struct xnn_hash_table* table, size_t hash, void* entry);

// Looks up an entry by hash inside table. Returns the value (a pointer), NULL
// if not found. An entry with a value of NULL is not distinguishable from a
// missing entry.
void* xnn_hash_table_lookup(struct xnn_hash_table* table, size_t hash);

// Release all memory allocated by the table (buckets).
void xnn_hash_table_release(struct xnn_hash_table* table);

#ifdef __cplusplus
} // extern "C"
#endif
