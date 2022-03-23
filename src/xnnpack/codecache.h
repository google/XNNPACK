// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>            // For size_t.
#include <stdint.h>            // For uint32_t.
#include <xnnpack.h>           // For xnn_status.
#include <xnnpack/allocator.h> // For xnn_code_buffer.

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_CACHE_NOT_FOUND SIZE_MAX // Return value when code is not found in the cache.

struct xnn_byte_span {
  void* start;
  size_t size;
};

// A cache for arbitrary bytes.
// The implementation is similar to a hash table with open addressing and linear
// probing, but restricted to our use cases.

// Similar to buckets in a hash table implementation, this is an entry in the
// cache. It stores "metadata" about the generated code (size and offset). The
// actual bytes are in the cache's buffer.
struct xnn_cache_bucket {
  // A hash for quick comparison.
  uint32_t hash;
  // Size of bytes.
  size_t size;
  // Offset of bytes, relative to cache's buffer.
  size_t offset;
};

enum xnn_cache_type {
  xnn_cache_type_invalid = 0,
  xnn_cache_type_code,
  xnn_cache_type_weights,
};

struct xnn_cache {
  enum xnn_cache_type type;
  // A growing buffer that is used to keep all generated code or repacked weights.
  union {
    struct xnn_code_buffer code;
    struct xnn_weights_buffer weights;
  };

  // Entries in the cache.
  struct xnn_cache_bucket* buckets;
  // Capacity of the cache, when the load factor (num_entries/num_buckets) grows
  // beyond a limit, the cache is expanded.
  size_t num_buckets;
  size_t num_entries;
  size_t hits;
  size_t misses;
};

// A cache for JIT generated microkernel code.
struct xnn_code_cache {
  struct xnn_cache cache;
};

enum xnn_status xnn_init_code_cache(struct xnn_code_cache* cache);
enum xnn_status xnn_release_code_cache(struct xnn_code_cache* cache);
// Looks up byte_span in the cache, returns offset into cache's buffer if found.
// byte_span should already point into cache->buffer.
// If it already exists within the cache, the buffer will be rewound, so we can
// reuse the same section of the buffer.
size_t xnn_code_cache_get_or_insert(struct xnn_code_cache* cache, struct xnn_byte_span byte_span);

// A cache for repacked weights.
struct xnn_weights_cache {
  struct xnn_cache cache;
};

enum xnn_status xnn_init_weights_cache(struct xnn_weights_cache* cache);
enum xnn_status xnn_release_weights_cache(struct xnn_weights_cache* cache);

#ifdef __cplusplus
} // extern "C"
#endif
