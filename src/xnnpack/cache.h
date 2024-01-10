// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>          // For size_t.
#include <stdint.h>          // For uint32_t.
#include <xnnpack.h>         // For xnn_status.
#include <xnnpack/common.h>  // For XNN_INLINE.
#include <xnnpack/memory.h>  // For xnn_code_buffer.
#include <xnnpack/mutex.h>   // For xnn_mutex.

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_CACHE_NOT_FOUND \
  SIZE_MAX  // Return value when code is not found in the cache.

// Murmur hash (https://en.wikipedia.org/wiki/MurmurHash) on the buffer specified by `key` and `size`.
uint32_t murmur_hash3(const void* key, size_t len, uint32_t seed);

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
  // A growing buffer that is used to keep all generated code or repacked
  // weights.
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
// Looks up `ptr` in the cache, returns offset into cache's buffer if found.
// `ptr` should already point into cache->buffer.
// If it already exists within the cache, the buffer will be rewound, so we can
// reuse the same section of the buffer.
size_t xnn_get_or_insert_code_cache(struct xnn_code_cache* cache, void* ptr,
                                    size_t size);

XNN_INLINE static bool xnn_code_cache_valid(struct xnn_code_cache* code_cache) {
  return code_cache != NULL && code_cache->cache.type == xnn_cache_type_code;
}

// The state of weights cache finalization.
enum xnn_cache_state {
  // Not finalized.
  xnn_cache_state_not_finalized,
  // The underlying memory is trimmed to be as compact as possible.
  xnn_cache_state_hard_finalized,
  // The underlying memory has some extra space at the end.
  xnn_cache_state_soft_finalized,
};

// Internal implementation of cache for repacked weights.
struct xnn_internal_weights_cache {
  struct xnn_cache cache;
  // Protects updates of `cache`, it has the same lifetime as `cache`, and so
  // should be initialized/destroyed together with the `cache`.
  struct xnn_mutex mutex;
  // Maximum size of packed weights that have been inserted into the cache.
  size_t max_weights_size;
  enum xnn_cache_state finalization_state;
};

enum xnn_status xnn_internal_init_weights_cache_with_size(struct xnn_internal_weights_cache* cache, size_t size);

enum xnn_status xnn_internal_finalize_weights_cache(
  struct xnn_internal_weights_cache* cache, enum xnn_weights_cache_finalization_kind finalization_kind);

enum xnn_status xnn_internal_release_weights_cache(struct xnn_internal_weights_cache* cache);

// Ensures that cache has enough space for `n` bytes, locks the mutex to protect
// future updates. Mutex must be unlocked using xnn_internal_get_or_insert_weights_cache.
void* xnn_internal_reserve_space_in_weights_cache(struct xnn_internal_weights_cache* cache, size_t n);

// Looks up packed weights at `ptr` in the cache. If it is found, reuse it.
// Otherwise, it is added to the cache. Mutex must already be locked before
// calling this, it will be unlocked at the end of this function.
size_t xnn_internal_get_or_insert_weights_cache(
  struct xnn_internal_weights_cache* cache, const struct xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size);

bool xnn_internal_weights_cache_is_finalized(struct xnn_internal_weights_cache* cache);

size_t xnn_internal_weights_cache_look_up(
  struct xnn_internal_weights_cache* cache, const struct xnn_weights_cache_look_up_key* cache_key);

void* xnn_internal_weights_cache_offset_to_addr(struct xnn_internal_weights_cache* weights_cache, size_t offset);

enum xnn_status xnn_internal_delete_weights_cache(struct xnn_internal_weights_cache* weights_cache);

// Wrapper function of the function pointers in `xnn_weights_cache_t`.
bool xnn_weights_cache_is_finalized(xnn_weights_cache_t cache);

size_t xnn_look_up_or_insert_weights_cache(
  xnn_weights_cache_t cache, const struct xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size);

size_t xnn_weights_cache_look_up(
  xnn_weights_cache_t cache, const struct xnn_weights_cache_look_up_key* cache_key);

#ifdef __cplusplus
}  // extern "C"
#endif
