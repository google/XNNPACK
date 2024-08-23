// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>  // For assert.
#include <stddef.h>  // For size_t.
#include <stdint.h>  // For uint32_t.
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/memory.h"
#include "xnnpack/mutex.h"

#define XNN_CACHE_HASH_SEED 7
#define XNN_CACHE_INITIAL_BUCKETS 32
#define XNN_CACHE_MAX_LOAD 0.75
// Max load factor is 0.75 (3/4), i.e. num_entries / num_buckets > 3 / 4.
#define XNN_CACHE_MAX_LOAD_ENTRIES_MULTIPLIER 4
#define XNN_CACHE_MAX_LOAD_BUCKETS_MULTIPLIER 3
#define XNN_CACHE_GROWTH_FACTOR 2

// MurmurHash3 implementation, copied from smhasher, with minor modifications in
// style and main loop.

static inline uint32_t fmix32(uint32_t h)
{
  h ^= h >> 16;
  h *= UINT32_C(0x85EBCA6B);
  h ^= h >> 13;
  h *= UINT32_C(0xC2B2AE35);
  h ^= h >> 16;

  return h;
}

uint32_t murmur_hash3(const void* key, size_t len, uint32_t seed)
{
  const uint8_t* data = (const uint8_t*) key;

  uint32_t h1 = seed;

  const uint32_t c1 = UINT32_C(0xCC9E2D51);
  const uint32_t c2 = UINT32_C(0x1B873593);

  const uint32_t* blocks = (const uint32_t*) data;
  for (; len >= sizeof(uint32_t); len -= sizeof(uint32_t)) {
    uint32_t k1 = *blocks++;

    k1 *= c1;
    k1 = math_rotl_u32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = math_rotl_u32(h1, 13);
    h1 = h1 * 5 + UINT32_C(0xE6546B64);
  }

  const uint8_t* tail = (const uint8_t*) blocks;

  uint32_t k1 = 0;

  switch (len & 3) {
    case 3:
      k1 ^= tail[2] << 16;
    case 2:
      k1 ^= tail[1] << 8;
    case 1:
      k1 ^= tail[0];
      k1 *= c1;
      k1 = math_rotl_u32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  };

  h1 ^= len;

  return fmix32(h1);
}

static inline void* cache_start(struct xnn_cache* cache) {
  switch (cache->type) {
    case xnn_cache_type_weights:
      return cache->weights.start;
    default:
      XNN_UNREACHABLE;
  }
  return NULL;
}

enum xnn_status xnn_init_cache_with_size(struct xnn_cache* cache, size_t num_buckets, enum xnn_cache_type cache_type)
{
  memset(cache, 0, sizeof(struct xnn_cache));
  cache->buckets = (struct xnn_cache_bucket*) xnn_allocate_zero_memory(num_buckets * sizeof(struct xnn_cache_bucket));
  if (cache->buckets == NULL) {
    xnn_log_error("fail to allocate memory for cache buckets");
    return xnn_status_out_of_memory;
  }

  cache->type = cache_type;
  cache->num_buckets = num_buckets;
  return xnn_status_success;
}

static bool cache_buckets_grow(struct xnn_cache* cache)
{
  const size_t new_num_buckets = cache->num_buckets * XNN_CACHE_GROWTH_FACTOR;
  assert(is_po2(new_num_buckets));
  struct xnn_cache tmp_cache;
  xnn_init_cache_with_size(&tmp_cache, new_num_buckets, cache->type);

  for (size_t i = 0; i < cache->num_buckets; i++) {
    struct xnn_cache_bucket b = cache->buckets[i];
    if (b.size == 0) {
      continue;
    }

    // Find the first empty slot by linear probing to insert. No need to check
    // hashes since we are not looking up anything, just moving things around
    // into a bigger hash table.
    const size_t mask = tmp_cache.num_buckets - 1;
    size_t idx = b.hash & mask;
    while (tmp_cache.buckets[idx].size != 0) {
      idx = (idx + 1) & mask;
    }
    tmp_cache.buckets[idx].hash = b.hash;
    tmp_cache.buckets[idx].size = b.size;
    tmp_cache.buckets[idx].offset = b.offset;
  }

  xnn_release_memory(cache->buckets);

  cache->buckets = tmp_cache.buckets;
  cache->num_buckets = tmp_cache.num_buckets;
  return true;
}

static inline bool bytes_equal(struct xnn_cache* cache, void* ptr, size_t size, size_t offset)
{
  return memcmp(ptr, (void*) ((uintptr_t) cache_start(cache) + offset), size) == 0;
}

static bool lookup(struct xnn_cache* cache, void* ptr, size_t size, uint32_t hash, size_t* index)
{
  assert(is_po2(cache->num_buckets));
  const size_t mask = cache->num_buckets - 1;
  size_t idx = hash & mask;
  const struct xnn_cache_bucket* buckets = cache->buckets;

  // Linear probing.
  while (buckets[idx].size != 0 &&
         !(buckets[idx].hash == hash &&
           size == buckets[idx].size &&
           bytes_equal(cache, ptr, buckets[idx].size, buckets[idx].offset))) {
    idx = (idx + 1) & mask;
  }
  *index = idx;
  if (buckets[idx].size == 0) {
    return false;
  } else {
    return true;
  }
}

static bool insert(struct xnn_cache* cache, void* ptr, size_t size)
{
  const uint32_t hash = murmur_hash3(ptr, size, /*seed=*/XNN_CACHE_HASH_SEED);
  size_t idx;
  const bool found = lookup(cache, ptr, size, hash, &idx);
  if (found) {
    return false;
  }

  // Ensure we have enough buckets to keep under our load limit.
  if (cache->num_entries * XNN_CACHE_MAX_LOAD_ENTRIES_MULTIPLIER >
      cache->num_buckets * XNN_CACHE_MAX_LOAD_BUCKETS_MULTIPLIER) {
    if (!cache_buckets_grow(cache)) {
      // Can't grow hash table anymore.
      xnn_log_error("failed to grow cache buckets");
      return false;
    }
    xnn_log_debug("successfully grew cache buckets");

    // If the cache grew, idx is stale, since that is based on the old cache's num_buckets.
    const bool found_in_grown_cache = lookup(cache, ptr, size, hash, &idx);
    assert(!found_in_grown_cache);
    (void) found_in_grown_cache;  // Silence unused variable warnings.
  }

  // Check that ptr points into cache's buffer.
  assert((uintptr_t) ptr >= (uintptr_t) cache_start(cache));

  const size_t offset = (uintptr_t) ptr - (uintptr_t) cache_start(cache);

  // Insert the entry.
  cache->buckets[idx].size = size;
  cache->buckets[idx].hash = hash;
  cache->buckets[idx].offset = offset;
  cache->num_entries++;
  return true;
}

// Checks if a generated microkernel is already in the cache, returns the offset
// if found, XNN_CACHE_NOT_FOUND otherwise.
static size_t lookup_cache(struct xnn_cache* cache, void* ptr, size_t size)
{
  const uint32_t hash = murmur_hash3(ptr, size, /*seed=*/XNN_CACHE_HASH_SEED);
  size_t bucket_idx;
  if (lookup(cache, ptr, size, hash, &bucket_idx)) {
    cache->hits++;
    return cache->buckets[bucket_idx].offset;
  } else {
    cache->misses++;
    return XNN_CACHE_NOT_FOUND;
  }
}

size_t xnn_get_or_insert_cache(struct xnn_cache* cache, void* ptr, size_t size)
{
  const size_t found_offset = lookup_cache(cache, ptr, size);
  if (found_offset != XNN_CACHE_NOT_FOUND) {
    return found_offset;
  }

  if (cache->type == xnn_cache_type_weights) {
    // Cache miss, weights packing functions don't update buffer size, update it here.
    cache->weights.size += size;
  }

  const size_t offset = (uintptr_t) ptr - (uintptr_t) cache_start(cache);
  if (!insert(cache, ptr, size)) {
    return XNN_CACHE_NOT_FOUND;
  }
  return offset;
}

enum xnn_status xnn_internal_init_weights_cache(
  struct xnn_internal_weights_cache* cache,
  size_t num_buckets,
  size_t buffer_size)
{
  memset(cache, 0, sizeof(struct xnn_internal_weights_cache));

  enum xnn_status status = xnn_status_success;
  status = xnn_init_cache_with_size(&cache->cache, num_buckets, xnn_cache_type_weights);
  if (status != xnn_status_success) {
    goto error;
  }

  status = xnn_allocate_weights_memory(&cache->cache.weights, buffer_size);
  if (status != xnn_status_success) {
    goto error;
  }

  status = xnn_mutex_init(&cache->mutex);
  if (status != xnn_status_success) {
    goto error;
  }

  return xnn_status_success;

error:
  xnn_internal_release_weights_cache(cache);
  return status;
}

enum xnn_status xnn_internal_init_weights_cache_with_size(struct xnn_internal_weights_cache* cache, size_t size)
{
  return xnn_internal_init_weights_cache(cache, XNN_CACHE_INITIAL_BUCKETS, size);
}

enum xnn_status xnn_internal_finalize_weights_cache(
  struct xnn_internal_weights_cache* cache, enum xnn_weights_cache_finalization_kind finalization_kind)
{
  switch (cache->finalization_state) {
    case xnn_cache_state_hard_finalized:
    case xnn_cache_state_soft_finalized:
      xnn_log_error("failed to finalize an already final weights cache");
      return xnn_status_invalid_state;
    case xnn_cache_state_not_finalized: {
      enum xnn_status status;
      enum xnn_cache_state finalized_state;

      if (finalization_kind == xnn_weights_cache_finalization_kind_hard) {
        xnn_log_debug("hard finalizing weights cache");
        status = xnn_finalize_weights_memory(&cache->cache.weights);
        // Also release the memory used by hash table (but not the weights memory).
        xnn_release_memory(cache->cache.buckets);
        cache->cache.buckets = NULL;
        finalized_state = xnn_cache_state_hard_finalized;
      } else {
        xnn_log_debug("soft finalizing weights cache");
        assert(finalization_kind == xnn_weights_cache_finalization_kind_soft);
        // Finalize weights cache by reserving sufficient space for the insertion of the largest cached weights. This
        // ensures that we have space to write packed weights to check for cache hits without growing and moving the
        // memory. This has some memory overhead, which can be as large as the size of the largest cached weights,
        // rounded up to page size.
        status = xnn_reserve_weights_memory(&cache->cache.weights, cache->max_weights_size);
        finalized_state = xnn_cache_state_soft_finalized;
      }
      if (status != xnn_status_success) {
        xnn_log_error("failed to finalize weights cache memory");
        return xnn_status_invalid_state;
      }

      cache->finalization_state = finalized_state;
      return xnn_status_success;
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_internal_release_weights_cache(struct xnn_internal_weights_cache* cache)
{
  if XNN_LIKELY(cache != NULL) {
    assert(cache->cache.type == xnn_cache_type_weights);
    xnn_release_weights_memory(&cache->cache.weights);
    if (cache->cache.buckets != NULL) {
      xnn_release_memory(cache->cache.buckets);
    }
    const enum xnn_status status = xnn_mutex_destroy(&cache->mutex);
    if (status != xnn_status_success) {
      return status;
    }
  }
  return xnn_status_success;
}

static inline bool cache_has_space(
  struct xnn_internal_weights_cache* cache, size_t n)
{
  const struct xnn_weights_buffer buf = cache->cache.weights;
  return buf.size + n <= buf.capacity;
}

void* xnn_internal_reserve_space_in_weights_cache(struct xnn_internal_weights_cache* cache, size_t n)
{
  switch (cache->finalization_state) {
    case xnn_cache_state_hard_finalized:
      xnn_log_error("cannot reserve additional space in a finalized compact weights cache");
      return NULL;
    case xnn_cache_state_soft_finalized:
      if (!cache_has_space(cache, n)) {
        xnn_log_error("cannot reserve additional space in a finalized weights cache");
        return NULL;
      }
      // If the cache is finalized, and has space for `n` bytes, we still want to lock the mutex, because we can have
      // multiple writers attempting to write to this space.
      break;
    default:
      break;
  }

  enum xnn_status status = xnn_mutex_lock(&cache->mutex);
  if (status != xnn_status_success) {
    return NULL;
  }

  struct xnn_weights_buffer* buffer = &cache->cache.weights;
  status = xnn_reserve_weights_memory(buffer, n);
  if (status != xnn_status_success) {
    xnn_mutex_unlock(&cache->mutex);
    return NULL;
  }

  return (void*) ((uintptr_t) buffer->start + buffer->size);
}

size_t xnn_internal_get_or_insert_weights_cache(
  struct xnn_internal_weights_cache* cache, const struct xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size)
{
  size_t offset = XNN_CACHE_NOT_FOUND;

  switch (cache->finalization_state) {
    case xnn_cache_state_hard_finalized: {
      xnn_log_error("cannot insert into a finalized compact weights cache");
      return XNN_CACHE_NOT_FOUND;
    }
    case xnn_cache_state_soft_finalized: {
      // Inserting into a finalized weights cache is okay as long as:
      // 1. there is sufficient space in the memory (to write the incoming packed weights), or
      // 2. incoming packed weights is already in cache
      if (!cache_has_space(cache, size)) {
        xnn_log_error("insufficient extra space in finalized weights cache buffer");
        return XNN_CACHE_NOT_FOUND;
      }

      // We need to release the mutex from this point onwards, because xnn_reserve_space_in_weights would have returned
      // non-NULL (which means that it locked the mutex).
      const size_t found_offset = lookup_cache(&cache->cache, ptr, size);
      if (found_offset == XNN_CACHE_NOT_FOUND) {
        xnn_log_error("packed weights not found in finalized weights cache");
      }

      offset = found_offset;
      break;
    }
    case xnn_cache_state_not_finalized: {
      offset = xnn_get_or_insert_cache(&cache->cache, ptr, size);
      if (offset != XNN_CACHE_NOT_FOUND) {
        // Found or inserted packed weights, update the largest size seen so far, this will be used when finalizing the
        // weights cache, to ensure there is an extra space at the end for future cache checks.
        cache->max_weights_size = max(size, cache->max_weights_size);
      }
      break;
    }
  }

  // Mutex is locked in xnn_reserve_space_in_weights_cache when it returns non-NULL, i.e. when cache is not finalized,
  // or if it is xnn_cache_state_soft_finalized and has sufficient space.
  const enum xnn_status status = xnn_mutex_unlock(&cache->mutex);
  (void) status;
  assert(status == xnn_status_success);
  return offset;
}

bool xnn_internal_weights_cache_is_finalized(struct xnn_internal_weights_cache* cache)
{
  return cache->finalization_state != xnn_cache_state_not_finalized;
}

size_t xnn_internal_weights_cache_look_up(
  struct xnn_internal_weights_cache* cache, const struct xnn_weights_cache_look_up_key* cache_key)
{
  // The default implementation does not support this query.
  return XNN_CACHE_NOT_FOUND;
}

void* xnn_internal_weights_cache_offset_to_addr(struct xnn_internal_weights_cache* weights_cache, size_t offset)
{
  return (void*) ((uintptr_t)weights_cache->cache.weights.start + offset);
}

enum xnn_status xnn_internal_delete_weights_cache(struct xnn_internal_weights_cache* weights_cache)
{
  enum xnn_status status = xnn_internal_release_weights_cache(weights_cache);
  if (status != xnn_status_success) {
    return status;
  }
  xnn_release_memory(weights_cache);
  return xnn_status_success;
}

bool xnn_weights_cache_is_finalized(xnn_weights_cache_t cache)
{
  return cache->is_finalized(cache->context);
}

size_t xnn_look_up_or_insert_weights_cache(
  xnn_weights_cache_t cache, const struct xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size)
{
  return cache->look_up_or_insert(cache->context, cache_key, ptr, size);
}

enum xnn_status xnn_finalize_weights_cache(
  xnn_weights_cache_t weights_cache, enum xnn_weights_cache_finalization_kind finalization_kind)
{
  return xnn_internal_finalize_weights_cache(weights_cache->context, finalization_kind);
}

size_t xnn_weights_cache_look_up(
  xnn_weights_cache_t cache, const struct xnn_weights_cache_look_up_key* cache_key)
{
  return cache->look_up(cache->context, cache_key);
}
