// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "xnnpack/codecache.h"

#include <assert.h> // For assert.
#include <stddef.h> // For size_t.
#include <stdint.h> // For uint32_t.

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"

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

static uint32_t murmur_hash3(const void* key, size_t len, uint32_t seed)
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

static inline size_t cache_size(struct xnn_cache* cache) {
  switch (cache->type) {
    case xnn_cache_type_code:
      return cache->code.size;
    case xnn_cache_type_weights:
      return cache->weights.size;
    default:
      XNN_UNREACHABLE;
  }
  return SIZE_MAX;
}

static inline void* cache_start(struct xnn_cache* cache) {
  switch (cache->type) {
    case xnn_cache_type_code:
      return cache->code.start;
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
  cache->num_entries = 0;
  cache->hits = 0;
  cache->misses = 0;
  return xnn_status_success;
}

enum xnn_status xnn_init_code_cache_with_size(struct xnn_code_cache* cache, size_t num_buckets)
{
  memset(cache, 0, sizeof(struct xnn_code_cache));
  enum xnn_status status = xnn_status_success;
  status = xnn_init_cache_with_size(&cache->cache, num_buckets, xnn_cache_type_code);
  if (status != xnn_status_success) {
    goto error;
  }

  status = xnn_allocate_code_memory(&cache->cache.code, XNN_DEFAULT_CODE_BUFFER_SIZE);
  if (status != xnn_status_success) {
    goto error;
  }

  return xnn_status_success;

error:
  xnn_release_code_cache(cache);
  return status;
}

enum xnn_status xnn_init_code_cache(struct xnn_code_cache* cache)
{
  return xnn_init_code_cache_with_size(cache, XNN_CACHE_INITIAL_BUCKETS);
}

static bool cache_buckets_grow(struct xnn_cache* cache)
{
  struct xnn_code_cache tmp_code_cache;
  struct xnn_cache* tmp_cache = NULL;
  const size_t new_num_buckets = cache->num_buckets * XNN_CACHE_GROWTH_FACTOR;
  if (cache->type == xnn_cache_type_code) {
    bool init_ok = xnn_init_code_cache_with_size(&tmp_code_cache, new_num_buckets) == xnn_status_success;
    if (!init_ok) {
      return false;
    }
    tmp_cache = &tmp_code_cache.cache;
  } else {
    // TODO(zhin): Unsupported for now.
    assert(false);
  }

  for (size_t i = 0; i < cache->num_buckets; i++) {
    struct xnn_cache_bucket b = cache->buckets[i];
    if (b.size == 0) {
      continue;
    }

    // Find the first empty slot by linear probing to insert. No need to check
    // hashes since we are not looking up anything, just moving things around
    // into a bigger hash table.
    const size_t mask = cache->num_buckets - 1;
    size_t idx = b.hash & mask;
    while (tmp_cache->buckets[idx].size != 0) {
      idx = (idx + 1) & mask;
    }
    tmp_cache->buckets[idx].size = b.size;
    tmp_cache->buckets[idx].offset = b.offset;
  }

  xnn_release_memory(cache->buckets);

  cache->buckets = tmp_cache->buckets;
  cache->num_buckets = tmp_cache->num_buckets;
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
      return false;
    }
  }

  // Check that ptr points into cache's buffer.
  assert((uintptr_t) ptr >= (uintptr_t) cache_start(cache));
  if (cache->type == xnn_cache_type_code) {
    assert((uintptr_t) ptr < (uintptr_t) cache_start(cache) + cache_size(cache));
  }

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
    if (cache->type == xnn_cache_type_code) {
      // Found in the cache, rewind the buffer because code generators update buffer size.
      cache->code.size -= size;
    }
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

size_t xnn_get_or_insert_code_cache(struct xnn_code_cache* cache, void* ptr, size_t size)
{
  return xnn_get_or_insert_cache(&cache->cache, ptr, size);
}

enum xnn_status xnn_release_code_cache(struct xnn_code_cache* cache)
{
  if XNN_LIKELY(cache != NULL) {
    assert(cache->cache.type == xnn_cache_type_code);
    xnn_release_code_memory(&cache->cache.code);
    xnn_release_memory(cache->cache.buckets);
  }
  return xnn_status_success;
}

enum xnn_status xnn_init_weights_cache_with_size(struct xnn_weights_cache* cache, size_t num_buckets)
{
  memset(cache, 0, sizeof(struct xnn_weights_cache));

  enum xnn_status status = xnn_status_success;
  status = xnn_init_cache_with_size(&cache->cache, num_buckets, xnn_cache_type_weights);
  if (status != xnn_status_success) {
    goto error;
  }

  status = xnn_allocate_weights_memory(&cache->cache.weights, XNN_DEFAULT_WEIGHTS_BUFFER_SIZE);
  if (status != xnn_status_success) {
    goto error;
  }

  return xnn_status_success;

error:
  xnn_release_weights_cache(cache);
  return status;
}

enum xnn_status xnn_init_weights_cache(struct xnn_weights_cache* cache)
{
  return xnn_init_weights_cache_with_size(cache, XNN_CACHE_INITIAL_BUCKETS);
}

enum xnn_status xnn_release_weights_cache(struct xnn_weights_cache* cache)
{
  if XNN_LIKELY(cache != NULL) {
    assert(cache->cache.type == xnn_cache_type_weights);
    xnn_release_weights_memory(&cache->cache.weights);
    xnn_release_memory(cache->cache.buckets);
  }
  return xnn_status_success;
}

size_t xnn_get_or_insert_weights_cache(struct xnn_weights_cache* cache, void* ptr, size_t size)
{
  return xnn_get_or_insert_cache(&cache->cache, ptr, size);
}
