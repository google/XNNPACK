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

#define XNN_CODE_CACHE_HASH_SEED 7
#define XNN_CODE_CACHE_INITIAL_BUCKETS 32
#define XNN_CODE_CACHE_MAX_LOAD 0.75
// Max load factor is 0.75 (3/4), i.e. num_entries / num_buckets > 3 / 4.
#define XNN_CODE_CACHE_MAX_LOAD_ENTRIES_MULTIPLIER 4
#define XNN_CODE_CACHE_MAX_LOAD_BUCKETS_MULTIPLIER 3
#define XNN_CODE_CACHE_GROWTH_FACTOR 2

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

enum xnn_status xnn_init_code_cache_with_size(struct xnn_code_cache* cache, size_t num_buckets)
{
  memset(cache, 0, sizeof(struct xnn_code_cache));

  cache->buckets = (struct xnn_cache_bucket*) xnn_allocate_zero_memory(num_buckets * sizeof(struct xnn_cache_bucket));

  enum xnn_status status;
  if (cache->buckets == NULL) {
    xnn_log_error("fail to allocate memory for JIT cache buckets");
    status = xnn_status_out_of_memory;
    goto error;
  }

  status = xnn_allocate_code_memory(&cache->code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  if (status != xnn_status_success) {
    goto error;
  }

  cache->num_buckets = num_buckets;
  cache->num_entries = 0;
  cache->hits = 0;
  cache->misses = 0;
  return xnn_status_success;

error:
  xnn_release_code_cache(cache);
  return status;
}

enum xnn_status xnn_init_code_cache(struct xnn_code_cache* cache)
{
  return xnn_init_code_cache_with_size(cache, XNN_CODE_CACHE_INITIAL_BUCKETS);
}

static bool code_cache_buckets_grow(struct xnn_code_cache* cache)
{
  struct xnn_code_cache tmp;
  const size_t new_num_buckets = cache->num_buckets * XNN_CODE_CACHE_GROWTH_FACTOR;
  bool init_ok = xnn_init_code_cache_with_size(&tmp, new_num_buckets) == xnn_status_success;
  if (!init_ok) {
    return false;
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
    while (tmp.buckets[idx].size != 0) {
      idx = (idx + 1) & mask;
    }
    tmp.buckets[idx].size = b.size;
    tmp.buckets[idx].offset = b.offset;
  }

  xnn_release_memory(cache->buckets);

  cache->buckets = tmp.buckets;
  cache->num_buckets = tmp.num_buckets;
  return true;
}

static inline bool code_equals(struct xnn_code_cache* cache, struct xnn_code_span code_span, size_t size, size_t offset)
{
  return code_span.size == size &&
         memcmp(code_span.code, (void*) ((uintptr_t) cache->code_buffer.code + offset), size) == 0;
}

static bool cache_lookup(struct xnn_code_cache* cache, struct xnn_code_span code_span, uint32_t hash, size_t* index)
{
  assert(is_po2(cache->num_buckets));
  const size_t mask = cache->num_buckets - 1;
  size_t idx = hash & mask;
  const struct xnn_cache_bucket* buckets = cache->buckets;

  // Linear probing.
  while (buckets[idx].size != 0 &&
         !(buckets[idx].hash == hash && code_equals(cache, code_span, buckets[idx].size, buckets[idx].offset))) {
    idx = (idx + 1) & mask;
  }
  *index = idx;
  if (buckets[idx].size == 0) {
    return false;
  } else {
    return true;
  }
}

bool xnn_code_cache_insert(struct xnn_code_cache* cache, struct xnn_code_span code_span)
{
  uint32_t hash = murmur_hash3(code_span.code, code_span.size, /*seed=*/XNN_CODE_CACHE_HASH_SEED);
  size_t idx;
  const bool found = cache_lookup(cache, code_span, hash, &idx);
  if (found) {
    return false;
  }

  // Ensure we have enough buckets to keep under our load limit.
  if (cache->num_entries * XNN_CODE_CACHE_MAX_LOAD_ENTRIES_MULTIPLIER >
      cache->num_buckets * XNN_CODE_CACHE_MAX_LOAD_BUCKETS_MULTIPLIER) {
    if (!code_cache_buckets_grow(cache)) {
      // Can't grow hash table anymore.
      return false;
    }
  }

  // Record the offset before we grow the buffer.
  size_t offset = cache->code_buffer.size;

  // Ensure we have enough space in the cache's code_buffer.
  if (cache->code_buffer.size + code_span.size > cache->code_buffer.capacity) {
    const enum xnn_status status = xnn_grow_code_memory(&cache->code_buffer, XNN_DEFAULT_MICROKERNEL_SIZE);
    if (xnn_status_success != status) {
      return false;
    }
    assert(cache->code_buffer.size + code_span.size <= cache->code_buffer.capacity);
  }

  // Copy to end of cache's code_buffer.
  memcpy((void*) ((uintptr_t) cache->code_buffer.code + cache->code_buffer.size), code_span.code, code_span.size);
  cache->code_buffer.size += code_span.size;

  // Insert the entry.
  cache->buckets[idx].size = code_span.size;
  cache->buckets[idx].hash = hash;
  cache->buckets[idx].offset = offset;
  cache->num_entries++;
  return true;
}

size_t xnn_code_cache_lookup(struct xnn_code_cache* code_cache, struct xnn_code_span code_span)
{
  uint32_t hash = murmur_hash3(code_span.code, code_span.size, /*seed=*/XNN_CODE_CACHE_HASH_SEED);
  size_t bucket_idx;
  if (cache_lookup(code_cache, code_span, hash, &bucket_idx)) {
    code_cache->hits++;
    return code_cache->buckets[bucket_idx].offset;
  } else {
    code_cache->misses++;
    return XNN_CODE_CACHE_NOT_FOUND;
  }
}

size_t xnn_code_cache_get_or_insert(struct xnn_code_cache* cache, struct xnn_code_span code_span)
{
  const size_t found_offset = xnn_code_cache_lookup(cache, code_span);
  if (found_offset != XNN_CODE_CACHE_NOT_FOUND) {
    return found_offset;
  }

  const size_t code_offset = cache->code_buffer.size;
  if (!xnn_code_cache_insert(cache, code_span)) {
    return XNN_CODE_CACHE_NOT_FOUND;
  }
  return code_offset;
}

enum xnn_status xnn_release_code_cache(struct xnn_code_cache* cache)
{
  xnn_release_code_memory(&cache->code_buffer);
  xnn_release_memory(cache->buckets);
  return xnn_status_success;
}
