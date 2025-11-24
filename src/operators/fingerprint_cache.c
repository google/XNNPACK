// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/operators/fingerprint_cache.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/operators/fingerprint_id.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/init-once.h"
#include "src/xnnpack/mutex.h"

#define XNN_FINGERPRINT_MAX_COUNT 256
static struct xnn_fingerprint fingerprint_vector[XNN_FINGERPRINT_MAX_COUNT];
static uint32_t fingerprint_vector_size = 0;

static struct xnn_mutex mutex;
XNN_INIT_ONCE_GUARD(mutex);

static void init_mutex_config() { xnn_mutex_init(&mutex); }

const struct xnn_fingerprint* xnn_get_fingerprint(const uint32_t id) {
  XNN_INIT_ONCE(mutex);
  uint32_t i = 0;
  xnn_mutex_lock(&mutex);
  for (; i < fingerprint_vector_size; ++i) {
    if (fingerprint_vector[i].id == id) {
      break;
    }
  }
  xnn_mutex_unlock(&mutex);
  return i < fingerprint_vector_size ? fingerprint_vector + i : NULL;
}

void xnn_set_fingerprint(const struct xnn_fingerprint fingerprint) {
  XNN_INIT_ONCE(mutex);
  uint32_t i = 0;
  xnn_mutex_lock(&mutex);
  for (; i < fingerprint_vector_size; ++i) {
    if (fingerprint_vector[i].id == fingerprint.id) {
      fingerprint_vector[i] = fingerprint;
      break;
    }
  }
  if (i >= fingerprint_vector_size) {
    assert(fingerprint_vector_size < XNN_FINGERPRINT_MAX_COUNT);
    fingerprint_vector[fingerprint_vector_size++] = fingerprint;
  }
  xnn_mutex_unlock(&mutex);
}

void xnn_clear_fingerprints() {
  XNN_INIT_ONCE(mutex);
  xnn_mutex_lock(&mutex);
  fingerprint_vector_size = 0;
  xnn_mutex_unlock(&mutex);
}

// The context for an XNNPack weight cache provider that we pass to operator
// `create` functions when we want to fingerprint them.
struct fingerprint_cache_context {
  void* buffer;
  size_t bytes;
  uint32_t hash;
};

static size_t fingerprint_cache_look_up(
    void* context, const struct xnn_weights_cache_look_up_key* cache_key) {
  return XNN_CACHE_NOT_FOUND;
}

static void* fingerprint_cache_reserve_space(void* const context, size_t n) {
  struct fingerprint_cache_context* const ctx = context;
  assert(ctx);
  if (ctx->buffer && ctx->bytes < n) {
    xnn_release_simd_memory(ctx->buffer);
    ctx->buffer = NULL;
  }
  if (ctx->buffer == NULL) {
    ctx->buffer = xnn_allocate_simd_memory(n);
    ctx->bytes = ctx->buffer ? n : 0;
  }
  return ctx->buffer;
}

static size_t fingerprint_cache_look_up_or_insert(
    void* context, const struct xnn_weights_cache_look_up_key* cache_key,
    void* ptr, size_t size) {
  assert(context);
  struct fingerprint_cache_context* const ctx = context;
  ctx->hash = murmur_hash3(ptr, size, /*seed=*/ctx->hash);
  return 0;
}

static bool fingerprint_cache_is_finalized(void* context) { return false; }

static void* fingerprint_cache_offset_to_addr(void* context, size_t offset) {
  assert(context);
  struct fingerprint_cache_context* const ctx = context;
  return ctx->buffer;
}

static enum xnn_status fingerprint_cache_delete_cache(void* context) {
  struct fingerprint_cache_context* const ctx = context;
  if (ctx) {
    if (ctx->buffer) {
      xnn_release_simd_memory(ctx->buffer);
    }
    *ctx = (struct fingerprint_cache_context){0};
  }
  return xnn_status_success;
}

struct fingerprint_context create_fingerprint_context(
    const enum xnn_fingerprint_id fingerprint_id) {
  struct fingerprint_context context = {
      .status = xnn_status_uninitialized,
      .fingerprint_id = fingerprint_id,
      .cache =
          (struct xnn_weights_cache_provider){
              .context = NULL,
              .look_up = fingerprint_cache_look_up,
              .reserve_space = fingerprint_cache_reserve_space,
              .look_up_or_insert = fingerprint_cache_look_up_or_insert,
              .is_finalized = fingerprint_cache_is_finalized,
              .offset_to_addr = fingerprint_cache_offset_to_addr,
              .delete_cache = fingerprint_cache_delete_cache},
      .op = NULL,
  };
  if (context.fingerprint_id == xnn_fingerprint_id_unknown) {
    context.status = xnn_status_unsupported_parameter;
  } else if (xnn_get_fingerprint(context.fingerprint_id)) {
    context.status = xnn_status_success;
  } else {
    // Do this after the checks to avoid a memory allocation when unnecessary.
    context.cache.context =
        xnn_allocate_zero_memory(sizeof(struct fingerprint_cache_context));
  }
  return context;
}

static void free_fingerprint_cache_provider(
    struct xnn_weights_cache_provider* const provider) {
  if (provider) {
    provider->delete_cache(provider->context);
    if (provider->context) {
      xnn_release_memory(provider->context);
    }
  }
}

void finalize_fingerprint_context(struct fingerprint_context* const context) {
  assert(context);
  if (context->status == xnn_status_uninitialized) {
    xnn_set_fingerprint((struct xnn_fingerprint){
        .id = context->fingerprint_id,
        .value = fingerprint_cache_get_fingerprint(&context->cache)});
  }
  free_fingerprint_cache_provider(&context->cache);
  if (context->op) {
    xnn_delete_operator(context->op);
  }
}

uint32_t fingerprint_cache_get_fingerprint(
    const struct xnn_weights_cache_provider* const provider) {
  assert(provider);
  assert(provider->context);
  return ((struct fingerprint_cache_context*)provider->context)->hash;
}

// A simple C implementation of a pseudo-random number generator to generate
// random looking deterministic data to initialize fingerprint buffers.
//
// See https://en.wikipedia.org/wiki/Xorshift#xorshift+.
//
// Note: we do not need anything that is cryptographically secure. We only want
// random looking data.
struct Xorshift128PlusState {
  uint64_t s0;
  uint64_t s1;
};

// Advances the pseudo-random number generator by one step and returns data.
static uint64_t xorshift128plus_next(struct Xorshift128PlusState* state) {
  uint64_t s1 = state->s0;
  uint64_t s0 = state->s1;
  const uint64_t result = s0 + s1;
  s1 ^= s1 << 23;
  state->s0 = s0;
  state->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
  return result;
}

// Initializes the state of the pseudo-random number generator and advances it
// by a few steps to avoid low quality initial values.
static struct Xorshift128PlusState xorshift128plus_init(uint64_t seed) {
  struct Xorshift128PlusState state = {seed, 0};
  for (int i = 0; i < 10; ++i) {
    xorshift128plus_next(&state);
  }
  return state;
}

void fill_fingerprint_buffer(uint8_t* data, size_t bytes) {
  struct Xorshift128PlusState prng_state = xorshift128plus_init(0xafb69024c);
  while (bytes >= sizeof(uint64_t)) {
    *((uint64_t*)(data)) = xorshift128plus_next(&prng_state);
    bytes -= sizeof(uint64_t);
    data += sizeof(uint64_t);
  }
  const uint64_t tail = xorshift128plus_next(&prng_state);
  for (int i = 0; i < bytes; ++i) {
    data[i] = tail >> (i * 8);
  }
}
