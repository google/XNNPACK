// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __MACH__
#define _POSIX_C_SOURCE 199309L
#endif

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/memory.h"
#include "src/xnnpack/params.h"



enum xnn_status xnn_create_weights_cache_with_size(size_t size, xnn_weights_cache_t* weights_cache_out)
{
  struct xnn_weights_cache_provider* cache_provider = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create weights cache: XNNPACK is not initialized");
    goto error;
  }

  cache_provider = xnn_allocate_zero_memory(sizeof(struct xnn_weights_cache_provider));
  if (cache_provider == NULL) {
    xnn_log_error("failed to allocate %zu bytes for weights cache provider descriptor", sizeof(struct xnn_weights_cache_provider));
    goto error;
  }

  cache_provider->context = xnn_allocate_zero_memory(sizeof(struct xnn_internal_weights_cache));
  if (cache_provider->context == NULL) {
    xnn_log_error("failed to allocate %zu bytes for weights cache descriptor", sizeof(struct xnn_internal_weights_cache));
    goto error;
  }

  status = xnn_internal_init_weights_cache_with_size(cache_provider->context, size);
  if (status != xnn_status_success) {
    goto error;
  }
  cache_provider->look_up = (size_t(*)(void*, const struct xnn_weights_cache_look_up_key*))xnn_internal_weights_cache_look_up;
  cache_provider->reserve_space = (void*(*)(void*, size_t))xnn_internal_reserve_space_in_weights_cache;
  cache_provider->look_up_or_insert = (size_t (*)(void*, const struct xnn_weights_cache_look_up_key*, void*, size_t))xnn_internal_get_or_insert_weights_cache;
  cache_provider->is_finalized = (bool (*)(void*))xnn_internal_weights_cache_is_finalized;
  cache_provider->offset_to_addr = (void*(*)(void*, size_t))xnn_internal_weights_cache_offset_to_addr;
  cache_provider->delete_cache = (enum xnn_status (*)(void*))xnn_internal_delete_weights_cache;
  *weights_cache_out = cache_provider;
  return xnn_status_success;

error:
  if (cache_provider != NULL) {
    xnn_internal_release_weights_cache(cache_provider->context);
  }
  return status;
}

enum xnn_status xnn_create_weights_cache(xnn_weights_cache_t* weights_cache_out)
{
  return xnn_create_weights_cache_with_size(XNN_DEFAULT_WEIGHTS_BUFFER_SIZE, weights_cache_out);
}

enum xnn_status xnn_delete_weights_cache(xnn_weights_cache_t weights_cache)
{
  if XNN_LIKELY(weights_cache != NULL) {
    enum xnn_status status = xnn_internal_release_weights_cache(weights_cache->context);
    if (status != xnn_status_success) {
      return status;
    }
    xnn_release_memory(weights_cache->context);
    xnn_release_memory(weights_cache);
  }
  return xnn_status_success;
}
