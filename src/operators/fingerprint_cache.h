// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_OPERATORS_FINGERPRINT_CACHE_H_
#define XNNPACK_SRC_OPERATORS_FINGERPRINT_CACHE_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "include/xnnpack.h"
#include "src/operators/fingerprint_id.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct fingerprint_context {
  enum xnn_status status;
  enum xnn_fingerprint_id fingerprint_id;
  struct xnn_weights_cache_provider cache;
  xnn_operator_t op;
};

// Creates a fingerprinting context with an explicit fingerprint id.
struct fingerprint_context create_fingerprint_context(
    enum xnn_fingerprint_id fingerprint_id);

// Releases the fingerprinting resources.
//
// If the context status is uninitialized, a new fingerprinting entry is created
// for the context's `fingerprint_id` and the value computed by
// `fingerprint_cache_get_fingerprint(&context->cache)`.
void finalize_fingerprint_context(struct fingerprint_context* context);

// Retrieves the fingerprint value from a fingerprinting cache that was created
// by `create_fingerprint_context`.
uint32_t fingerprint_cache_get_fingerprint(
    const struct xnn_weights_cache_provider* provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // XNNPACK_SRC_OPERATORS_FINGERPRINT_CACHE_H_
