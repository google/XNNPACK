// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include "include/xnnpack.h"

extern "C" {

xnn_status xnn_create_weights_cache_with_size(
    size_t size, xnn_weights_cache_t* weights_cache_out) {
  *weights_cache_out = (xnn_weights_cache_t)5;
  return xnn_status_success;
}

xnn_status xnn_create_weights_cache(xnn_weights_cache_t* weights_cache_out) {
  *weights_cache_out = (xnn_weights_cache_t)5;
  return xnn_status_success;
}

xnn_status xnn_finalize_weights_cache(
    xnn_weights_cache_t weights_cache,
    xnn_weights_cache_finalization_kind finalization_kind) {
  return xnn_status_success;
}

// Wrapper function of the function pointers in `xnn_weights_cache_t`.
bool xnn_weights_cache_is_finalized(xnn_weights_cache_t cache) { return true; }

xnn_status xnn_delete_weights_cache(xnn_weights_cache_t weights_cache) {
  return xnn_status_success;
}

}  // extern "C"
