// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>           // For xnn_caches_t, xnn_operator_t.
#include <xnnpack/allocator.h> // For XNN_ALLOCATION_ALIGNMENT.
#include <xnnpack/cache.h>     // For xnn_caches.
#include <xnnpack/operator.h>  // For xnn_operator definition.

void* xnn_get_pointer_to_write_weights(
  xnn_operator_t op,
  xnn_caches_t caches,
  size_t aligned_weights_size,
  int padding_byte)
{
  assert(aligned_weights_size % XNN_ALLOCATION_ALIGNMENT == 0);
  void* weights_ptr = NULL;
  if (use_weights_cache(caches)) {
    weights_ptr = xnn_reserve_space_in_weights_cache(caches->weights_cache, aligned_weights_size);
    if (weights_ptr == NULL) {
      return NULL;
    }
  } else {
    op->packed_weights.pointer = xnn_allocate_simd_memory(aligned_weights_size);
    if (op->packed_weights.pointer == NULL) {
      return NULL;
    }
    weights_ptr = op->packed_weights.pointer;
  }
  memset(weights_ptr, padding_byte, aligned_weights_size);
  return weights_ptr;
}
