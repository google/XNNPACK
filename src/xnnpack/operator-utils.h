// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"

static inline void* packed_weights(struct xnn_operator* op) {
  if (op->weights_cache == NULL) {
    return op->packed_weights.pointer;
  } else {
    return op->weights_cache->offset_to_addr(op->weights_cache->context, op->packed_weights.offset);
  }
}

static inline bool use_weights_cache(struct xnn_operator* op) {
  return op->weights_cache != NULL;
}

// Get a pointer to a region to pack weights into. If weights cache is available, use it, returning to a pointer to the
// cache's buffer, otherwise, allocate and return a pointer to a new region. Returns NULL on error.
XNN_INTERNAL void* xnn_get_pointer_to_write_weights(
  xnn_operator_t op,
  size_t aligned_weights_size,
  int padding_byte);

#ifdef __cplusplus
extern "C" {
#endif
XNN_INTERNAL size_t xnn_compute_convolution_output_dimension(
  size_t padded_input_dimension,
  size_t kernel_dimension,
  size_t dilation_dimension,
  size_t subsampling_dimension);

XNN_INTERNAL size_t xnn_compute_deconvolution_output_dimension(
  size_t input_dimension,
  size_t output_padding_dimension,
  size_t adjustment_dimension,
  size_t kernel_dimension,
  size_t dilation_dimension,
  size_t stride_dimension);

XNN_INTERNAL size_t xnn_compute_unpooling_output_dimension(
  size_t input_dimension,
  size_t input_padding_dimension,
  size_t kernel_dimension);

XNN_INTERNAL uint32_t xnn_get_heuristic_mr_gemm(
  size_t batch_size,
  uint32_t max_mr,
  uint32_t nr,
  struct xnn_hmp_gemm_ukernel *gemm_cases);

XNN_INTERNAL uint32_t xnn_get_heuristic_mr_igemm(
  size_t batch_size,
  uint32_t max_mr,
  uint32_t nr,
  struct xnn_hmp_igemm_ukernel *igemm_cases);

#ifdef __cplusplus
}
#endif
