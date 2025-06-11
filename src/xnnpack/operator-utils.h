// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_UTILS_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_UTILS_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/operator.h"

static inline bool use_weights_cache(struct xnn_operator* op) {
  return op->weights_cache != NULL;
}

static inline void* packed_weights(struct xnn_operator* op) {
  if (use_weights_cache(op)) {
    return op->weights_cache->offset_to_addr(op->weights_cache->context,
                                             op->packed_weights.offset);
  } else {
    return op->packed_weights.pointer;
  }
}

// Get a pointer to a region to pack weights into. If weights cache is
// available, use it, returning to a pointer to the cache's buffer, otherwise,
// allocate and return a pointer to a new region. Returns NULL on error.
XNN_INTERNAL void* xnn_get_pointer_to_write_weights(
    xnn_operator_t op, size_t aligned_weights_size);

#ifdef __cplusplus
extern "C" {
#endif
XNN_INTERNAL size_t xnn_compute_convolution_output_dimension(
    size_t padded_input_dimension, size_t kernel_dimension,
    size_t dilation_dimension, size_t subsampling_dimension);

XNN_INTERNAL size_t xnn_compute_deconvolution_output_dimension(
    size_t input_dimension, size_t output_padding_dimension,
    size_t adjustment_dimension, size_t kernel_dimension,
    size_t dilation_dimension, size_t stride_dimension);

XNN_INTERNAL size_t xnn_compute_unpooling_output_dimension(
    size_t input_dimension, size_t input_padding_dimension,
    size_t kernel_dimension);

XNN_INTERNAL uint32_t
xnn_get_heuristic_mr_gemm(size_t batch_size, uint32_t max_mr, uint32_t nr,
                          struct xnn_hmp_gemm_ukernel* gemm_cases);

XNN_INTERNAL uint32_t
xnn_get_heuristic_mr_igemm(size_t batch_size, uint32_t max_mr, uint32_t nr,
                           struct xnn_hmp_igemm_ukernel* igemm_cases);

XNN_INTERNAL enum xnn_status xnn_destroy_operator(xnn_operator_t op);

XNN_INTERNAL const char* xnn_unary_operator_to_string(
    enum xnn_unary_operator op);
XNN_INTERNAL const char* xnn_binary_operator_to_string(
    enum xnn_binary_operator op);

XNN_INTERNAL const char* xnn_operator_type_to_string_v2(xnn_operator_t op);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_UTILS_H_