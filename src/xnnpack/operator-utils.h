// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>

#if XNN_PLATFORM_JIT
// Generates code for all mr values up to max_mr.
// Offsets of all generated code will be kept in generated_code_offset.
XNN_INTERNAL void xnn_generate_gemms_up_to_max_mr(
    size_t max_mr,
    struct gemm_codegens generators,
    const struct jit_gemm_params *jit_gemm_params,
    size_t group_output_channels,
    size_t nr,
    size_t group_input_channels_in_bytes,
    xnn_operator_t convolution_op);
XNN_INTERNAL void xnn_generate_igemms_up_to_max_mr(
  size_t max_mr,
  struct gemm_codegens generators,
  const struct jit_gemm_params *jit_gemm_params,
  size_t group_output_channels,
  size_t nr,
  size_t group_input_channels_in_bytes,
  size_t kernel_size,
  xnn_operator_t convolution_op);

// Overwrite function pointer to GEMM microkernels with generated code if available.
XNN_INTERNAL void xnn_overwrite_gemm_cases_with_generated_code(
  xnn_operator_t convolution_op,
  struct xnn_hmp_gemm_ukernel *gemm_cases,
  size_t mr);
// Overwrite function pointer to IGEMM microkernels with generated code if available.
XNN_INTERNAL void xnn_overwrite_igemm_cases_with_generated_code(
  xnn_operator_t convolution_op,
  struct xnn_hmp_igemm_ukernel *igemm_cases,
  size_t mr);
#endif  // XNN_PLATFORM_JIT

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
  struct xnn_hmp_gemm_ukernel *gemm_cases,
  bool code_cache_available);

XNN_INTERNAL uint32_t xnn_get_heuristic_mr_igemm(
  size_t batch_size,
  uint32_t max_mr,
  uint32_t nr,
  struct xnn_hmp_igemm_ukernel *igemm_cases,
  bool code_cache_available);

#ifdef __cplusplus
}
#endif
