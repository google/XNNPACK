// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"  // For xnn_operator_t.
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"  // For XNN_ALLOCATION_ALIGNMENT.
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"  // For xnn_operator definition.
#include "xnnpack/operator-type.h"

void* xnn_get_pointer_to_write_weights(
  xnn_operator_t op,
  size_t aligned_weights_size,
  int padding_byte)
{
  assert(aligned_weights_size % XNN_ALLOCATION_ALIGNMENT == 0);
  void* weights_ptr = NULL;
  if (use_weights_cache(op)) {
    weights_ptr = op->weights_cache->reserve_space(op->weights_cache->context, aligned_weights_size);
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

size_t xnn_compute_convolution_output_dimension(
  size_t padded_input_dimension,
  size_t kernel_dimension,
  size_t dilation_dimension,
  size_t subsampling_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(padded_input_dimension, effective_kernel_dimension) / subsampling_dimension + 1;
}

size_t xnn_compute_deconvolution_output_dimension(
  size_t input_dimension,
  size_t output_padding_dimension,
  size_t adjustment_dimension,
  size_t kernel_dimension,
  size_t dilation_dimension,
  size_t stride_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(
    stride_dimension * (input_dimension - 1) + adjustment_dimension + effective_kernel_dimension,
    output_padding_dimension);
}

size_t xnn_compute_unpooling_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t kernel_dimension)
{
  return xnn_compute_deconvolution_output_dimension(
      input_dimension, input_padding_dimension, /*adjustment_dimension=*/0,
      kernel_dimension, /*dilation_dimension=*/1, /*stride_dimension=*/kernel_dimension);
}

// Calculate how much work a microkernel does.
// A MxN microkernel does M+N (scalar) loads and M*N (scalar) FMAs.
// So, given batch_size, the microkernel does:
//   divide_round_up(batch_size, mr) * (mr + nr) loads, and
//   divide_round_up(batch_size, mr) * (mr * nr) FMAs.
// The total cost is then a linear combination of these 2 operations. From experimental data, use a multiplier of 3 for
// loads, to prefer higher tile sizes which have better computation intensity.
static size_t calculate_microkernel_cost(size_t batch_size, uint32_t mr, uint32_t nr)
{
  return divide_round_up(batch_size, mr) * (3 * (mr + nr) + mr * nr);
}

static bool mr_is_available_gemm(size_t mr, struct xnn_hmp_gemm_ukernel *gemm_cases)
{
  return gemm_cases[mr-1].function[XNN_UARCH_DEFAULT] != NULL;
}

uint32_t xnn_get_heuristic_mr_gemm(
  size_t batch_size, uint32_t max_mr, uint32_t nr, struct xnn_hmp_gemm_ukernel *gemm_cases)
{
  if (batch_size <= max_mr && mr_is_available_gemm(batch_size, gemm_cases)) {
    // We have a microkernel with MR that is the exact match with batch_size.
    return batch_size;
  }

  // Try to find the best fitting mr.
  // - use a cost heuristic to calculate how much work is done by the microkernel (see calculate_microkernel_cost)
  // - smaller cost is better
  uint32_t best_mr = max_mr;
  size_t best_cost = SIZE_MAX;
  for (uint32_t mr = 1; mr <= max_mr; mr++) {
    if (!mr_is_available_gemm(mr, gemm_cases)){
      continue;
    }
    const size_t current_cost = calculate_microkernel_cost(batch_size, mr, nr);
    if (current_cost <= best_cost) {
      best_mr = mr;
      best_cost = current_cost;
    }
  }
  return best_mr;
}

static bool mr_is_available_igemm(size_t mr, struct xnn_hmp_igemm_ukernel *igemm_cases)
{
  return igemm_cases[mr-1].function[XNN_UARCH_DEFAULT] != NULL;
}

uint32_t xnn_get_heuristic_mr_igemm(
  size_t batch_size, uint32_t max_mr, uint32_t nr, struct xnn_hmp_igemm_ukernel *igemm_cases)
{
  if (batch_size <= max_mr && mr_is_available_igemm(batch_size, igemm_cases)) {
    // We have a microkernel with MR that is the exact match with batch_size.
    return batch_size;
  }

  // Try to find the best fitting mr.
  // - use a cost heuristic to calculate how much work is done by the microkernel (see calculate_microkernel_cost)
  // - smaller cost is better
  uint32_t best_mr = max_mr;
  size_t best_cost = SIZE_MAX;
  for (uint32_t mr = 1; mr <= max_mr; mr++) {
    if (!mr_is_available_igemm(mr, igemm_cases)){
      continue;
    }
    const size_t current_cost = calculate_microkernel_cost(batch_size, mr, nr);
    if (current_cost <= best_cost) {
      best_mr = mr;
      best_cost = current_cost;
    }
  }
  return best_mr;
}

enum xnn_status xnn_destroy_operator(xnn_operator_t op)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to delete operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (op == NULL) {
    return xnn_status_invalid_parameter;
  }

  xnn_release_memory(op->indirection_buffer);
  if (op->weights_cache == NULL) {
    xnn_release_simd_memory(op->packed_weights.pointer);
  }
  xnn_release_simd_memory(op->zero_buffer);
  if (op->zero_buffers) {
    for (size_t i = 1; i < op->batch_size; ++i) {
      xnn_release_simd_memory(op->zero_buffers[i]);
    }
    xnn_release_memory(op->zero_buffers);
  }
  xnn_release_memory(op->pixelwise_buffer);
  xnn_release_memory(op->subconvolution_buffer);
  xnn_release_simd_memory(op->lookup_table);
  return xnn_status_success;
}


const char* xnn_unary_operator_to_string(enum xnn_unary_operator op)
{
  switch (op) {
    case xnn_unary_abs:
      return "abs";
    case xnn_unary_bankers_rounding:
      return "bankers_rounding";
    case xnn_unary_ceiling:
      return "ceiling";
    case xnn_unary_clamp:
      return "clamp";
    case xnn_unary_convert:
      return "convert";
    case xnn_unary_elu:
      return "elu";
    case xnn_unary_exp:
      return "exp";
    case xnn_unary_floor:
      return "floor";
    case xnn_unary_gelu:
      return "gelu";
    case xnn_unary_hardswish:
      return "hardswish";
    case xnn_unary_leaky_relu:
      return "leaky_relu";
    case xnn_unary_log:
      return "log";
    case xnn_unary_negate:
      return "negate";
    case xnn_unary_reciprocal_square_root:
      return "reciprocal_square_root";
    case xnn_unary_sigmoid:
      return "sigmoid";
    case xnn_unary_square:
      return "square";
    case xnn_unary_square_root:
      return "square_root";
    case xnn_unary_tanh:
      return "tanh";
    case xnn_unary_invalid:
      return "invalid";
  }
  XNN_UNREACHABLE;
  return "unknown";
}

const char* xnn_binary_operator_to_string(enum xnn_binary_operator op)
{
  switch (op) {
    case xnn_binary_add:
      return "add";
    case xnn_binary_divide:
      return "divide";
    case xnn_binary_multiply:
      return "multiply";
    case xnn_binary_subtract:
      return "subtract";
    case xnn_binary_copysign:
      return "copysign";
    case xnn_binary_squared_difference:
      return "squared_difference";
    case xnn_binary_prelu:
      return "prelu";
    case xnn_binary_minimum:
      return "minimum";
    case xnn_binary_maximum:
      return "maximum";
    case xnn_binary_invalid:
      return "invalid";
  }
  XNN_UNREACHABLE;
  return "unknown";
}

enum xnn_operator_type xnn_reduce_operator_to_operator_type(enum xnn_reduce_operator type)
{
  switch (type) {
    case xnn_reduce_mean:
      return xnn_operator_type_mean_nd;
    case xnn_reduce_sum:
      return xnn_operator_type_sum_nd;
    default:
      return xnn_operator_type_invalid;
  }
}
