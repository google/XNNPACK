// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

ynn_status broadcast_inputs(ynn_subgraph_t subgraph, uint32_t& input_a_id,
                            uint32_t& input_b_id) {
  if (input_a_id == YNN_INVALID_VALUE_ID ||
      input_b_id == YNN_INVALID_VALUE_ID) {
    return ynn_status_success;
  }

  const int32_t all_axes[] = {-1, -2, -3, -4, -5, -6, -7, -8};

  uint32_t input_a_broadcasted_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_broadcast_like(
      subgraph, YNN_MAX_TENSOR_RANK, all_axes, input_a_id, input_b_id,
      &input_a_broadcasted_id, 0));

  uint32_t input_b_broadcasted_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_broadcast_like(
      subgraph, YNN_MAX_TENSOR_RANK, all_axes, input_b_id, input_a_id,
      &input_b_broadcasted_id, 0));

  input_a_id = input_a_broadcasted_id;
  input_b_id = input_b_broadcasted_id;
  return ynn_status_success;
}

}  // namespace

ynn_status define_dot_quantization(ynn_subgraph_t subgraph, size_t num_k_dims,
                                   uint32_t a_id, uint32_t a_zero_point_id,
                                   uint32_t a_scale_id, uint32_t b_id,
                                   uint32_t b_zero_point_id,
                                   uint32_t b_scale_id, uint32_t& zero_point_id,
                                   uint32_t& scale_id) {
  // 1. Compute scale_id = a_scale * b_scale
  scale_id = YNN_INVALID_VALUE_ID;
  if (a_scale_id != YNN_INVALID_VALUE_ID &&
      b_scale_id != YNN_INVALID_VALUE_ID) {
    uint32_t a_scale_broadcasted = a_scale_id;
    uint32_t b_scale_broadcasted = b_scale_id;
    YNN_RETURN_IF_ERROR(
        broadcast_inputs(subgraph, a_scale_broadcasted, b_scale_broadcasted));
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          a_scale_broadcasted,
                                          b_scale_broadcasted, &scale_id, 0));
  } else if (a_scale_id != YNN_INVALID_VALUE_ID) {
    scale_id = a_scale_id;
  } else if (b_scale_id != YNN_INVALID_VALUE_ID) {
    scale_id = b_scale_id;
  }

  // 2. Compute zero_point_id = a_zp * sum(b) + b_zp * sum(a) - a_zp * b_zp * k
  if (a_zero_point_id == YNN_INVALID_VALUE_ID &&
      b_zero_point_id == YNN_INVALID_VALUE_ID) {
    zero_point_id = YNN_INVALID_VALUE_ID;
    return ynn_status_success;
  }

  int32_t a_k_dims[YNN_MAX_TENSOR_RANK];
  int32_t b_k_dims[YNN_MAX_TENSOR_RANK];
  std::iota(a_k_dims, a_k_dims + num_k_dims, -static_cast<int>(num_k_dims));
  std::iota(b_k_dims, b_k_dims + num_k_dims, -static_cast<int>(num_k_dims) - 1);
  std::reverse(a_k_dims, a_k_dims + num_k_dims);
  std::reverse(b_k_dims, b_k_dims + num_k_dims);

  // Term 1: a_zp * sum(b)
  uint32_t term_a_sum_b = YNN_INVALID_VALUE_ID;
  if (a_zero_point_id != YNN_INVALID_VALUE_ID) {
    uint32_t sum_b = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, num_k_dims,
                                          b_k_dims, b_id, YNN_INVALID_VALUE_ID,
                                          &sum_b, 0));
    uint32_t sum_b_expanded = YNN_INVALID_VALUE_ID;
    if (num_k_dims > 0) {
      YNN_RETURN_IF_ERROR(ynn_define_static_expand_dims(
          subgraph, 1, &b_k_dims[0], sum_b, &sum_b_expanded, 0));
    } else {
      sum_b_expanded = sum_b;
    }
    uint32_t a_zp_broadcasted = a_zero_point_id;
    uint32_t sum_b_broadcasted = sum_b_expanded;
    YNN_RETURN_IF_ERROR(
        broadcast_inputs(subgraph, a_zp_broadcasted, sum_b_broadcasted));
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          a_zp_broadcasted, sum_b_broadcasted,
                                          &term_a_sum_b, 0));
  }

  // Term 2: b_zp * sum(a)
  uint32_t term_b_sum_a = YNN_INVALID_VALUE_ID;
  if (b_zero_point_id != YNN_INVALID_VALUE_ID) {
    uint32_t sum_a = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, num_k_dims,
                                          a_k_dims, a_id, YNN_INVALID_VALUE_ID,
                                          &sum_a, 0));
    uint32_t sum_a_expanded = YNN_INVALID_VALUE_ID;
    if (num_k_dims > 0) {
      YNN_RETURN_IF_ERROR(ynn_define_static_expand_dims(
          subgraph, 1, &a_k_dims[0], sum_a, &sum_a_expanded, 0));
    } else {
      sum_a_expanded = sum_a;
    }
    uint32_t b_zp_broadcasted = b_zero_point_id;
    uint32_t sum_a_broadcasted = sum_a_expanded;
    YNN_RETURN_IF_ERROR(
        broadcast_inputs(subgraph, b_zp_broadcasted, sum_a_broadcasted));
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          b_zp_broadcasted, sum_a_broadcasted,
                                          &term_b_sum_a, 0));
  }

  // Term 3: a_zp * b_zp * k
  uint32_t term_ab_zp = YNN_INVALID_VALUE_ID;
  if (a_zero_point_id != YNN_INVALID_VALUE_ID &&
      b_zero_point_id != YNN_INVALID_VALUE_ID) {
    uint32_t a_zp_broadcasted = a_zero_point_id;
    uint32_t b_zp_broadcasted = b_zero_point_id;
    YNN_RETURN_IF_ERROR(
        broadcast_inputs(subgraph, a_zp_broadcasted, b_zp_broadcasted));
    uint32_t zp_prod = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          a_zp_broadcasted, b_zp_broadcasted,
                                          &zp_prod, 0));
    uint32_t k_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_get_tensor_shape(
        subgraph, num_k_dims, b_k_dims, ynn_type_int32,
        /*rank=*/0, b_id, &k_id,
        /*flags=*/YNN_NODE_FLAG_RESHAPE_1D | YNN_NODE_FLAG_UNIQUE_DIMS));
    uint32_t zp_prod_broadcasted = zp_prod;
    uint32_t k_broadcasted = k_id;
    YNN_RETURN_IF_ERROR(
        broadcast_inputs(subgraph, zp_prod_broadcasted, k_broadcasted));
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          zp_prod_broadcasted, k_broadcasted,
                                          &term_ab_zp, 0));
  }

  // Combine terms: zero_point_id = term_a_sum_b + term_b_sum_a - term_ab_zp
  zero_point_id = YNN_INVALID_VALUE_ID;
  if (term_a_sum_b != YNN_INVALID_VALUE_ID &&
      term_b_sum_a != YNN_INVALID_VALUE_ID) {
    uint32_t sum_terms = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(
        subgraph, ynn_binary_add, term_a_sum_b, term_b_sum_a, &sum_terms, 0));
    if (term_ab_zp != YNN_INVALID_VALUE_ID) {
      YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                            sum_terms, term_ab_zp,
                                            &zero_point_id, 0));
    } else {
      zero_point_id = sum_terms;
    }
  } else if (term_a_sum_b != YNN_INVALID_VALUE_ID) {
    zero_point_id = term_a_sum_b;
  } else if (term_b_sum_a != YNN_INVALID_VALUE_ID) {
    zero_point_id = term_b_sum_a;
  }

  return ynn_status_success;
}

}  // namespace ynn
