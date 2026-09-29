// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_convert_f32_to_bf16_sum(ynn_subgraph_t subgraph,
                                          uint32_t input_id, size_t num_values,
                                          uint32_t* output_ids,
                                          uint32_t flags) {
  if (num_values == 0) return ynn_status_success;

  uint32_t node_flags = flags | YNN_NODE_FLAG_NO_EXCESS_PRECISION;
  uint32_t current_residual_fp32 = input_id;

  for (size_t i = 0; i < num_values; ++i) {
    if (output_ids[i] == YNN_INVALID_VALUE_ID) {
      YNN_RETURN_IF_ERROR(ynn_define_convert(subgraph, current_residual_fp32,
                                             ynn_type_bf16, &output_ids[i],
                                             node_flags));
    } else {
      YNN_RETURN_IF_ERROR(ynn_define_convert(subgraph, current_residual_fp32,
                                             ynn_type_bf16, &output_ids[i],
                                             node_flags));
    }

    if (i == num_values - 1) {
      break;
    }

    if (i == num_values - 2) {
      // Optimize the last residual computation by directly outputting bf16.
      if (output_ids[i + 1] == YNN_INVALID_VALUE_ID) {
        YNN_RETURN_IF_ERROR(ynn_define_tensor(
            subgraph, ynn_type_bf16, 0, nullptr, nullptr,
            YNN_VALUE_FLAG_NO_EXCESS_PRECISION, &output_ids[i + 1]));
      }
      YNN_RETURN_IF_ERROR(ynn_define_binary(
          subgraph, ynn_binary_subtract, current_residual_fp32, output_ids[i],
          &output_ids[i + 1], node_flags));
      break;
    }

    // General case: compute fp32 residual for next steps.
    uint32_t next_residual_fp32 = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                          current_residual_fp32, output_ids[i],
                                          &next_residual_fp32, node_flags));

    current_residual_fp32 = next_residual_fp32;
  }

  return ynn_status_success;
}

ynn_status define_dot_sum(ynn_subgraph_t subgraph, size_t num_k_dims,
                          size_t num_a_values, const uint32_t* a_ids,
                          size_t num_b_values, const uint32_t* b_ids,
                          uint32_t input_c_id, uint32_t& output_id,
                          uint32_t flags, int max_sum_index) {
  if (num_a_values == 0 || num_b_values == 0) {
    return ynn_status_invalid_parameter;
  }

  if (num_a_values == 1 && num_b_values == 1) {
    return ynn_define_dot(subgraph, num_k_dims, a_ids[0], b_ids[0], input_c_id,
                          &output_id, flags);
  }

  struct dot_sum_term {
    size_t i;
    size_t j;
    int sum_index;
    bool operator<(const dot_sum_term& other) const {
      return std::make_tuple(-sum_index, i, j) <
             std::make_tuple(-other.sum_index, other.i, other.j);
    }
  };
  std::vector<dot_sum_term> residuals;
  for (size_t i = 0; i < num_a_values; ++i) {
    for (size_t j = 0; j < num_b_values; ++j) {
      if (i == 0 && j == 0) continue;
      int sum_index = static_cast<int>(i + j);
      if (max_sum_index >= 0 && sum_index >= max_sum_index) continue;
      residuals.push_back({i, j, sum_index});
    }
  }

  if (residuals.empty()) {
    return ynn_define_dot(subgraph, num_k_dims, a_ids[0], b_ids[0], input_c_id,
                          &output_id, flags);
  }

  std::sort(residuals.begin(), residuals.end());

  uint32_t current_tail_id = YNN_INVALID_VALUE_ID;
  for (const auto& pair : residuals) {
    uint32_t next_tail_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, num_k_dims, a_ids[pair.i],
                                       b_ids[pair.j], current_tail_id,
                                       &next_tail_id, flags));
    current_tail_id = next_tail_id;
  }

  // Now compute the main dot: a_ids[0] * b_ids[0] + input_c_id
  uint32_t main_dot_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, num_k_dims, a_ids[0], b_ids[0],
                                     input_c_id, &main_dot_id, flags));

  // Finally add them.
  return ynn_define_binary(subgraph, ynn_binary_add, main_dot_id,
                           current_tail_id, &output_id, flags);
}

namespace {

ynn_status define_blockwise_scale(ynn_subgraph_t subgraph, uint32_t scale_id,
                                  uint32_t& output_scale_id,
                                  bool expand_m_dim) {
  if (scale_id == YNN_INVALID_VALUE_ID) {
    return ynn_status_invalid_parameter;
  }

  // 1. Transpose the last two dimensions: [..., N, num_blocks] -> [...,
  // num_blocks, N].
  int32_t perm[2] = {-1, -2};
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(
      subgraph, 2, perm, scale_id, &transposed_id, YNN_NODE_FLAG_KEEP_DIMS));

  uint32_t result_id = transposed_id;
  if (expand_m_dim) {
    // 2. Expand dimension before the last dimension: [..., num_blocks, N] ->
    // [..., num_blocks, 1, N].
    int32_t expand_axis[1] = {-2};
    uint32_t expanded_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_static_expand_dims(
        subgraph, 1, expand_axis, transposed_id, &expanded_id, 0));
    result_id = expanded_id;
  }

  output_scale_id = result_id;
  return ynn_status_success;
}

}  // namespace

ynn_status define_blockwise_dot(ynn_subgraph_t subgraph, uint32_t a_id,
                                uint32_t a_zero_point_id, uint32_t a_scale_id,
                                uint32_t b_id, uint32_t b_zero_point_id,
                                uint32_t b_scale_id, size_t block_size,
                                uint32_t bias_id, ynn_type output_type,
                                uint32_t& output_id, uint32_t flags) {
  if (block_size == 0) {
    return ynn_status_invalid_parameter;
  }

  // Split A's reduction dimension K into [num_blocks, block_size].
  size_t split_dims[2] = {0, block_size};
  uint32_t a_split_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_split_dim(subgraph, -1, 2, split_dims, a_id, &a_split_id, 0));

  // Transpose M and num_blocks axes:
  // [..., M, num_blocks, block_size] -> [..., num_blocks, M, block_size].
  int32_t perm_a[2] = {-2, -3};
  uint32_t transposed_a_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(subgraph, 2, perm_a,
                                                  a_split_id, &transposed_a_id,
                                                  YNN_NODE_FLAG_KEEP_DIMS));
  uint32_t a_inner_id = transposed_a_id;

  // Split B's second-to-last axis (K) into [num_blocks, block_size].
  // B: [..., K, N] -> [..., num_blocks, block_size, N].
  uint32_t b_inner_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_split_dim(subgraph, -2, 2, split_dims, b_id, &b_inner_id, 0));

  // Align B's scale.
  uint32_t b_scale_aligned_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_blockwise_scale(
      subgraph, b_scale_id, b_scale_aligned_id, /*expand_m_dim=*/true));

  // Align B's zero point (if present).
  uint32_t b_zp_aligned_id = YNN_INVALID_VALUE_ID;
  if (b_zero_point_id != YNN_INVALID_VALUE_ID) {
    YNN_RETURN_IF_ERROR(define_blockwise_scale(
        subgraph, b_zero_point_id, b_zp_aligned_id, /*expand_m_dim=*/true));
  }

  // B's zero point contribution per block is -b_zp * sum_k(a), which varies at
  // runtime and is initialized in each block's accumulator.
  uint32_t accum_init_id = YNN_INVALID_VALUE_ID;
  if (b_zero_point_id != YNN_INVALID_VALUE_ID) {
    int32_t a_k_axis = -1;
    uint32_t sum_a_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_reduce(
        subgraph, ynn_reduce_sum, 1, &a_k_axis, a_inner_id,
        YNN_INVALID_VALUE_ID, &sum_a_id, YNN_NODE_FLAG_KEEP_DIMS));
    uint32_t b_zp_term_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          b_zp_aligned_id, sum_a_id,
                                          &b_zp_term_id, 0));
    YNN_RETURN_IF_ERROR(ynn_define_unary(subgraph, ynn_unary_negate,
                                         b_zp_term_id, &accum_init_id, 0));
  }

  // Run inner dot: produces [..., num_blocks, M, N].
  uint32_t dot_blocks_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, a_inner_id,
                                     b_inner_id, accum_init_id, &dot_blocks_id,
                                     flags));

  // Multiply by b_scale.
  uint32_t scaled_blocks_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                        dot_blocks_id, b_scale_aligned_id,
                                        &scaled_blocks_id, 0));

  // Reduce sum along num_blocks axis.
  int32_t reduce_axis = -3;
  uint32_t reduced_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 1,
                                        &reduce_axis, scaled_blocks_id,
                                        YNN_INVALID_VALUE_ID, &reduced_id, 0));

  // A's zero point contribution is
  // sum_b scale[n,b] * (zp[m] * colsum_(b - b_zp)[n]), which factors into
  // zp[m] * C[n] with C = sum_b scale[n,b] * colsum_(b - b_zp)[n] computed
  // entirely from static tensors (so it constant-folds), and can be subtracted
  // once from the final result instead of initializing every block's
  // accumulator with it.
  if (a_zero_point_id != YNN_INVALID_VALUE_ID) {
    uint32_t b_eff_id = b_inner_id;
    if (b_zero_point_id != YNN_INVALID_VALUE_ID) {
      b_eff_id = YNN_INVALID_VALUE_ID;
      YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                            b_inner_id, b_zp_aligned_id,
                                            &b_eff_id, 0));
    }

    int32_t colsum_axis = -2;
    uint32_t colsum_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_reduce(
        subgraph, ynn_reduce_sum, 1, &colsum_axis, b_eff_id,
        YNN_INVALID_VALUE_ID, &colsum_id, YNN_NODE_FLAG_KEEP_DIMS));

    uint32_t scaled_colsum_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          colsum_id, b_scale_aligned_id,
                                          &scaled_colsum_id, 0));
    uint32_t c_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_reduce(subgraph, ynn_reduce_sum, 1,
                                          &reduce_axis, scaled_colsum_id,
                                          YNN_INVALID_VALUE_ID, &c_id, 0));

    // reduced -= a_zp * C, applied once to the final result.
    uint32_t zp_correction_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          a_zero_point_id, c_id,
                                          &zp_correction_id, 0));
    uint32_t corrected_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_subtract,
                                          reduced_id, zp_correction_id,
                                          &corrected_id, 0));
    reduced_id = corrected_id;
  }

  // Multiply by a_scale if present.
  if (a_scale_id != YNN_INVALID_VALUE_ID) {
    uint32_t scaled_out_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_multiply,
                                          reduced_id, a_scale_id,
                                          &scaled_out_id, 0));
    reduced_id = scaled_out_id;
  }

  // Add bias if present.
  if (bias_id != YNN_INVALID_VALUE_ID) {
    uint32_t biased_out_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_binary(subgraph, ynn_binary_add, reduced_id,
                                          bias_id, &biased_out_id, 0));
    reduced_id = biased_out_id;
  }

  // Convert/assign to output_id.
  if (output_type != ynn_type_invalid && output_type != ynn_type_fp32) {
    return ynn_define_convert(subgraph, reduced_id, output_type, &output_id, 0);
  } else if (output_id != YNN_INVALID_VALUE_ID && output_id != reduced_id) {
    return ynn_define_convert(subgraph, reduced_id, ynn_type_fp32, &output_id,
                              0);
  } else {
    output_id = reduced_id;
    return ynn_status_success;
  }
}

}  // namespace ynn
