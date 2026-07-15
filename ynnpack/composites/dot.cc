// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

}  // namespace ynn
