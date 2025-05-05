// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "bench/subgraph/models.h"
#include "include/xnnpack.h"

namespace models {

xnn_subgraph_t FP32Softmax(size_t m, size_t n, size_t k, uint32_t norm_mask,
                           bool use_softmax) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims = {{k, m, n}};
  std::array<size_t, 3> reduction_dims = dims;
  std::vector<size_t> reduction_axes;
  reduction_axes.reserve(reduction_dims.size());
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_dims[i] = 1;
      reduction_axes.push_back(i);
    }
  }

  uint32_t v0 = 0;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, v0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = 1;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, v1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  if (use_softmax) {
    // Create a permutation that pushes the reduction dimensions to the
    // inside dimensions.
    std::vector<size_t> perm;
    std::vector<size_t> inv_perm(dims.size());
    std::vector<size_t> perm_dims(dims.size());
    std::vector<size_t> reshaped_dims;
    for (size_t k = 0; k < dims.size(); k++) {
      if ((norm_mask & (1 << k)) == 0) {
        inv_perm[k] = perm.size();
        perm.push_back(k);
        reshaped_dims.push_back(dims[k]);
      }
    }
    for (size_t k = 0; k < dims.size(); k++) {
      if ((norm_mask & (1 << k)) != 0) {
        inv_perm[k] = perm.size();
        perm.push_back(k);
      }
      perm_dims[k] = dims[perm[k]];
    }
    reshaped_dims.push_back(0);

    // Transpose the reduction dimensions to the innermost dimensions (if
    // needed).
    uint32_t transposed_v0 = v0;
    uint32_t transposed_v1 = v1;
    const bool needs_transpose = !std::is_sorted(perm.begin(), perm.end());
    if (needs_transpose) {
      status = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
          /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &transposed_v0);
      if (status != xnn_status_success) {
        std::cerr << "failed to create tensor transposed_v0" << std::endl;
        return nullptr;
      }

      status = xnn_define_static_transpose(subgraph, perm.size(), perm.data(),
                                           v0, transposed_v0,
                                           /*flags=*/0);
      if (status != xnn_status_success) {
        std::cerr << "failed to create transpose node" << std::endl;
        return nullptr;
      }

      status = xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, reduction_dims.size(),
          reduction_dims.data(),
          /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &transposed_v1);
      if (status != xnn_status_success) {
        std::cerr << "failed to create tensor transposed_v1" << std::endl;
        return nullptr;
      }
    }

    // Reshape to group the reduction dimensions (if needed).
    uint32_t transposed_reshaped_v0 = transposed_v0;
    uint32_t transposed_reshaped_v1 = transposed_v1;
    if (reshaped_dims.size() < dims.size()) {
      status =
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                  dims.size(), dims.data(),
                                  /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &transposed_reshaped_v0);
      if (status != xnn_status_success) {
        std::cerr << "failed to create tensor transposed_reshaped_v0"
                  << std::endl;
        return nullptr;
      }

      status = xnn_define_static_reshape(subgraph, reshaped_dims.size(),
                                         reshaped_dims.data(), transposed_v0,
                                         transposed_reshaped_v0,
                                         /*flags=*/0);
      if (status != xnn_status_success) {
        std::cerr << "failed to create reshape node" << std::endl;
        return nullptr;
      }

      status =
          xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                  dims.size(), dims.data(),
                                  /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &transposed_reshaped_v1);
      if (status != xnn_status_success) {
        std::cerr << "failed to create tensor transposed_reshaped_v1"
                  << std::endl;
        return nullptr;
      }
    }

    // Compute the softmax.
    status = xnn_define_softmax(subgraph, transposed_reshaped_v0,
                                transposed_reshaped_v1,
                                /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create softmax node" << std::endl;
      return nullptr;
    }

    // Reshape back to the original rank.
    if (reshaped_dims.size() < dims.size()) {
      status = xnn_define_static_reshape(subgraph, perm_dims.size(),
                                         perm_dims.data(),
                                         transposed_reshaped_v1, transposed_v1,
                                         /*flags=*/0);
      if (status != xnn_status_success) {
        std::cerr << "failed to create reshape node" << std::endl;
        return nullptr;
      }
    }

    // Transpose back to the original shape.
    if (needs_transpose) {
      status = xnn_define_static_transpose(subgraph, inv_perm.size(),
                                           inv_perm.data(), transposed_v1, v1,
                                           /*flags=*/0);
      if (status != xnn_status_success) {
        std::cerr << "failed to create inverse transpose node" << std::endl;
        return nullptr;
      }
    }

  } else {
    uint32_t max_v0 = XNN_INVALID_VALUE_ID;
    status =
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                reduction_dims.size(), reduction_dims.data(),
                                /*data=*/nullptr, max_v0, /*flags=*/0, &max_v0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor max_v0" << std::endl;
      return nullptr;
    }

    status = xnn_define_static_reduce(
        subgraph, xnn_reduce_max, reduction_axes.size(), reduction_axes.data(),
        v0, max_v0, /*flags=*/XNN_FLAG_KEEP_DIMS);
    if (status != xnn_status_success) {
      std::cerr << "failed to create reduce max" << std::endl;
      return nullptr;
    }

    uint32_t v0_minus_max = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
        /*data=*/nullptr, v0_minus_max, /*flags=*/0, &v0_minus_max);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor v0_minus_max" << std::endl;
      return nullptr;
    }

    status = xnn_define_binary(subgraph, xnn_binary_subtract, nullptr, v0,
                               max_v0, v0_minus_max, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create binary subtract" << std::endl;
      return nullptr;
    }

    uint32_t exp_v0_minus_max = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
        /*data=*/nullptr, exp_v0_minus_max, /*flags=*/0, &exp_v0_minus_max);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor exp_v0_minus_max" << std::endl;
      return nullptr;
    }

    status = xnn_define_unary(subgraph, xnn_unary_exp, nullptr, v0_minus_max,
                              exp_v0_minus_max, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary exp" << std::endl;
      return nullptr;
    }

    uint32_t sum_exp_v0_minus_max = XNN_INVALID_VALUE_ID;
    status =
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                reduction_dims.size(), reduction_dims.data(),
                                /*data=*/nullptr, sum_exp_v0_minus_max,
                                /*flags=*/0, &sum_exp_v0_minus_max);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor sum_exp_v0_minus_max" << std::endl;
      return nullptr;
    }

    status = xnn_define_static_reduce(
        subgraph, xnn_reduce_sum, reduction_axes.size(), reduction_axes.data(),
        exp_v0_minus_max, sum_exp_v0_minus_max,
        /*flags=*/XNN_FLAG_KEEP_DIMS);
    if (status != xnn_status_success) {
      std::cerr << "failed to create reduce sum" << std::endl;
      return nullptr;
    }

    uint32_t inv_sum = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, reduction_dims.size(),
        reduction_dims.data(),
        /*data=*/nullptr, inv_sum, /*flags=*/0, &inv_sum);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor inv_sum" << std::endl;
      return nullptr;
    }

    static const float one_value = 1.0f;
    uint32_t one = XNN_INVALID_VALUE_ID;
    status =
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 0, nullptr,
                                /*data=*/&one_value, one, /*flags=*/0, &one);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor one" << std::endl;
      return nullptr;
    }

    status = xnn_define_binary(subgraph, xnn_binary_divide, nullptr, one,
                               sum_exp_v0_minus_max, inv_sum, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary reciprocal" << std::endl;
      return nullptr;
    }

    status = xnn_define_binary(subgraph, xnn_binary_multiply, nullptr,
                               exp_v0_minus_max, inv_sum, v1, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create binary multiply" << std::endl;
      return nullptr;
    }
  }

  return subgraph;
}

}  // namespace models
