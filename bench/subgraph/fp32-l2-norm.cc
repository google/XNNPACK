// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "include/xnnpack.h"

namespace models {

xnn_subgraph_t FP32L2Norm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims = {{m, n, k}};
  std::array<size_t, 3> reduction_dims = dims;
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_dims[i] = 1;
    }
  }

  uint32_t input = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  uint32_t input_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                                   dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &input_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input_sq" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_square, nullptr, input,
                            input_sq, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary square" << std::endl;
    return nullptr;
  }

  uint32_t sum_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, reduction_dims.size(), reduction_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &sum_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor sum_sq" << std::endl;
    return nullptr;
  }

  std::vector<size_t> reduction_axes;
  reduction_axes.reserve(reduction_dims.size());
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_axes.push_back(i);
    }
  }
  status =
      xnn_define_static_reduce(subgraph, xnn_reduce_sum, reduction_axes.size(),
                               reduction_axes.data(), input_sq, sum_sq,
                               /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce sum" << std::endl;
    return nullptr;
  }

  uint32_t inv_sqrt_sum_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &inv_sqrt_sum_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor inv_sqrt_sum_sq" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_reciprocal_square_root, nullptr,
                            sum_sq, inv_sqrt_sum_sq, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary square" << std::endl;
    return nullptr;
  }

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &params, input,
                             inv_sqrt_sum_sq, output,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary subtract" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
