// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

#include "include/xnnpack.h"

namespace models {

xnn_subgraph_t FP32Elementwise(size_t batch_size, size_t num_elements,
                               size_t depth) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/3, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::array<size_t, 2> dims = {{batch_size, num_elements}};

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 2, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};
  uint32_t mul = v0;
  uint32_t add = v1;
  for (int i = 0; i < depth; ++i) {
    uint32_t new_add = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &new_add);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor add" << std::endl;
      return nullptr;
    }

    status =
        xnn_define_binary(subgraph, xnn_binary_add, &params, mul, add, new_add,
                          /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create node #0" << std::endl;
      return nullptr;
    }
    add = new_add;

    mul = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &mul);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor mul" << std::endl;
      return nullptr;
    }

    status = xnn_define_binary(subgraph, xnn_binary_multiply, &params, new_add,
                               new_add, mul,
                               /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create node #0" << std::endl;
      return nullptr;
    }
  }

  status =
      xnn_define_binary(subgraph, xnn_binary_subtract, &params, mul, v0, output,
                        /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
