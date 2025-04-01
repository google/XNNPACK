// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "bench/models/models.h"
#include "include/xnnpack.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t FP32DepthwiseSeparable(size_t w, size_t h, size_t kw, size_t ci,
                                      size_t co,
                                      FP32DepthwiseSeparableWeights& weights) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  uint32_t v0 = 0;
  std::array<size_t, 4> v0_dims = {{1, h, w, ci}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, v0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v1_dims = {{1, h, w, ci}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v1_dims.size(), v1_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = 1;
  std::array<size_t, 4> v2_dims = {{1, h, w, co}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v2_dims.size(), v2_dims.data(),
      /*data=*/nullptr, v2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  weights.w0 =
      std::vector<float>(kw * kw * ci + XNN_EXTRA_BYTES / sizeof(float));
  weights.w1 = std::vector<float>(ci + XNN_EXTRA_BYTES / sizeof(float));
  weights.w2 = std::vector<float>(co * ci + XNN_EXTRA_BYTES / sizeof(float));
  weights.w3 = std::vector<float>(co + XNN_EXTRA_BYTES / sizeof(float));

  uint32_t w0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w0_dims = {{1, kw, kw, ci}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w0_dims.size(), w0_dims.data(),
      /*data=*/weights.w0.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w0" << std::endl;
    return nullptr;
  }

  uint32_t w1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w1_dims = {{ci}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w1_dims.size(), w1_dims.data(),
      /*data=*/weights.w1.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w1" << std::endl;
    return nullptr;
  }

  uint32_t w2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w2_dims = {{co, 1, 1, ci}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w2_dims.size(), w2_dims.data(),
      /*data=*/weights.w2.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w2" << std::endl;
    return nullptr;
  }

  uint32_t w3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w3_dims = {{co}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w3_dims.size(), w3_dims.data(),
      /*data=*/weights.w3.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w3" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));
  std::generate(weights.w0.begin(), weights.w0.end(), std::ref(f32rng));
  std::generate(weights.w1.begin(), weights.w1.end(), std::ref(f32rng));
  std::generate(weights.w2.begin(), weights.w2.end(), std::ref(f32rng));
  std::generate(weights.w3.begin(), weights.w3.end(), std::ref(f32rng));

  status = xnn_define_depthwise_convolution_2d(
      subgraph,
      /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1,
      /*padding_left=*/1,
      /*kernel_height=*/kw, /*kernel_width=*/kw,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*depth_multiplier=*/1,
      /*input_channels=*/ci,
      /*output_min=*/0.0f, /*output_max=*/6.0f, v0, w0, w1, v1,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph,
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/1, /*kernel_width=*/1,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/ci,
      /*group_output_channels=*/co,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v1, w2, w3, v2,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
