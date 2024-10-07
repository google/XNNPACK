// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "xnnpack.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t FP32Attention(size_t b, size_t t, size_t h, size_t n, size_t s) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/4, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v0_dims = {{b, s, n, t}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v0_dims.size(), v0_dims.data(),
    /*data=*/nullptr,
    0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v1_dims = {{b, t, n, h}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v1_dims.size(), v1_dims.data(),
    /*data=*/nullptr,
    1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v2_dims = {{b, s, n, h}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v2_dims.size(), v2_dims.data(),
    /*data=*/nullptr,
    2, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  uint32_t v3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v3_dims = {{b, t, n, h}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v3_dims.size(), v3_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v3" << std::endl;
    return nullptr;
  }

  uint32_t v4 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v4_dims = {{b, n, t, h}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v4_dims.size(), v4_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v4" << std::endl;
    return nullptr;
  }

  uint32_t v5 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v5_dims = {{b, n, s, h}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v5_dims.size(), v5_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v5" << std::endl;
    return nullptr;
  }

  uint32_t v6 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v6_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v6_dims.size(), v6_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v6" << std::endl;
    return nullptr;
  }

  uint32_t v7 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v7_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v7_dims.size(), v7_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v7" << std::endl;
    return nullptr;
  }

  uint32_t v8 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v8_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v8_dims.size(), v8_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v8" << std::endl;
    return nullptr;
  }

  uint32_t v9 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v9_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v9_dims.size(), v9_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v9" << std::endl;
    return nullptr;
  }

  uint32_t v10 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v10_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v10_dims.size(), v10_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v10" << std::endl;
    return nullptr;
  }

  uint32_t v11 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v11_dims = {{b, n, t, t}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v11_dims.size(), v11_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v11" << std::endl;
    return nullptr;
  }

  uint32_t v12 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v12_dims = {{b, t, n, t}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v12_dims.size(), v12_dims.data(),
    /*data=*/nullptr,
    3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v12" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w13_data;
  uint32_t w13 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w13_dims = {{1}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w13_dims.size(), w13_dims.data(),
    /*data=*/w13_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w13" << std::endl;
    return nullptr;
  }

  static const std::array<int32_t, 4> w14_data = {
    0x000000, 0x000002, 0x000001, 0x000003,
  };
  uint32_t w14 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w14_dims = {{4}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_int32,
    w14_dims.size(), w14_dims.data(),
    /*data=*/w14_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w14" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w15_data;
  uint32_t w15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w15_dims = {{1}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w15_dims.size(), w15_dims.data(),
    /*data=*/w15_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w15" << std::endl;
    return nullptr;
  }

  static const std::array<int32_t, 4> w16_data = {
    0x000000, 0x000002, 0x000003, 0x000001,
  };
  uint32_t w16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w16_dims = {{4}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_int32,
    w16_dims.size(), w16_dims.data(),
    /*data=*/w16_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w16" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  std::generate(w13_data.begin(), w13_data.end(), std::ref(f32rng));
  std::generate(w15_data.begin(), w15_data.end(), std::ref(f32rng));

  xnn_binary_params v3_params = { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_multiply,
    &v3_params,
    v1,
    w13,
    v3,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v3_w14_v4 = { (size_t)w14_data[0], (size_t)w14_data[1], (size_t)w14_data[2], (size_t)w14_data[3] };
  status = xnn_define_static_transpose(
    subgraph,
    /*num_dims=*/perm_v3_w14_v4.size(),
    /*perm=*/perm_v3_w14_v4.data(),
    v3,
    v4,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v2_w14_v5 = { (size_t)w14_data[0], (size_t)w14_data[1], (size_t)w14_data[2], (size_t)w14_data[3] };
  status = xnn_define_static_transpose(
    subgraph,
    /*num_dims=*/perm_v2_w14_v5.size(),
    /*perm=*/perm_v2_w14_v5.data(),
    v2,
    v5,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(
    subgraph,
    v4,
    v5,
    v6,
    XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #3" << std::endl;
    return nullptr;
  }

  xnn_binary_params v7_params = { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_multiply,
    &v7_params,
    v6,
    w15,
    v7,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  status = xnn_define_tanh(
    subgraph,
    v7,
    v8,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  status = xnn_define_softmax(
    subgraph,
    v8,
    v9,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v0_w16_v10 = { (size_t)w16_data[0], (size_t)w16_data[1], (size_t)w16_data[2], (size_t)w16_data[3] };
  status = xnn_define_static_transpose(
    subgraph,
    /*num_dims=*/perm_v0_w16_v10.size(),
    /*perm=*/perm_v0_w16_v10.data(),
    v0,
    v10,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(
    subgraph,
    v9,
    v10,
    v11,
    XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #8" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v11_w14_v12 = { (size_t)w14_data[0], (size_t)w14_data[1], (size_t)w14_data[2], (size_t)w14_data[3] };
  status = xnn_define_static_transpose(
    subgraph,
    /*num_dims=*/perm_v11_w14_v12.size(),
    /*perm=*/perm_v11_w14_v12.data(),
    v11,
    v12,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
