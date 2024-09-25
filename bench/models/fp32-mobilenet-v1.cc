// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!

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

xnn_subgraph_t FP32MobileNetV1() {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v0_dims = {{1, 224, 224, 3}};
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
  std::array<size_t, 4> v1_dims = {{1, 112, 112, 32}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v1_dims.size(), v1_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v2_dims = {{1, 112, 112, 32}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v2_dims.size(), v2_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  uint32_t v3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v3_dims = {{1, 112, 112, 64}};
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
  std::array<size_t, 4> v4_dims = {{1, 56, 56, 64}};
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
  std::array<size_t, 4> v5_dims = {{1, 56, 56, 128}};
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
  std::array<size_t, 4> v6_dims = {{1, 56, 56, 128}};
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
  std::array<size_t, 4> v7_dims = {{1, 56, 56, 128}};
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
  std::array<size_t, 4> v8_dims = {{1, 28, 28, 128}};
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
  std::array<size_t, 4> v9_dims = {{1, 28, 28, 256}};
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
  std::array<size_t, 4> v10_dims = {{1, 28, 28, 256}};
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
  std::array<size_t, 4> v11_dims = {{1, 28, 28, 256}};
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
  std::array<size_t, 4> v12_dims = {{1, 14, 14, 256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v12_dims.size(), v12_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v12" << std::endl;
    return nullptr;
  }

  uint32_t v13 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v13_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v13_dims.size(), v13_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v13" << std::endl;
    return nullptr;
  }

  uint32_t v14 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v14_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v14_dims.size(), v14_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v14" << std::endl;
    return nullptr;
  }

  uint32_t v15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v15_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v15_dims.size(), v15_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v15" << std::endl;
    return nullptr;
  }

  uint32_t v16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v16_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v16_dims.size(), v16_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v16" << std::endl;
    return nullptr;
  }

  uint32_t v17 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v17_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v17_dims.size(), v17_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v17" << std::endl;
    return nullptr;
  }

  uint32_t v18 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v18_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v18_dims.size(), v18_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v18" << std::endl;
    return nullptr;
  }

  uint32_t v19 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v19_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v19_dims.size(), v19_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v19" << std::endl;
    return nullptr;
  }

  uint32_t v20 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v20_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v20_dims.size(), v20_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v20" << std::endl;
    return nullptr;
  }

  uint32_t v21 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v21_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v21_dims.size(), v21_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v21" << std::endl;
    return nullptr;
  }

  uint32_t v22 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v22_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v22_dims.size(), v22_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v22" << std::endl;
    return nullptr;
  }

  uint32_t v23 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v23_dims = {{1, 14, 14, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v23_dims.size(), v23_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v23" << std::endl;
    return nullptr;
  }

  uint32_t v24 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v24_dims = {{1, 7, 7, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v24_dims.size(), v24_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v24" << std::endl;
    return nullptr;
  }

  uint32_t v25 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v25_dims = {{1, 7, 7, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v25_dims.size(), v25_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v25" << std::endl;
    return nullptr;
  }

  uint32_t v26 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v26_dims = {{1, 7, 7, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v26_dims.size(), v26_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v26" << std::endl;
    return nullptr;
  }

  uint32_t v27 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v27_dims = {{1, 7, 7, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v27_dims.size(), v27_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v27" << std::endl;
    return nullptr;
  }

  uint32_t v28 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v28_dims = {{1, 1, 1, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v28_dims.size(), v28_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v28" << std::endl;
    return nullptr;
  }

  uint32_t v29 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v29_dims = {{1, 1, 1, 1001}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v29_dims.size(), v29_dims.data(),
    /*data=*/nullptr,
    1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v29" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(864, float)> w30_data;
  uint32_t w30 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w30_dims = {{32, 3, 3, 3}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w30_dims.size(), w30_dims.data(),
    /*data=*/w30_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w30" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(32, float)> w31_data;
  uint32_t w31 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w31_dims = {{32}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w31_dims.size(), w31_dims.data(),
    /*data=*/w31_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w31" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(288, float)> w32_data;
  uint32_t w32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w32_dims = {{1, 3, 3, 32}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w32_dims.size(), w32_dims.data(),
    /*data=*/w32_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w32" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(32, float)> w33_data;
  uint32_t w33 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w33_dims = {{32}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w33_dims.size(), w33_dims.data(),
    /*data=*/w33_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w33" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(2048, float)> w34_data;
  uint32_t w34 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w34_dims = {{64, 1, 1, 32}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w34_dims.size(), w34_dims.data(),
    /*data=*/w34_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w34" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(64, float)> w35_data;
  uint32_t w35 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w35_dims = {{64}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w35_dims.size(), w35_dims.data(),
    /*data=*/w35_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w35" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(576, float)> w36_data;
  uint32_t w36 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w36_dims = {{1, 3, 3, 64}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w36_dims.size(), w36_dims.data(),
    /*data=*/w36_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w36" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(64, float)> w37_data;
  uint32_t w37 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w37_dims = {{64}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w37_dims.size(), w37_dims.data(),
    /*data=*/w37_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w37" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(8192, float)> w38_data;
  uint32_t w38 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w38_dims = {{128, 1, 1, 64}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w38_dims.size(), w38_dims.data(),
    /*data=*/w38_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w38" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w39_data;
  uint32_t w39 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w39_dims = {{128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w39_dims.size(), w39_dims.data(),
    /*data=*/w39_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w39" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1152, float)> w40_data;
  uint32_t w40 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w40_dims = {{1, 3, 3, 128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w40_dims.size(), w40_dims.data(),
    /*data=*/w40_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w40" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w41_data;
  uint32_t w41 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w41_dims = {{128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w41_dims.size(), w41_dims.data(),
    /*data=*/w41_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w41" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(16384, float)> w42_data;
  uint32_t w42 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w42_dims = {{128, 1, 1, 128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w42_dims.size(), w42_dims.data(),
    /*data=*/w42_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w42" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w43_data;
  uint32_t w43 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w43_dims = {{128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w43_dims.size(), w43_dims.data(),
    /*data=*/w43_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w43" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1152, float)> w44_data;
  uint32_t w44 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w44_dims = {{1, 3, 3, 128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w44_dims.size(), w44_dims.data(),
    /*data=*/w44_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w44" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w45_data;
  uint32_t w45 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w45_dims = {{128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w45_dims.size(), w45_dims.data(),
    /*data=*/w45_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w45" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(32768, float)> w46_data;
  uint32_t w46 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w46_dims = {{256, 1, 1, 128}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w46_dims.size(), w46_dims.data(),
    /*data=*/w46_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w46" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w47_data;
  uint32_t w47 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w47_dims = {{256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w47_dims.size(), w47_dims.data(),
    /*data=*/w47_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w47" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(2304, float)> w48_data;
  uint32_t w48 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w48_dims = {{1, 3, 3, 256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w48_dims.size(), w48_dims.data(),
    /*data=*/w48_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w48" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w49_data;
  uint32_t w49 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w49_dims = {{256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w49_dims.size(), w49_dims.data(),
    /*data=*/w49_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w49" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(65536, float)> w50_data;
  uint32_t w50 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w50_dims = {{256, 1, 1, 256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w50_dims.size(), w50_dims.data(),
    /*data=*/w50_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w50" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w51_data;
  uint32_t w51 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w51_dims = {{256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w51_dims.size(), w51_dims.data(),
    /*data=*/w51_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w51" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(2304, float)> w52_data;
  uint32_t w52 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w52_dims = {{1, 3, 3, 256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w52_dims.size(), w52_dims.data(),
    /*data=*/w52_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w52" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w53_data;
  uint32_t w53 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w53_dims = {{256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w53_dims.size(), w53_dims.data(),
    /*data=*/w53_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w53" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(131072, float)> w54_data;
  uint32_t w54 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w54_dims = {{512, 1, 1, 256}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w54_dims.size(), w54_dims.data(),
    /*data=*/w54_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w54" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w55_data;
  uint32_t w55 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w55_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w55_dims.size(), w55_dims.data(),
    /*data=*/w55_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w55" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w56_data;
  uint32_t w56 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w56_dims = {{1, 3, 3, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w56_dims.size(), w56_dims.data(),
    /*data=*/w56_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w56" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w57_data;
  uint32_t w57 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w57_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w57_dims.size(), w57_dims.data(),
    /*data=*/w57_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w57" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w58_data;
  uint32_t w58 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w58_dims = {{512, 1, 1, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w58_dims.size(), w58_dims.data(),
    /*data=*/w58_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w58" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w59_data;
  uint32_t w59 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w59_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w59_dims.size(), w59_dims.data(),
    /*data=*/w59_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w59" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w60_data;
  uint32_t w60 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w60_dims = {{1, 3, 3, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w60_dims.size(), w60_dims.data(),
    /*data=*/w60_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w60" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w61_data;
  uint32_t w61 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w61_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w61_dims.size(), w61_dims.data(),
    /*data=*/w61_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w61" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w62_data;
  uint32_t w62 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w62_dims = {{512, 1, 1, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w62_dims.size(), w62_dims.data(),
    /*data=*/w62_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w62" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w63_data;
  uint32_t w63 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w63_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w63_dims.size(), w63_dims.data(),
    /*data=*/w63_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w63" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w64_data;
  uint32_t w64 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w64_dims = {{1, 3, 3, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w64_dims.size(), w64_dims.data(),
    /*data=*/w64_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w64" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w65_data;
  uint32_t w65 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w65_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w65_dims.size(), w65_dims.data(),
    /*data=*/w65_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w65" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w66_data;
  uint32_t w66 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w66_dims = {{512, 1, 1, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w66_dims.size(), w66_dims.data(),
    /*data=*/w66_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w66" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w67_data;
  uint32_t w67 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w67_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w67_dims.size(), w67_dims.data(),
    /*data=*/w67_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w67" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w68_data;
  uint32_t w68 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w68_dims = {{1, 3, 3, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w68_dims.size(), w68_dims.data(),
    /*data=*/w68_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w68" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w69_data;
  uint32_t w69 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w69_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w69_dims.size(), w69_dims.data(),
    /*data=*/w69_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w69" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w70_data;
  uint32_t w70 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w70_dims = {{512, 1, 1, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w70_dims.size(), w70_dims.data(),
    /*data=*/w70_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w70" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w71_data;
  uint32_t w71 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w71_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w71_dims.size(), w71_dims.data(),
    /*data=*/w71_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w71" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w72_data;
  uint32_t w72 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w72_dims = {{1, 3, 3, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w72_dims.size(), w72_dims.data(),
    /*data=*/w72_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w72" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w73_data;
  uint32_t w73 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w73_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w73_dims.size(), w73_dims.data(),
    /*data=*/w73_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w73" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w74_data;
  uint32_t w74 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w74_dims = {{512, 1, 1, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w74_dims.size(), w74_dims.data(),
    /*data=*/w74_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w74" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w75_data;
  uint32_t w75 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w75_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w75_dims.size(), w75_dims.data(),
    /*data=*/w75_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w75" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w76_data;
  uint32_t w76 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w76_dims = {{1, 3, 3, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w76_dims.size(), w76_dims.data(),
    /*data=*/w76_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w76" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w77_data;
  uint32_t w77 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w77_dims = {{512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w77_dims.size(), w77_dims.data(),
    /*data=*/w77_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w77" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(524288, float)> w78_data;
  uint32_t w78 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w78_dims = {{1024, 1, 1, 512}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w78_dims.size(), w78_dims.data(),
    /*data=*/w78_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w78" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> w79_data;
  uint32_t w79 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w79_dims = {{1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w79_dims.size(), w79_dims.data(),
    /*data=*/w79_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w79" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(9216, float)> w80_data;
  uint32_t w80 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w80_dims = {{1, 3, 3, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w80_dims.size(), w80_dims.data(),
    /*data=*/w80_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w80" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> w81_data;
  uint32_t w81 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w81_dims = {{1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w81_dims.size(), w81_dims.data(),
    /*data=*/w81_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w81" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1048576, float)> w82_data;
  uint32_t w82 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w82_dims = {{1024, 1, 1, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w82_dims.size(), w82_dims.data(),
    /*data=*/w82_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w82" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> w83_data;
  uint32_t w83 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w83_dims = {{1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w83_dims.size(), w83_dims.data(),
    /*data=*/w83_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w83" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1025024, float)> w84_data;
  uint32_t w84 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w84_dims = {{1001, 1, 1, 1024}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w84_dims.size(), w84_dims.data(),
    /*data=*/w84_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w84" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1001, float)> w85_data;
  uint32_t w85 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w85_dims = {{1001}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    w85_dims.size(), w85_dims.data(),
    /*data=*/w85_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w85" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  std::generate(w30_data.begin(), w30_data.end(), std::ref(f32rng));
  std::generate(w31_data.begin(), w31_data.end(), std::ref(f32rng));
  std::generate(w32_data.begin(), w32_data.end(), std::ref(f32rng));
  std::generate(w33_data.begin(), w33_data.end(), std::ref(f32rng));
  std::generate(w34_data.begin(), w34_data.end(), std::ref(f32rng));
  std::generate(w35_data.begin(), w35_data.end(), std::ref(f32rng));
  std::generate(w36_data.begin(), w36_data.end(), std::ref(f32rng));
  std::generate(w37_data.begin(), w37_data.end(), std::ref(f32rng));
  std::generate(w38_data.begin(), w38_data.end(), std::ref(f32rng));
  std::generate(w39_data.begin(), w39_data.end(), std::ref(f32rng));
  std::generate(w40_data.begin(), w40_data.end(), std::ref(f32rng));
  std::generate(w41_data.begin(), w41_data.end(), std::ref(f32rng));
  std::generate(w42_data.begin(), w42_data.end(), std::ref(f32rng));
  std::generate(w43_data.begin(), w43_data.end(), std::ref(f32rng));
  std::generate(w44_data.begin(), w44_data.end(), std::ref(f32rng));
  std::generate(w45_data.begin(), w45_data.end(), std::ref(f32rng));
  std::generate(w46_data.begin(), w46_data.end(), std::ref(f32rng));
  std::generate(w47_data.begin(), w47_data.end(), std::ref(f32rng));
  std::generate(w48_data.begin(), w48_data.end(), std::ref(f32rng));
  std::generate(w49_data.begin(), w49_data.end(), std::ref(f32rng));
  std::generate(w50_data.begin(), w50_data.end(), std::ref(f32rng));
  std::generate(w51_data.begin(), w51_data.end(), std::ref(f32rng));
  std::generate(w52_data.begin(), w52_data.end(), std::ref(f32rng));
  std::generate(w53_data.begin(), w53_data.end(), std::ref(f32rng));
  std::generate(w54_data.begin(), w54_data.end(), std::ref(f32rng));
  std::generate(w55_data.begin(), w55_data.end(), std::ref(f32rng));
  std::generate(w56_data.begin(), w56_data.end(), std::ref(f32rng));
  std::generate(w57_data.begin(), w57_data.end(), std::ref(f32rng));
  std::generate(w58_data.begin(), w58_data.end(), std::ref(f32rng));
  std::generate(w59_data.begin(), w59_data.end(), std::ref(f32rng));
  std::generate(w60_data.begin(), w60_data.end(), std::ref(f32rng));
  std::generate(w61_data.begin(), w61_data.end(), std::ref(f32rng));
  std::generate(w62_data.begin(), w62_data.end(), std::ref(f32rng));
  std::generate(w63_data.begin(), w63_data.end(), std::ref(f32rng));
  std::generate(w64_data.begin(), w64_data.end(), std::ref(f32rng));
  std::generate(w65_data.begin(), w65_data.end(), std::ref(f32rng));
  std::generate(w66_data.begin(), w66_data.end(), std::ref(f32rng));
  std::generate(w67_data.begin(), w67_data.end(), std::ref(f32rng));
  std::generate(w68_data.begin(), w68_data.end(), std::ref(f32rng));
  std::generate(w69_data.begin(), w69_data.end(), std::ref(f32rng));
  std::generate(w70_data.begin(), w70_data.end(), std::ref(f32rng));
  std::generate(w71_data.begin(), w71_data.end(), std::ref(f32rng));
  std::generate(w72_data.begin(), w72_data.end(), std::ref(f32rng));
  std::generate(w73_data.begin(), w73_data.end(), std::ref(f32rng));
  std::generate(w74_data.begin(), w74_data.end(), std::ref(f32rng));
  std::generate(w75_data.begin(), w75_data.end(), std::ref(f32rng));
  std::generate(w76_data.begin(), w76_data.end(), std::ref(f32rng));
  std::generate(w77_data.begin(), w77_data.end(), std::ref(f32rng));
  std::generate(w78_data.begin(), w78_data.end(), std::ref(f32rng));
  std::generate(w79_data.begin(), w79_data.end(), std::ref(f32rng));
  std::generate(w80_data.begin(), w80_data.end(), std::ref(f32rng));
  std::generate(w81_data.begin(), w81_data.end(), std::ref(f32rng));
  std::generate(w82_data.begin(), w82_data.end(), std::ref(f32rng));
  std::generate(w83_data.begin(), w83_data.end(), std::ref(f32rng));
  std::generate(w84_data.begin(), w84_data.end(), std::ref(f32rng));
  std::generate(w85_data.begin(), w85_data.end(), std::ref(f32rng));

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/3,
    /*group_output_channels=*/32,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v0,
    w30,
    w31,
    v1,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/32,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v1,
    w32,
    w33,
    v2,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/64,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v2,
    w34,
    w35,
    v3,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/64,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v3,
    w36,
    w37,
    v4,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #3" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/128,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v4,
    w38,
    w39,
    v5,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/128,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v5,
    w40,
    w41,
    v6,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/128,
    /*group_output_channels=*/128,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v6,
    w42,
    w43,
    v7,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/128,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v7,
    w44,
    w45,
    v8,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/128,
    /*group_output_channels=*/256,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v8,
    w46,
    w47,
    v9,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #8" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/256,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v9,
    w48,
    w49,
    v10,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/256,
    /*group_output_channels=*/256,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v10,
    w50,
    w51,
    v11,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #10" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/256,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v11,
    w52,
    w53,
    v12,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #11" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/256,
    /*group_output_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v12,
    w54,
    w55,
    v13,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #12" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v13,
    w56,
    w57,
    v14,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #13" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v14,
    w58,
    w59,
    v15,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #14" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v15,
    w60,
    w61,
    v16,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #15" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v16,
    w62,
    w63,
    v17,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #16" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v17,
    w64,
    w65,
    v18,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #17" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v18,
    w66,
    w67,
    v19,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #18" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v19,
    w68,
    w69,
    v20,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #19" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v20,
    w70,
    w71,
    v21,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #20" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v21,
    w72,
    w73,
    v22,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #21" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v22,
    w74,
    w75,
    v23,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #22" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/512,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v23,
    w76,
    w77,
    v24,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #23" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/1024,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v24,
    w78,
    w79,
    v25,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #24" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/1024,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v25,
    w80,
    w81,
    v26,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #25" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/1024,
    /*group_output_channels=*/1024,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v26,
    w82,
    w83,
    v27,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #26" << std::endl;
    return nullptr;
  }

  status = xnn_define_average_pooling_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*pooling_height=*/7, /*pooling_width=*/7,
    /*stride_height=*/2, /*stride_width=*/2,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v27,
    v28,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #27" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/1024,
    /*group_output_channels=*/1001,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v28,
    w84,
    w85,
    v29,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #28" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
