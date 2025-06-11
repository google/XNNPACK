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

#include "bench/subgraph/models.h"
#include "include/xnnpack.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t QD8TransformerBlock(size_t batch_size, size_t sequence_length,
                                   size_t embedding_dim, size_t num_heads,
                                   size_t head_dim, size_t hidden_dim) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;  // NOLINT(runtime/random_device)
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v0_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v1_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v1_dims.size(), v1_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v2_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v2_dims.size(), v2_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  uint32_t v3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v3_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v3_dims.size(), v3_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v3" << std::endl;
    return nullptr;
  }

  uint32_t v4 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v4_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v4_dims.size(), v4_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v4" << std::endl;
    return nullptr;
  }

  uint32_t v5 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v5_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v5_dims.size(), v5_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v5" << std::endl;
    return nullptr;
  }

  uint32_t v6 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v6_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v6_dims.size(), v6_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v6" << std::endl;
    return nullptr;
  }

  uint32_t v7 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v7_dims = {{batch_size, sequence_length, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v7_dims.size(), v7_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v7" << std::endl;
    return nullptr;
  }

  uint32_t v8 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v8_dims = {{batch_size, sequence_length, 1, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v8_dims.size(), v8_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v8" << std::endl;
    return nullptr;
  }

  uint32_t v9 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v9_dims = {{batch_size, 1, head_dim, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v9_dims.size(), v9_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v9" << std::endl;
    return nullptr;
  }

  uint32_t v10 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v10_dims = {
      {batch_size, sequence_length, num_heads * head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v10_dims.size(), v10_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v10" << std::endl;
    return nullptr;
  }

  uint32_t v11 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v11_dims = {
      {batch_size, sequence_length, 1, num_heads, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v11_dims.size(), v11_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v11" << std::endl;
    return nullptr;
  }

  uint32_t v12 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v12_dims = {
      {batch_size, 1, num_heads, sequence_length, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v12_dims.size(), v12_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v12" << std::endl;
    return nullptr;
  }

  uint32_t v13 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v13_dims = {
      {batch_size, 1, num_heads * sequence_length, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v13_dims.size(), v13_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v13" << std::endl;
    return nullptr;
  }

  uint32_t v14 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v14_dims = {
      {batch_size, 1, num_heads * sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v14_dims.size(), v14_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v14" << std::endl;
    return nullptr;
  }

  uint32_t v15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v15_dims = {
      {batch_size, 1, num_heads, sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v15_dims.size(), v15_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v15" << std::endl;
    return nullptr;
  }

  uint32_t v16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v16_dims = {
      {batch_size, 1, num_heads, sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v16_dims.size(), v16_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v16" << std::endl;
    return nullptr;
  }

  uint32_t v17 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v17_dims = {
      {batch_size, 1, num_heads, sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v17_dims.size(), v17_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v17" << std::endl;
    return nullptr;
  }

  uint32_t v18 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v18_dims = {
      {batch_size, 1, num_heads, sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v18_dims.size(), v18_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v18" << std::endl;
    return nullptr;
  }

  uint32_t v19 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v19_dims = {
      {batch_size, 1, num_heads * sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v19_dims.size(), v19_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v19" << std::endl;
    return nullptr;
  }

  uint32_t v20 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v20_dims = {{batch_size, 1, sequence_length, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v20_dims.size(), v20_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v20" << std::endl;
    return nullptr;
  }

  uint32_t v21 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v21_dims = {
      {batch_size, 1, num_heads * sequence_length, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v21_dims.size(), v21_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v21" << std::endl;
    return nullptr;
  }

  uint32_t v22 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v22_dims = {
      {batch_size, 1, num_heads, sequence_length, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v22_dims.size(), v22_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v22" << std::endl;
    return nullptr;
  }

  uint32_t v23 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> v23_dims = {
      {batch_size, sequence_length, 1, num_heads, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v23_dims.size(), v23_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v23" << std::endl;
    return nullptr;
  }

  uint32_t v24 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v24_dims = {
      {batch_size, sequence_length, num_heads * head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v24_dims.size(), v24_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v24" << std::endl;
    return nullptr;
  }

  uint32_t v25 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v25_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v25_dims.size(), v25_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v25" << std::endl;
    return nullptr;
  }

  uint32_t v26 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v26_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v26_dims.size(), v26_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v26" << std::endl;
    return nullptr;
  }

  uint32_t v27 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v27_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v27_dims.size(), v27_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v27" << std::endl;
    return nullptr;
  }

  uint32_t v28 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v28_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v28_dims.size(), v28_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v28" << std::endl;
    return nullptr;
  }

  uint32_t v29 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v29_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v29_dims.size(), v29_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v29" << std::endl;
    return nullptr;
  }

  uint32_t v30 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v30_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v30_dims.size(), v30_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v30" << std::endl;
    return nullptr;
  }

  uint32_t v31 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v31_dims = {{batch_size, sequence_length, 1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v31_dims.size(), v31_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v31" << std::endl;
    return nullptr;
  }

  uint32_t v32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v32_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v32_dims.size(), v32_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v32" << std::endl;
    return nullptr;
  }

  uint32_t v33 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v33_dims = {
      {batch_size, sequence_length, hidden_dim / 2}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v33_dims.size(), v33_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v33" << std::endl;
    return nullptr;
  }

  uint32_t v34 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v34_dims = {
      {batch_size, sequence_length, hidden_dim / 2}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v34_dims.size(), v34_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v34" << std::endl;
    return nullptr;
  }

  uint32_t v35 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v35_dims = {
      {batch_size, sequence_length, hidden_dim / 2}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v35_dims.size(), v35_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v35" << std::endl;
    return nullptr;
  }

  uint32_t v36 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v36_dims = {
      {batch_size, sequence_length, hidden_dim / 2}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v36_dims.size(), v36_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v36" << std::endl;
    return nullptr;
  }

  uint32_t v37 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v37_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v37_dims.size(), v37_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v37" << std::endl;
    return nullptr;
  }

  uint32_t v38 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> v38_dims = {
      {batch_size, sequence_length, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v38_dims.size(), v38_dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v38" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 1> w39_data;
  w39_data = {
      -0x00001,
  };
  uint32_t w39 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w39_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w39_dims.size(), w39_dims.data(),
      /*data=*/w39_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w39" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w40_data;
  uint32_t w40 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w40_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w40_dims.size(), w40_dims.data(),
      /*data=*/w40_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w40" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w41_data;
  uint32_t w41 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w41_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w41_dims.size(), w41_dims.data(),
      /*data=*/w41_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w41" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w42_data;
  w42_data.resize(XNN_PAD_EXTRA_BYTES(head_dim * embedding_dim, int8_t));
  uint32_t w42 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w42_dims = {{head_dim, embedding_dim}};
  static std::vector<float> w42_scale;
  w42_scale.resize(head_dim);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w42_scale.begin(), w42_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w42_scale.data(), w42_dims.size(), 0, w42_dims.data(),
      /*data=*/w42_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w42" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 4> w43_data;
  w43_data = {
      -0x00001,
      (int32_t)sequence_length,
      0x000001,
      (int32_t)head_dim,
  };
  uint32_t w43 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w43_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w43_dims.size(), w43_dims.data(),
      /*data=*/w43_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w43" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 4> w44_data;
  w44_data = {
      0x000000,
      0x000002,
      0x000003,
      0x000001,
  };
  uint32_t w44 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w44_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w44_dims.size(), w44_dims.data(),
      /*data=*/w44_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w44" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w45_data;
  w45_data.resize(
      XNN_PAD_EXTRA_BYTES(num_heads * head_dim * embedding_dim, int8_t));
  uint32_t w45 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w45_dims = {{num_heads * head_dim, embedding_dim}};
  static std::vector<float> w45_scale;
  w45_scale.resize(num_heads * head_dim);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w45_scale.begin(), w45_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w45_scale.data(), w45_dims.size(), 0, w45_dims.data(),
      /*data=*/w45_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w45" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 5> w46_data;
  w46_data = {
      (int32_t)batch_size, (int32_t)sequence_length, 0x000001,
      (int32_t)num_heads,  (int32_t)head_dim,
  };
  uint32_t w46 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w46_dims = {{5}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w46_dims.size(), w46_dims.data(),
      /*data=*/w46_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w46" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 5> w47_data;
  w47_data = {
      0x000000, 0x000002, 0x000003, 0x000001, 0x000004,
  };
  uint32_t w47 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w47_dims = {{5}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w47_dims.size(), w47_dims.data(),
      /*data=*/w47_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w47" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 4> w48_data;
  w48_data = {
      -0x00001,
      0x000001,
      (int32_t)(sequence_length * num_heads),
      (int32_t)head_dim,
  };
  uint32_t w48 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w48_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w48_dims.size(), w48_dims.data(),
      /*data=*/w48_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w48" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 5> w49_data;
  w49_data = {
      -0x00001,
      0x000001,
      (int32_t)num_heads,
      (int32_t)sequence_length,
      (int32_t)sequence_length,
  };
  uint32_t w49 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w49_dims = {{5}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w49_dims.size(), w49_dims.data(),
      /*data=*/w49_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w49" << std::endl;
    return nullptr;
  }

  static std::vector<float> w50_data;
  w50_data.resize(XNN_PAD_EXTRA_BYTES(
      batch_size * sequence_length * sequence_length, float));
  uint32_t w50 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> w50_dims = {
      {batch_size, 1, 1, sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w50_dims.size(), w50_dims.data(),
      /*data=*/w50_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w50" << std::endl;
    return nullptr;
  }

  static std::vector<float> w51_data;
  w51_data.resize(XNN_PAD_EXTRA_BYTES(
      batch_size * sequence_length * sequence_length, float));
  uint32_t w51 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 5> w51_dims = {
      {batch_size, 1, 1, sequence_length, sequence_length}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w51_dims.size(), w51_dims.data(),
      /*data=*/w51_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w51" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 4> w52_data;
  w52_data = {
      -0x00001,
      0x000001,
      (int32_t)(sequence_length * num_heads),
      (int32_t)sequence_length,
  };
  uint32_t w52 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w52_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w52_dims.size(), w52_dims.data(),
      /*data=*/w52_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w52" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 4> w53_data;
  w53_data = {
      (int32_t)batch_size,
      0x000001,
      (int32_t)sequence_length,
      (int32_t)head_dim,
  };
  uint32_t w53 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w53_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w53_dims.size(), w53_dims.data(),
      /*data=*/w53_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w53" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 5> w54_data;
  w54_data = {
      -0x00001,          0x000001, (int32_t)num_heads, (int32_t)sequence_length,
      (int32_t)head_dim,
  };
  uint32_t w54 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w54_dims = {{5}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w54_dims.size(), w54_dims.data(),
      /*data=*/w54_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w54" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 5> w55_data;
  w55_data = {
      0x000000, 0x000003, 0x000001, 0x000002, 0x000004,
  };
  uint32_t w55 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w55_dims = {{5}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w55_dims.size(), w55_dims.data(),
      /*data=*/w55_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w55" << std::endl;
    return nullptr;
  }

  static std::array<int32_t, 3> w56_data;
  w56_data = {
      -0x00001,
      (int32_t)sequence_length,
      (int32_t)(num_heads * head_dim),
  };
  uint32_t w56 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w56_dims = {{3}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w56_dims.size(), w56_dims.data(),
      /*data=*/w56_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w56" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w57_data;
  w57_data.resize(
      XNN_PAD_EXTRA_BYTES(embedding_dim * num_heads * head_dim, int8_t));
  uint32_t w57 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w57_dims = {{embedding_dim, num_heads * head_dim}};
  static std::vector<float> w57_scale;
  w57_scale.resize(embedding_dim);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w57_scale.begin(), w57_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w57_scale.data(), w57_dims.size(), 0, w57_dims.data(),
      /*data=*/w57_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w57" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w58_data;
  w58_data.resize(
      XNN_PAD_EXTRA_BYTES((hidden_dim / 2) * embedding_dim, int8_t));
  uint32_t w58 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w58_dims = {{hidden_dim / 2, embedding_dim}};
  static std::vector<float> w58_scale;
  w58_scale.resize(hidden_dim / 2);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w58_scale.begin(), w58_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w58_scale.data(), w58_dims.size(), 0, w58_dims.data(),
      /*data=*/w58_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w58" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w59_data;
  w59_data.resize(
      XNN_PAD_EXTRA_BYTES((hidden_dim / 2) * embedding_dim, int8_t));
  uint32_t w59 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w59_dims = {{hidden_dim / 2, embedding_dim}};
  static std::vector<float> w59_scale;
  w59_scale.resize(hidden_dim / 2);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w59_scale.begin(), w59_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w59_scale.data(), w59_dims.size(), 0, w59_dims.data(),
      /*data=*/w59_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w59" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w60_data;
  w60_data.resize(
      XNN_PAD_EXTRA_BYTES(embedding_dim * (hidden_dim / 2), int8_t));
  uint32_t w60 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w60_dims = {{embedding_dim, hidden_dim / 2}};
  static std::vector<float> w60_scale;
  w60_scale.resize(embedding_dim);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w60_scale.begin(), w60_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w60_scale.data(), w60_dims.size(), 0, w60_dims.data(),
      /*data=*/w60_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w60" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));
  auto qc8rng = std::bind(
      std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max()),
      std::ref(rng));
  std::generate(w40_data.begin(), w40_data.end(), std::ref(f32rng));
  std::generate(w41_data.begin(), w41_data.end(), std::ref(f32rng));
  std::generate(w42_data.begin(), w42_data.end(), std::ref(qc8rng));
  std::generate(w45_data.begin(), w45_data.end(), std::ref(qc8rng));
  std::generate(w50_data.begin(), w50_data.end(), std::ref(f32rng));
  std::generate(w51_data.begin(), w51_data.end(), std::ref(f32rng));
  std::generate(w57_data.begin(), w57_data.end(), std::ref(qc8rng));
  std::generate(w58_data.begin(), w58_data.end(), std::ref(qc8rng));
  std::generate(w59_data.begin(), w59_data.end(), std::ref(qc8rng));
  std::generate(w60_data.begin(), w60_data.end(), std::ref(qc8rng));

  status = xnn_define_unary(subgraph, xnn_unary_square,
                            /*params=*/nullptr, v0, v1, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  std::array<int64_t, 1> axis_v1_w39_v2 = {(int64_t)w39_data[0]};
  status =
      xnn_define_static_reduce_v2(subgraph,
                                  /*reduce_operator=*/xnn_reduce_mean,
                                  /*num_reduction_axes=*/axis_v1_w39_v2.size(),
                                  /*reduction_axes=*/axis_v1_w39_v2.data(),
                                  /*input_id=*/v1,
                                  /*output_id=*/v2,
                                  /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  xnn_binary_params v3_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v3_params,
                             /*input1_id=*/v2,
                             /*input2_id=*/w40,
                             /*output_id=*/v3,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_square_root,
                            /*params=*/nullptr, v3, v4, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #3" << std::endl;
    return nullptr;
  }

  xnn_binary_params v5_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_divide, &v5_params,
                             /*input1_id=*/w41,
                             /*input2_id=*/v4,
                             /*output_id=*/v5,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  xnn_binary_params v6_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v6_params,
                             /*input1_id=*/v0,
                             /*input2_id=*/v5,
                             /*output_id=*/v6,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  uint32_t v6_w42_v7_dq = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v6_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v6_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v6_w42_v7_dq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v6, /*output_id=*/v6_w42_v7_dq,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v6_w42_v7_dq,
      /*filter_id=*/w42,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v7,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> shape_v7_w43_v8 = {
      (size_t)std::max(w43_data[0], 0), (size_t)std::max(w43_data[1], 0),
      (size_t)std::max(w43_data[2], 0), (size_t)std::max(w43_data[3], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v7_w43_v8.size(),
                                     /*new_shape=*/shape_v7_w43_v8.data(),
                                     /*input_id=*/v7,
                                     /*output_id=*/v8,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v8_w44_v9 = {
      (size_t)w44_data[0], (size_t)w44_data[1], (size_t)w44_data[2],
      (size_t)w44_data[3]};
  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/perm_v8_w44_v9.size(),
                                       /*perm=*/perm_v8_w44_v9.data(),
                                       /*input_id=*/v8,
                                       /*output_id=*/v9,
                                       /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #8" << std::endl;
    return nullptr;
  }

  uint32_t v6_w45_v10_dq = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v6_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v6_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v6_w45_v10_dq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v6, /*output_id=*/v6_w45_v10_dq,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v6_w45_v10_dq,
      /*filter_id=*/w45,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v10,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  std::array<size_t, 5> shape_v10_w46_v11 = {
      (size_t)std::max(w46_data[0], 0), (size_t)std::max(w46_data[1], 0),
      (size_t)std::max(w46_data[2], 0), (size_t)std::max(w46_data[3], 0),
      (size_t)std::max(w46_data[4], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v10_w46_v11.size(),
                                     /*new_shape=*/shape_v10_w46_v11.data(),
                                     /*input_id=*/v10,
                                     /*output_id=*/v11,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #10" << std::endl;
    return nullptr;
  }

  std::array<size_t, 5> perm_v11_w47_v12 = {
      (size_t)w47_data[0], (size_t)w47_data[1], (size_t)w47_data[2],
      (size_t)w47_data[3], (size_t)w47_data[4]};
  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/perm_v11_w47_v12.size(),
                                       /*perm=*/perm_v11_w47_v12.data(),
                                       /*input_id=*/v11,
                                       /*output_id=*/v12,
                                       /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #11" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> shape_v12_w48_v13 = {
      (size_t)std::max(w48_data[0], 0), (size_t)std::max(w48_data[1], 0),
      (size_t)std::max(w48_data[2], 0), (size_t)std::max(w48_data[3], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v12_w48_v13.size(),
                                     /*new_shape=*/shape_v12_w48_v13.data(),
                                     /*input_id=*/v12,
                                     /*output_id=*/v13,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #12" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph,
                                            /*input1_id=*/v13,
                                            /*input2_id=*/v9,
                                            /*output_id=*/v14,
                                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #13" << std::endl;
    return nullptr;
  }

  std::array<size_t, 5> shape_v14_w49_v15 = {
      (size_t)std::max(w49_data[0], 0), (size_t)std::max(w49_data[1], 0),
      (size_t)std::max(w49_data[2], 0), (size_t)std::max(w49_data[3], 0),
      (size_t)std::max(w49_data[4], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v14_w49_v15.size(),
                                     /*new_shape=*/shape_v14_w49_v15.data(),
                                     /*input_id=*/v14,
                                     /*output_id=*/v15,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #14" << std::endl;
    return nullptr;
  }

  xnn_binary_params v16_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v16_params,
                             /*input1_id=*/v15,
                             /*input2_id=*/w50,
                             /*output_id=*/v16,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #15" << std::endl;
    return nullptr;
  }

  status = xnn_define_softmax(subgraph, v16, v17,
                              /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #16" << std::endl;
    return nullptr;
  }

  xnn_binary_params v18_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v18_params,
                             /*input1_id=*/v17,
                             /*input2_id=*/w51,
                             /*output_id=*/v18,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #17" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> shape_v18_w52_v19 = {
      (size_t)std::max(w52_data[0], 0), (size_t)std::max(w52_data[1], 0),
      (size_t)std::max(w52_data[2], 0), (size_t)std::max(w52_data[3], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v18_w52_v19.size(),
                                     /*new_shape=*/shape_v18_w52_v19.data(),
                                     /*input_id=*/v18,
                                     /*output_id=*/v19,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #18" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> shape_v7_w53_v20 = {
      (size_t)std::max(w53_data[0], 0), (size_t)std::max(w53_data[1], 0),
      (size_t)std::max(w53_data[2], 0), (size_t)std::max(w53_data[3], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v7_w53_v20.size(),
                                     /*new_shape=*/shape_v7_w53_v20.data(),
                                     /*input_id=*/v7,
                                     /*output_id=*/v20,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #19" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph,
                                            /*input1_id=*/v19,
                                            /*input2_id=*/v20,
                                            /*output_id=*/v21,
                                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #20" << std::endl;
    return nullptr;
  }

  std::array<size_t, 5> shape_v21_w54_v22 = {
      (size_t)std::max(w54_data[0], 0), (size_t)std::max(w54_data[1], 0),
      (size_t)std::max(w54_data[2], 0), (size_t)std::max(w54_data[3], 0),
      (size_t)std::max(w54_data[4], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v21_w54_v22.size(),
                                     /*new_shape=*/shape_v21_w54_v22.data(),
                                     /*input_id=*/v21,
                                     /*output_id=*/v22,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #21" << std::endl;
    return nullptr;
  }

  std::array<size_t, 5> perm_v22_w55_v23 = {
      (size_t)w55_data[0], (size_t)w55_data[1], (size_t)w55_data[2],
      (size_t)w55_data[3], (size_t)w55_data[4]};
  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/perm_v22_w55_v23.size(),
                                       /*perm=*/perm_v22_w55_v23.data(),
                                       /*input_id=*/v22,
                                       /*output_id=*/v23,
                                       /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #22" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> shape_v23_w56_v24 = {(size_t)std::max(w56_data[0], 0),
                                             (size_t)std::max(w56_data[1], 0),
                                             (size_t)std::max(w56_data[2], 0)};
  status = xnn_define_static_reshape(subgraph,
                                     /*num_dims=*/shape_v23_w56_v24.size(),
                                     /*new_shape=*/shape_v23_w56_v24.data(),
                                     /*input_id=*/v23,
                                     /*output_id=*/v24,
                                     /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #23" << std::endl;
    return nullptr;
  }

  uint32_t v24_w57_v25_dq = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v24_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v24_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v24_w57_v25_dq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v24, /*output_id=*/v24_w57_v25_dq,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v24_w57_v25_dq,
      /*filter_id=*/w57,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v25,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #24" << std::endl;
    return nullptr;
  }

  xnn_binary_params v26_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v26_params,
                             /*input1_id=*/v0,
                             /*input2_id=*/v25,
                             /*output_id=*/v26,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #25" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_square,
                            /*params=*/nullptr, v26, v27, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #26" << std::endl;
    return nullptr;
  }

  std::array<int64_t, 1> axis_v27_w39_v28 = {(int64_t)w39_data[0]};
  status = xnn_define_static_reduce_v2(
      subgraph,
      /*reduce_operator=*/xnn_reduce_mean,
      /*num_reduction_axes=*/axis_v27_w39_v28.size(),
      /*reduction_axes=*/axis_v27_w39_v28.data(),
      /*input_id=*/v27,
      /*output_id=*/v28,
      /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #27" << std::endl;
    return nullptr;
  }

  xnn_binary_params v29_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v29_params,
                             /*input1_id=*/v28,
                             /*input2_id=*/w40,
                             /*output_id=*/v29,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #28" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_square_root,
                            /*params=*/nullptr, v29, v30, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #29" << std::endl;
    return nullptr;
  }

  xnn_binary_params v31_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_divide, &v31_params,
                             /*input1_id=*/w41,
                             /*input2_id=*/v30,
                             /*output_id=*/v31,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #30" << std::endl;
    return nullptr;
  }

  xnn_binary_params v32_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v32_params,
                             /*input1_id=*/v26,
                             /*input2_id=*/v31,
                             /*output_id=*/v32,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #31" << std::endl;
    return nullptr;
  }

  uint32_t v32_w58_v33_dq = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v32_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v32_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v32_w58_v33_dq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v32, /*output_id=*/v32_w58_v33_dq,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v32_w58_v33_dq,
      /*filter_id=*/w58,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v33,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #32" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_gelu,
                            /*params=*/nullptr, v33, v34, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #33" << std::endl;
    return nullptr;
  }

  uint32_t v32_w59_v35_dq = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v32_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v32_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v32_w59_v35_dq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v32, /*output_id=*/v32_w59_v35_dq,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v32_w59_v35_dq,
      /*filter_id=*/w59,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v35,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #34" << std::endl;
    return nullptr;
  }

  xnn_binary_params v36_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &v36_params,
                             /*input1_id=*/v34,
                             /*input2_id=*/v35,
                             /*output_id=*/v36,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #35" << std::endl;
    return nullptr;
  }

  uint32_t v36_w60_v37_dq = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v36_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v36_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v36_w60_v37_dq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v36, /*output_id=*/v36_w60_v37_dq,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v36_w60_v37_dq,
      /*filter_id=*/w60,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v37,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #36" << std::endl;
    return nullptr;
  }

  xnn_binary_params v38_params = {-std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_add, &v38_params,
                             /*input1_id=*/v37,
                             /*input2_id=*/v26,
                             /*output_id=*/v38,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #37" << std::endl;
    return nullptr;
  }

  return subgraph;
}  // NOLINT(readability/fn_size)

}  // namespace models
