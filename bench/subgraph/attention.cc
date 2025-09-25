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

#include "bench/subgraph/benchmark.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

struct QD8AttentionWeights {
  std::vector<int8_t> query_data;
  std::vector<float> query_scale;
  std::vector<int8_t> key_data;
  std::vector<float> key_scale;
  std::vector<int8_t> value_data;
  std::vector<float> value_scale;
  std::vector<int8_t> post_proj_data;
  std::vector<float> post_proj_scale;
};

xnn_subgraph_t FP32Attention(size_t b, size_t t, size_t h, size_t n, size_t s) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/4, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  xnnpack::ReplicableRandomDevice rng;

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v0_dims = {{b, s, n, t}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v1_dims = {{b, t, n, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v1_dims.size(), v1_dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v2_dims = {{b, s, n, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v2_dims.size(), v2_dims.data(),
      /*data=*/nullptr, 2, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  uint32_t v3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v3_dims = {{b, t, n, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v3_dims.size(), v3_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v3" << std::endl;
    return nullptr;
  }

  uint32_t v4 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v4_dims = {{b, n, t, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v4_dims.size(), v4_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v4" << std::endl;
    return nullptr;
  }

  uint32_t v5 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v5_dims = {{b, n, s, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v5_dims.size(), v5_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v5" << std::endl;
    return nullptr;
  }

  uint32_t v6 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v6_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v6_dims.size(), v6_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v6" << std::endl;
    return nullptr;
  }

  uint32_t v7 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v7_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v7_dims.size(), v7_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v7" << std::endl;
    return nullptr;
  }

  uint32_t v8 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v8_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v8_dims.size(), v8_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v8" << std::endl;
    return nullptr;
  }

  uint32_t v9 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v9_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v9_dims.size(), v9_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v9" << std::endl;
    return nullptr;
  }

  uint32_t v10 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v10_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v10_dims.size(), v10_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v10" << std::endl;
    return nullptr;
  }

  uint32_t v11 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v11_dims = {{b, n, t, t}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v11_dims.size(), v11_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &v11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v11" << std::endl;
    return nullptr;
  }

  uint32_t v12 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v12_dims = {{b, t, n, t}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v12_dims.size(), v12_dims.data(),
      /*data=*/nullptr, 3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v12" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w13_data;
  uint32_t w13 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w13_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w13_dims.size(), w13_dims.data(),
      /*data=*/w13_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w13" << std::endl;
    return nullptr;
  }

  static const std::array<int32_t, 4> w14_data = {
      0x000000,
      0x000002,
      0x000001,
      0x000003,
  };
  uint32_t w14 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w14_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w14_dims.size(), w14_dims.data(),
      /*data=*/w14_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w14" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w15_data;
  uint32_t w15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w15_dims = {{1}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w15_dims.size(), w15_dims.data(),
      /*data=*/w15_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w15" << std::endl;
    return nullptr;
  }

  static const std::array<int32_t, 4> w16_data = {
      0x000000,
      0x000002,
      0x000003,
      0x000001,
  };
  uint32_t w16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w16_dims = {{4}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_int32, w16_dims.size(), w16_dims.data(),
      /*data=*/w16_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w16" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));
  std::generate(w13_data.begin(), w13_data.end(), std::ref(f32rng));
  std::generate(w15_data.begin(), w15_data.end(), std::ref(f32rng));

  xnn_binary_params v3_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status =
      xnn_define_binary(subgraph, xnn_binary_multiply, &v3_params, v1, w13, v3,
                        /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v3_w14_v4 = {
      (size_t)w14_data[0], (size_t)w14_data[1], (size_t)w14_data[2],
      (size_t)w14_data[3]};
  status =
      xnn_define_static_transpose(subgraph,
                                  /*num_dims=*/perm_v3_w14_v4.size(),
                                  /*perm=*/perm_v3_w14_v4.data(), v3, v4, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v2_w14_v5 = {
      (size_t)w14_data[0], (size_t)w14_data[1], (size_t)w14_data[2],
      (size_t)w14_data[3]};
  status =
      xnn_define_static_transpose(subgraph,
                                  /*num_dims=*/perm_v2_w14_v5.size(),
                                  /*perm=*/perm_v2_w14_v5.data(), v2, v5, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph, v4, v5, v6,
                                            XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #3" << std::endl;
    return nullptr;
  }

  xnn_binary_params v7_params = {-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  status =
      xnn_define_binary(subgraph, xnn_binary_multiply, &v7_params, v6, w15, v7,
                        /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_tanh,
                            /*params=*/nullptr, v7, v8, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  status = xnn_define_softmax(subgraph, v8, v9,
                              /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v0_w16_v10 = {
      (size_t)w16_data[0], (size_t)w16_data[1], (size_t)w16_data[2],
      (size_t)w16_data[3]};
  status =
      xnn_define_static_transpose(subgraph,
                                  /*num_dims=*/perm_v0_w16_v10.size(),
                                  /*perm=*/perm_v0_w16_v10.data(), v0, v10, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph, v9, v10, v11,
                                            XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #8" << std::endl;
    return nullptr;
  }

  std::array<size_t, 4> perm_v11_w14_v12 = {
      (size_t)w14_data[0], (size_t)w14_data[1], (size_t)w14_data[2],
      (size_t)w14_data[3]};
  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/perm_v11_w14_v12.size(),
                                       /*perm=*/perm_v11_w14_v12.data(), v11,
                                       v12, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  return subgraph;
}

xnn_subgraph_t QD8Attention(size_t batch_size, size_t seq_len,
                            size_t embedding_dim, size_t num_heads,
                            size_t head_dim, QD8AttentionWeights& weights) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  xnnpack::ReplicableRandomDevice rng;
  // Scales must be positive.
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, +1.0f),
                          std::ref(rng));
  auto i8rng =
      std::bind(std::uniform_int_distribution<int>(-127, 127), std::ref(rng));

  // External inputs and outputs.
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> input_dims = {{batch_size, seq_len, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(),
      /*data=*/nullptr, /*external_id=*/0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create input tensor " << std::endl;
    return nullptr;
  }

  uint32_t quantized_input_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, input_dims.size(),
      /*num_non_batch_dims=*/1, input_dims.data(), XNN_INVALID_VALUE_ID,
      /*flags=*/0, &quantized_input_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized input tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            input_id, quantized_input_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> output_dims = {{batch_size, seq_len, embedding_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(),
      /*data=*/nullptr, /*external_id=*/1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create output tensor " << std::endl;
    return nullptr;
  }

  // Static query, key and value tensors.
  uint32_t value_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> value_dims = {{head_dim, embedding_dim}};
  weights.value_scale.resize(head_dim, 1.f);
  weights.value_data.resize(head_dim * embedding_dim, 1);
  std::generate(weights.value_scale.begin(), weights.value_scale.end(),
                std::ref(f32rng));
  std::generate(weights.value_data.begin(), weights.value_data.end(),
                std::ref(i8rng));
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8, weights.value_scale.data(),
      value_dims.size(), value_dims.size() - 2, value_dims.data(),
      weights.value_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &value_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor value" << std::endl;
    return nullptr;
  }

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> query_dims = {{num_heads * head_dim, embedding_dim}};
  weights.query_scale.resize(num_heads * head_dim, 1.f);
  weights.query_data.resize(head_dim * embedding_dim * num_heads, 1);
  std::generate(weights.query_scale.begin(), weights.query_scale.end(),
                std::ref(f32rng));
  std::generate(weights.query_data.begin(), weights.query_data.end(),
                std::ref(i8rng));
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8, weights.query_scale.data(),
      query_dims.size(), query_dims.size() - 2, query_dims.data(),
      weights.query_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &query_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query" << std::endl;
    return nullptr;
  }

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> key_dims = {{head_dim, embedding_dim}};
  weights.key_scale.resize(head_dim, 1.f);
  weights.key_data.resize(head_dim * embedding_dim, 1);
  std::generate(weights.key_scale.begin(), weights.key_scale.end(),
                std::ref(f32rng));
  std::generate(weights.key_data.begin(), weights.key_data.end(),
                std::ref(i8rng));
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8, weights.key_scale.data(), key_dims.size(),
      key_dims.size() - 2, key_dims.data(), weights.key_data.data(),
      XNN_INVALID_VALUE_ID, /*flags=*/0, &key_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor key" << std::endl;
    return nullptr;
  }

  uint32_t query_proj_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> query_proj_dims = {
      {batch_size, seq_len, num_heads * head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_proj_dims.size(),
      query_proj_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &query_proj_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query proj" << std::endl;
    return nullptr;
  }

  uint32_t key_proj_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> key_proj_dims = {{seq_len, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_proj_dims.size(), key_proj_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &key_proj_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor key proj" << std::endl;
    return nullptr;
  }

  uint32_t value_proj_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> value_proj_dims = {{seq_len, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_proj_dims.size(),
      value_proj_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &value_proj_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor value proj" << std::endl;
    return nullptr;
  }

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  status = xnn_define_fully_connected(
      subgraph, output_min, output_max, quantized_input_id, query_id,
      XNN_INVALID_VALUE_ID, query_proj_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create FC node" << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph, output_min, output_max, quantized_input_id, key_id,
      XNN_INVALID_VALUE_ID, key_proj_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create FC node" << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph, output_min, output_max, quantized_input_id, value_id,
      XNN_INVALID_VALUE_ID, value_proj_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create FC node" << std::endl;
    return nullptr;
  }

  uint32_t query_reshaped_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> query_reshaped_dims = {
      {batch_size, seq_len, num_heads, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_reshaped_dims.size(),
      query_reshaped_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &query_reshaped_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query reshaped" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_reshape(subgraph, query_reshaped_dims.size(),
                                     query_reshaped_dims.data(), query_proj_id,
                                     query_reshaped_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape query_proj" << std::endl;
    return nullptr;
  }

  uint32_t key_reshaped_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> key_reshaped_dims = {
      {batch_size, 1, seq_len, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_reshaped_dims.size(),
      key_reshaped_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &key_reshaped_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor key reshaped" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_reshape(subgraph, key_reshaped_dims.size(),
                                     key_reshaped_dims.data(), key_proj_id,
                                     key_reshaped_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape key_proj" << std::endl;
    return nullptr;
  }

  uint32_t logits_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> logits_dims = {
      {batch_size, seq_len, num_heads, seq_len}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, logits_dims.size(), logits_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &logits_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor logits" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph, query_reshaped_id,
                                            key_reshaped_id, logits_id,
                                            /*flags=*/XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create batch matrix multiply" << std::endl;
    return nullptr;
  }

  uint32_t probs_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> probs_dims = {
      {batch_size, seq_len, num_heads, seq_len}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, probs_dims.size(), probs_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &probs_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor probs" << std::endl;
    return nullptr;
  }

  status = xnn_define_softmax(subgraph, logits_id, probs_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create softmax" << std::endl;
    return nullptr;
  }

  uint32_t outcome_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> outcome_dims = {
      {batch_size, seq_len, num_heads, head_dim}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, outcome_dims.size(), outcome_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &outcome_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor outcome" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph, probs_id, value_proj_id,
                                            outcome_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create batch matrix multiply" << std::endl;
    return nullptr;
  }

  uint32_t outcome_reshaped_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 3> outcome_reshaped_dims = {
      {batch_size, seq_len, num_heads * head_dim}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                   outcome_reshaped_dims.size(),
                                   outcome_reshaped_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &outcome_reshaped_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor outcome reshaped" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_reshape(subgraph, outcome_reshaped_dims.size(),
                                     outcome_reshaped_dims.data(), outcome_id,
                                     outcome_reshaped_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape outcome" << std::endl;
    return nullptr;
  }

  // Static post projection weights.
  uint32_t post_proj_weights_id = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> post_proj_dims = {
      {embedding_dim, num_heads * head_dim}};
  weights.post_proj_scale.resize(embedding_dim, 1.f);
  weights.post_proj_data.resize(head_dim * embedding_dim * num_heads, 1);
  std::generate(weights.post_proj_scale.begin(), weights.post_proj_scale.end(),
                std::ref(f32rng));
  std::generate(weights.post_proj_data.begin(), weights.post_proj_data.end(),
                std::ref(i8rng));
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8, weights.post_proj_scale.data(),
      post_proj_dims.size(), post_proj_dims.size() - 2, post_proj_dims.data(),
      weights.post_proj_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0,
      &post_proj_weights_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor post proj weights" << std::endl;
    return nullptr;
  }

  uint32_t quantized_outcome_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, outcome_reshaped_dims.size(),
      /*num_non_batch_dims=*/1, outcome_reshaped_dims.data(),
      XNN_INVALID_VALUE_ID,
      /*flags=*/0, &quantized_outcome_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create dynamically quantized outcome tensor "
              << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            outcome_reshaped_id, quantized_outcome_id,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph, output_min, output_max, quantized_outcome_id,
      post_proj_weights_id, XNN_INVALID_VALUE_ID, output_id, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create FC node" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models

static void FP32Attention(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP32Attention(FLAGS_batch_size, state.range(0),
                                 state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void FP16Attention(benchmark::State& state) {
  xnnpack::RunBenchmark(
      state,
      [&state]() {
        return models::FP32Attention(FLAGS_batch_size, state.range(0),
                                     state.range(1), state.range(2),
                                     state.range(3));
      },
      XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void QD8Attention(benchmark::State& state) {
  models::QD8AttentionWeights weights;
  xnnpack::RunBenchmark(state, [&state, &weights]() {
    return models::QD8Attention(FLAGS_batch_size, state.range(0),
                                state.range(1), state.range(2), state.range(3),
                                weights);
  });
}

static void AttentionArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"T", "H", "N", "S"});
  b->Args({16, 25, 24, 4});
  b->Args({1536, 128, 12, 18});
  b->Args({1024, 256, 4, 46});
  b->Args({1792, 256, 8, 36});
  b->Args({1536, 256, 6, 22});
  b->Args({2048, 256, 8, 18});
  b->Args({3072, 256, 16, 28});
  b->Args({2304, 256, 8, 26});
  b->Args({2048, 64, 32, 24});
}

BENCHMARK(FP32Attention)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(FP16Attention)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(QD8Attention)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(AttentionArguments);
