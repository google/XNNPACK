// Copyright 2024 Google LLC
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

#include "models.h"
#include "xnnpack.h"

namespace models {

xnn_subgraph_t QD8Attention(size_t batch_size, size_t seq_len,
                            size_t embedding_dim, size_t num_heads,
                            size_t head_dim, QD8AttentionWeights &weights) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  // Scales must be positive.
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, +1.0f),
                          std::ref(rng));

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

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr, input_id, quantized_input_id,
                              /*XNN_FLAG_MAYBE_PACK_FOR_GEMM=*/0x00000080);
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
                std::ref(f32rng));
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
                std::ref(f32rng));
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
                std::ref(f32rng));
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
                std::ref(f32rng));
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

  status =
      xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr, outcome_reshaped_id, quantized_outcome_id,
                         /*XNN_FLAG_MAYBE_PACK_FOR_GEMM=*/0x00000080);
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
