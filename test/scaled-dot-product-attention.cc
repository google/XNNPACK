// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate.
#include <array>      // For std::array.
#include <cassert>
#include <cmath>
#include <cstddef>  // For size_t.
#include <cstdint>  // For uint32_t.
#include <functional>
#include <limits>   // For std::numeric_limits.
#include <memory>   // For std::unique_ptr.
#include <numeric>  // For std::accumulate.
#include <random>   // For std::uniform_real_distribution.
#include <vector>   // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/node-type.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <class T>
class ScaledDotProductAttentionTestBase : public ::testing::Test {
 protected:
  ScaledDotProductAttentionTestBase() {
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    dim_dist = std::uniform_int_distribution<size_t>(5, 15);
    bernoulli_dist = std::bernoulli_distribution(0.5);
    cap_dist = std::uniform_real_distribution<float>(1.0f, 50.0f);
    auto shape_dist = std::uniform_int_distribution<size_t>(3, XNN_MAX_TENSOR_DIMS);

    // Query is [..., H, T, C].
    query_dims = RandomShape(shape_dist(rng));
    batch_size = 1;
    for (size_t i = 0; i + 3 < query_dims.size(); ++i) {
      batch_size *= query_dims[i];
    }
    query_heads = query_dims[query_dims.size() - 3];
    query_tokens = query_dims[query_dims.size() - 2];
    channels = query_dims[query_dims.size() - 1];

    key_dims = query_dims;
    const bool test_multi_query = bernoulli_dist(rng);
    if (test_multi_query) {
      // Key/Value is [..., U, C].
      key_dims.erase(key_dims.end() - 3);
      // key_dims[key_dims.size() - 2] = 1;
      key_value_heads = 1;
    } else {
      // Key/Value is [..., H, U, C].
      key_value_heads = key_dims[key_dims.size() - 3];
    }

    // Change key_value_tokens dim.
    const size_t key_value_tokens_dim = key_dims.size() - 2;
    key_value_tokens = dim_dist(rng);
    key_dims[key_value_tokens_dim] = key_value_tokens;

    // Value is [..., H, U, D] or [..., U, D].
    value_dims = key_dims;
    value_channels = dim_dist(rng);
    value_dims[value_dims.size() - 1] = value_channels;
    // Mask is [T, U].
    mask_dims = {query_tokens, key_value_tokens};
    // Mask is [C].
    scale_dims = {channels};
    // Output is [..., H, T, D].
    output_dims = query_dims;
    output_dims[output_dims.size() - 1] = value_channels;

    cap_type = bernoulli_dist(rng) ? xnn_attention_logits_cap_type_none : xnn_attention_logits_cap_type_tanh;
    if (cap_type == xnn_attention_logits_cap_type_tanh) {
      cap_params.cap = cap_dist(rng);
    } else {
      cap_params.cap = 0.0f;
    }

    query = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(query_dims));
    key = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(key_dims));
    value = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(value_dims));
    scale = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(scale_dims));
    mask = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(mask_dims));
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(operator_output.size());
  };

  /*
   * Resize internal member vectors to match new shapes.
   * Also update other private variables to reflect new shapes.
   */
  void ResizeTensors(
      std::vector<size_t>& query_dims,
      std::vector<size_t>& key_dims,
      std::vector<size_t>& value_dims,
      std::vector<size_t>& mask_dims,
      std::vector<size_t>& scale_dims,
      std::vector<size_t>& output_dims,
      bool resize_operator_output = true,
      bool multi_query = false){

    // Make sure of our assumptions
    assert (query_dims.size() == 4);
    assert (key_dims.size() == 4 || key_dims.size() == 3);
    assert (value_dims.size() == 4 || value_dims.size() == 3);
    assert (mask_dims.size() == 2);
    assert (scale_dims.size() == 1);
    assert (output_dims.size() == 4);

    batch_size = 1;
    for (size_t i = 0; i + 3 < query_dims.size(); ++i) {
      batch_size *= query_dims[i];
    }
    query_heads = query_dims[query_dims.size() - 3];
    query_tokens = query_dims[query_dims.size() - 2];
    channels = query_dims[query_dims.size() - 1];

    key_value_heads = (multi_query) ? 1 : key_dims[key_dims.size() - 3];

    key_value_tokens = key_dims[key_dims.size() - 2];

    value_channels = value_dims[value_dims.size() - 1];

    query.resize(XNN_EXTRA_BYTES / sizeof(T) + NumElements(query_dims));
    key.resize(XNN_EXTRA_BYTES / sizeof(T) + NumElements(key_dims));
    value.resize(XNN_EXTRA_BYTES / sizeof(T) + NumElements(value_dims));
    scale.resize(XNN_EXTRA_BYTES / sizeof(T) + NumElements(scale_dims));
    mask.resize(XNN_EXTRA_BYTES / sizeof(T) + NumElements(mask_dims));
    subgraph_output.resize(NumElements(output_dims));

    // Resize operator output tensor only if explicitly requested.
    if (resize_operator_output) {
      operator_output.resize(NumElements(output_dims));
    }
  }

  std::vector<size_t> RandomShape(size_t num_dims)
  {
    std::vector<size_t> dims(num_dims);
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  size_t NumElements(std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> cap_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::bernoulli_distribution bernoulli_dist;

  size_t batch_size;
  size_t query_heads;
  size_t query_tokens;
  size_t key_value_heads;
  size_t key_value_tokens;
  size_t channels;
  size_t value_channels;

  xnn_attention_logits_cap_type cap_type;
  xnn_attention_logits_cap_tanh_params cap_params;

  std::vector<size_t> query_dims;
  std::vector<size_t> key_dims;
  std::vector<size_t> value_dims;
  std::vector<size_t> scale_dims;
  std::vector<size_t> mask_dims;
  std::vector<size_t> output_dims;

  std::vector<T> query;
  std::vector<T> key;
  std::vector<T> value;
  std::vector<T> scale;
  std::vector<T> mask;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using ScaledDotProductAttentionTestF16 = ScaledDotProductAttentionTestBase<uint16_t>;
using ScaledDotProductAttentionTestF32 = ScaledDotProductAttentionTestBase<float>;

TEST_F(ScaledDotProductAttentionTestF16, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_product_attention(
      subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_scaled_dot_product_attention);
  EXPECT_EQ(node->compute_type, xnn_compute_type_fp16);
  EXPECT_EQ(node->num_inputs, 5);
  EXPECT_EQ(node->inputs[0], query_id);
  EXPECT_EQ(node->inputs[1], key_id);
  EXPECT_EQ(node->inputs[2], value_id);
  EXPECT_EQ(node->inputs[3], scale_id);
  EXPECT_EQ(node->inputs[4], mask_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->params.scaled_dot_product_attention.cap_type, cap_type);
  EXPECT_EQ(node->params.scaled_dot_product_attention.cap_tanh_params.cap, cap_params.cap);
  EXPECT_EQ(node->flags, 0);
}

TEST_F(ScaledDotProductAttentionTestF32, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_product_attention(
      subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_scaled_dot_product_attention);
  EXPECT_EQ(node->compute_type, xnn_compute_type_fp32);
  EXPECT_EQ(node->num_inputs, 5);
  EXPECT_EQ(node->inputs[0], query_id);
  EXPECT_EQ(node->inputs[1], key_id);
  EXPECT_EQ(node->inputs[2], value_id);
  EXPECT_EQ(node->inputs[3], scale_id);
  EXPECT_EQ(node->inputs[4], mask_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->params.scaled_dot_product_attention.cap_type, cap_type);
  EXPECT_EQ(node->params.scaled_dot_product_attention.cap_tanh_params.cap, cap_params.cap);
  EXPECT_EQ(node->flags, 0);
}

TEST_F(ScaledDotProductAttentionTestF16, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;
  std::generate(query.begin(), query.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(key.begin(), key.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(value.begin(), value.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(scale.begin(), scale.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(mask.begin(), mask.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  // Call operator API.
  const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f16(cap_type, &cap_params, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_scaled_dot_product_attention_nhtc_f16(
                          op, batch_size, query_heads, query_tokens, key_value_heads, key_value_tokens,
                          channels, value_channels,
                          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_scaled_dot_product_attention_nhtc_f16(op, workspace.data(), query.data(), key.data(), value.data(),
                                                    scale.data(), mask.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_product_attention(
      subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 6> external = {
    xnn_external_value{query_id, query.data()},
    xnn_external_value{key_id, key.data()},
    xnn_external_value{value_id, value.data()},
    xnn_external_value{scale_id, scale.data()},
    xnn_external_value{mask_id, mask.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    EXPECT_EQ(subgraph_output[i], operator_output[i]) << i;
  }
}

TEST_F(ScaledDotProductAttentionTestF32, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;
  std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
  std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
  std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
  std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
  std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f32(cap_type, &cap_params, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_scaled_dot_product_attention_nhtc_f32(
                          op, batch_size, query_heads, query_tokens, key_value_heads, key_value_tokens,
                          channels, value_channels,
                          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_scaled_dot_product_attention_nhtc_f32(op, workspace.data(), query.data(), key.data(), value.data(),
                                                    scale.data(), mask.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_product_attention(
      subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 6> external = {
    xnn_external_value{query_id, query.data()},
    xnn_external_value{key_id, key.data()},
    xnn_external_value{value_id, value.data()},
    xnn_external_value{scale_id, scale.data()},
    xnn_external_value{mask_id, mask.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(subgraph_output[i], operator_output[i],
                std::abs(operator_output[i]) * 5 *
                    std::numeric_limits<float>::epsilon())
        << "at offset " << i;
  }
}

TEST_F(ScaledDotProductAttentionTestF32, matches_operator_api_dynamic_shape_no_reallocation)
{
  /*
   * This test makes sure the subgraph is able to match the operator API's with dynamically changing shapes.
   * In this test, we will
   *   1. Prepare a set of input tensors for the operator API.
   *   2. Run the operator API, and save the output tensor.
   *   3. Prepare an equivalent single node subgraph but with larger input shapes than ones used for the operator API in
   * step 1.
   *   4. Run the subgraph, and make sure it works.
   *   5. Reshape the external inputs to match shapes in step 1.
   *   6. Run the subgraph again, and make sure it produces similar output as the operator API, without requiring any
   * reallocation after step 4.
   */

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  size_t N = 2;    // batch size
  size_t H = 2;    // num heads
  size_t Tok = 4;  // tokens
  size_t C = 5;    // channels
  size_t U = Tok;  // tokens (self attention)
  size_t D = 11;   // value channels

  std::vector<size_t> op_query_dims = {N, H, Tok, C};
  std::vector<size_t> op_key_dims = {N, H, U, C};
  std::vector<size_t> op_value_dims = {N, H, U, D};
  std::vector<size_t> op_scale_dims = {C};
  std::vector<size_t> op_mask_dims = {Tok, U};
  std::vector<size_t> op_output_dims = {N, H, Tok, D};

  // Prepare the inputs and outputs for the operator API.
  ResizeTensors(op_query_dims, op_key_dims, op_value_dims, op_mask_dims, op_scale_dims, op_output_dims);

  std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
  std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
  std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
  std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
  std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f32(cap_type, &cap_params, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_scaled_dot_product_attention_nhtc_f32(
                          op, batch_size, query_heads, query_tokens, key_value_heads, key_value_tokens, channels,
                          value_channels, &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_scaled_dot_product_attention_nhtc_f32(
      op, workspace.data(), query.data(), key.data(), value.data(), scale.data(), mask.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Prepare input shapes for the Subgraph API.
  // Make the input/output tensors shapes to be larger than the actual input/output tensors used for the operator API.
  size_t N2 = N + 1;      // batch size
  size_t H2 = H + 1;      // num heads
  size_t Tok2 = Tok + 2;  // tokens
  size_t C2 = C + 3;      // channels
  size_t U2 = Tok2;       // tokens (self attention)
  size_t D2 = D + 2;      // value channels

  std::vector<size_t> subgraph_query_dims = {N2, H2, Tok2, C2};
  std::vector<size_t> subgraph_key_dims = {N2, H2, U2, C2};
  std::vector<size_t> subgraph_value_dims = {N2, H2, U2, D2};
  std::vector<size_t> subgraph_scale_dims = {C2};
  std::vector<size_t> subgraph_mask_dims = {Tok2, U2};
  std::vector<size_t> subgraph_output_dims = {N2, H2, Tok2, D2};

  // Resize the internal member tensors to match the new larger shapes (padding zeros).
  // We don't want to modify the operator output tensor anymore.
  ResizeTensors(
    subgraph_query_dims, subgraph_key_dims, subgraph_value_dims, subgraph_mask_dims, subgraph_scale_dims,
    subgraph_output_dims, /*resize_operator_output=*/false);

  xnn_subgraph_t subgraph = nullptr;

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_query_dims.size(), subgraph_query_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_key_dims.size(), subgraph_key_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_value_dims.size(), subgraph_value_dims.data(), nullptr,
                          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_scale_dims.size(), subgraph_scale_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_mask_dims.size(), subgraph_mask_dims.data(), nullptr,
                          /*external_id=*/4, XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_output_dims.size(), subgraph_output_dims.data(),
                          nullptr, /*external_id=*/5, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_product_attention(
      subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  std::array<xnn_external_value, 6> external = {
    xnn_external_value{query_id, query.data()}, xnn_external_value{key_id, key.data()},
    xnn_external_value{value_id, value.data()}, xnn_external_value{scale_id, scale.data()},
    xnn_external_value{mask_id, mask.data()},   xnn_external_value{output_id, subgraph_output.data()}};

  const struct xnn_node* node = &subgraph->nodes[0];
  // Since this is before setup, expect it to require reallocation
  ASSERT_EQ(
    xnn_status_reallocation_required,
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, nullptr /* thradpool*/));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Reshape the external tensors to the subgraph with the same shape as the ones used in the
  // operator API, which are smaller than the ones supplied at subgraph creation time.
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, query_id, op_query_dims.size(), op_query_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, key_id, op_key_dims.size(), op_key_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, value_id, op_value_dims.size(), op_value_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, scale_id, op_scale_dims.size(), op_scale_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, mask_id, op_mask_dims.size(), op_mask_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, output_id, op_output_dims.size(), op_output_dims.data()));

  // Resize again to remove those extra elements we added earlier
  ResizeTensors(
    op_query_dims, op_key_dims, op_value_dims, op_mask_dims, op_scale_dims, op_output_dims,
    /*resize_output_tensor*/ false);

  // No reallocation since we should require less memory
  ASSERT_EQ(
    xnn_status_success,
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, nullptr /* thradpool*/));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check output shape.
  size_t observed_output_num_dims = 0;
  std::vector<size_t> observed_output_dims(XNN_MAX_TENSOR_DIMS, 0);
  ASSERT_EQ(
    xnn_status_success,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, observed_output_dims.data()));
  ASSERT_EQ(op_output_dims.size(), observed_output_num_dims);
  for (size_t i = 0; i < observed_output_num_dims; i++) {
    ASSERT_EQ(op_output_dims[i], observed_output_dims[i]);
  }

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(subgraph_output[i], operator_output[i],
                std::abs(operator_output[i]) * 5 *
                    std::numeric_limits<float>::epsilon())
        << "at offset " << i;
  }
}

TEST_F(ScaledDotProductAttentionTestF32, matches_operator_api_dynamic_shape_requires_reallocation)
{
  /*
   * This test makes sure the subgraph is able to match the operator API's with dynamically changing shapes.
   * In this test, we will
   *   1. Prepare a set of input tensors for the operator API.
   *   2. Run the operator API, and save the output tensor.
   *   3. Prepare an equivalent single node subgraph but with smaller shapes than the ones used for the operator API in
   * step 1.
   *   4. Run the subgraph, and make sure it works.
   *   5. Reshape the external inputs to match shapes in step 1.
   *   6. Run the subgraph again, and make sure it produces similar output as the operator API, with requiring memory
   * reallocation.
   */

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  size_t N = 2;    // batch size
  size_t H = 2;    // num heads
  size_t Tok = 4;  // tokens
  size_t C = 5;    // channels
  size_t U = Tok;  // tokens (self attention)
  size_t D = 11;   // value channels

  std::vector<size_t> op_query_dims = {N, H, Tok, C};
  std::vector<size_t> op_key_dims = {N, H, U, C};
  std::vector<size_t> op_value_dims = {N, H, U, D};
  std::vector<size_t> op_scale_dims = {C};
  std::vector<size_t> op_mask_dims = {Tok, U};
  std::vector<size_t> op_output_dims = {N, H, Tok, D};

  // Prepare the inputs and outputs for both operator and subgraph
  ResizeTensors(op_query_dims, op_key_dims, op_value_dims, op_mask_dims, op_scale_dims, op_output_dims);

  std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
  std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
  std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
  std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
  std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f32(cap_type, &cap_params, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_scaled_dot_product_attention_nhtc_f32(
                          op, batch_size, query_heads, query_tokens, key_value_heads, key_value_tokens, channels,
                          value_channels, &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_scaled_dot_product_attention_nhtc_f32(
      op, workspace.data(), query.data(), key.data(), value.data(), scale.data(), mask.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Prepare for Subgraph API
  // Input and Output external tensors are smaller than the actual input/output tensors used
  size_t N2 = N - 1;      // batch size
  size_t H2 = H - 1;      // num heads
  size_t Tok2 = Tok - 2;  // tokens
  size_t C2 = C - 3;      // channels
  size_t U2 = Tok2;       // tokens (self attention)
  size_t D2 = D - 2;      // value channels

  std::vector<size_t> subgraph_query_dims = {N2, H2, Tok2, C2};
  std::vector<size_t> subgraph_key_dims = {N2, H2, U2, C2};
  std::vector<size_t> subgraph_value_dims = {N2, H2, U2, D2};
  std::vector<size_t> subgraph_scale_dims = {C2};
  std::vector<size_t> subgraph_mask_dims = {Tok2, U2};
  std::vector<size_t> subgraph_output_dims = {N2, H2, Tok2, D2};

  xnn_subgraph_t subgraph = nullptr;

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_query_dims.size(), subgraph_query_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_key_dims.size(), subgraph_key_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_value_dims.size(), subgraph_value_dims.data(), nullptr,
                          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_scale_dims.size(), subgraph_scale_dims.data(), nullptr,
                          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_mask_dims.size(), subgraph_mask_dims.data(), nullptr,
                          /*external_id=*/4, XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, subgraph_output_dims.size(), subgraph_output_dims.data(),
                          nullptr, /*external_id=*/5, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_product_attention(
      subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  std::array<xnn_external_value, 6> external = {
    xnn_external_value{query_id, query.data()}, xnn_external_value{key_id, key.data()},
    xnn_external_value{value_id, value.data()}, xnn_external_value{scale_id, scale.data()},
    xnn_external_value{mask_id, mask.data()},   xnn_external_value{output_id, subgraph_output.data()}};

  const struct xnn_node* node = &subgraph->nodes[0];
  // Since this is before setup, expect it to require reallocation
  ASSERT_EQ(
    xnn_status_reallocation_required,
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, nullptr /* thradpool*/));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Reshape the external tensors to the subgraph with the same shape as the ones used in the
  // operator API, which are larger than the ones supplied at subgraph creation time.
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, query_id, op_query_dims.size(), op_query_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, key_id, op_key_dims.size(), op_key_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, value_id, op_value_dims.size(), op_value_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, scale_id, op_scale_dims.size(), op_scale_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, mask_id, op_mask_dims.size(), op_mask_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, output_id, op_output_dims.size(), op_output_dims.data()));

  // We will need more memory to run with the larger shape.
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check output shape.
  size_t observed_output_num_dims = 0;
  std::vector<size_t> observed_output_dims(XNN_MAX_TENSOR_DIMS, 0);
  ASSERT_EQ(
    xnn_status_success,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, observed_output_dims.data()));
  ASSERT_EQ(op_output_dims.size(), observed_output_num_dims);
  for (size_t i = 0; i < observed_output_num_dims; i++) {
    ASSERT_EQ(op_output_dims[i], observed_output_dims[i]);
  }

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(subgraph_output[i], operator_output[i],
                std::abs(operator_output[i]) * 5 *
                    std::numeric_limits<float>::epsilon())
        << "at offset " << i;
  }
}

namespace {
void DefineScaledDotProductAttentionSubgraph(
  xnn_status* status_out,
  xnn_attention_logits_cap_type cap_type,
  xnn_attention_logits_cap_tanh_params cap_params,
  std::vector<size_t> query_dims,
  std::vector<size_t> key_dims,
  std::vector<size_t> value_dims,
  std::vector<size_t> scale_dims,
  std::vector<size_t> mask_dims,
  std::vector<size_t> output_dims,
  uint32_t batch_matrix_multiply_flags = 0)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  *status_out = xnn_define_scaled_dot_product_attention(
    subgraph, cap_type, &cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0);
}
}  // namespace

TEST(ScaledDotProductAttentionTest, batch_dims_omitted_is_ok) {
  std::vector<size_t> query_dims = {2, 3, 5};
  std::vector<size_t> key_dims = {2, 7, 5};
  std::vector<size_t> value_dims = {2, 7, 9};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {2, 3, 9};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params {};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_success, status);
}

TEST(ScaledDotProductAttentionTest, cap_tanh_cap_value_is_finite) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_tanh;
  xnn_attention_logits_cap_tanh_params cap_params { std::numeric_limits<float>::infinity() };
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, cap_tanh_cap_value_is_gt_zero) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_tanh;
  xnn_attention_logits_cap_tanh_params cap_params { 0.0f };
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

  TEST(ScaledDotProductAttentionTest, query_num_dim_lt_3) {
    std::vector<size_t> query_dims = {3, 5};
    std::vector<size_t> key_dims = {3, 5};
    std::vector<size_t> value_dims = {3, 5};
    std::vector<size_t> scale_dims = {5};
    std::vector<size_t> mask_dims = {3, 3};
    std::vector<size_t> output_dims = {1, 2, 3, 5};
    xnn_status status = xnn_status_success;
    xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
    xnn_attention_logits_cap_tanh_params cap_params{};
    DefineScaledDotProductAttentionSubgraph(
      &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
    EXPECT_EQ(xnn_status_invalid_parameter, status);
  }

  TEST(ScaledDotProductAttentionTest, key_num_dim_lt_2) {
    std::vector<size_t> query_dims = {2, 3, 5};
    std::vector<size_t> key_dims = {5};
    std::vector<size_t> value_dims = {3, 5};
    std::vector<size_t> scale_dims = {5};
    std::vector<size_t> mask_dims = {3, 3};
    std::vector<size_t> output_dims = {1, 2, 3, 5};
    xnn_status status = xnn_status_success;
    xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
    xnn_attention_logits_cap_tanh_params cap_params{};
    DefineScaledDotProductAttentionSubgraph(
      &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
    EXPECT_EQ(xnn_status_invalid_parameter, status);
  }

  TEST(ScaledDotProductAttentionTest, key_num_dim_eq_query_or_1_less) {
    std::vector<size_t> query_dims = {1, 2, 3, 5};
    std::vector<size_t> key_dims = {1, 2, 3, 5, 6};
    std::vector<size_t> value_dims = {1, 2, 3, 5};
    std::vector<size_t> scale_dims = {5};
    std::vector<size_t> mask_dims = {3, 3};
    std::vector<size_t> output_dims = {1, 2, 3, 5};
    xnn_status status = xnn_status_success;
    xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
    xnn_attention_logits_cap_tanh_params cap_params{};
    DefineScaledDotProductAttentionSubgraph(
      &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
    EXPECT_EQ(xnn_status_invalid_parameter, status);
  }

  TEST(ScaledDotProductAttentionTest, value_num_dim_lt_2) {
  std::vector<size_t> query_dims = {2, 3, 5};
  std::vector<size_t> key_dims = {3, 5};
  std::vector<size_t> value_dims = {3};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, query_channels_eq_key_channels) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 7};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, query_heads_eq_key_heads) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 7, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, query_heads_eq_value_heads) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 7, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, key_num_dims_ne_query_num_dims) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {7, 1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, key_batch_size_ne_query_batch_size) {
  std::vector<size_t> query_dims = {1, 2, 2, 3, 5};
  std::vector<size_t> key_dims = {2, 1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, key_batch_size_ne_query_batch_size_multi_query) {
  std::vector<size_t> query_dims = {1, 7, 2, 3, 5};
  std::vector<size_t> key_dims = {7, 1, 3, 5};
  std::vector<size_t> value_dims = {1, 7, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, value_num_dims_ne_key_num_dims) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {7, 1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, value_batch_size_ne_query_batch_size) {
  std::vector<size_t> query_dims = {1, 2, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 2, 3, 5};
  std::vector<size_t> value_dims = {2, 1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, value_batch_size_ne_query_batch_size_multi_query) {
  std::vector<size_t> query_dims = {1, 7, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 7, 3, 5};
  std::vector<size_t> value_dims = {7, 1, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, key_tokens_ne_value_tokens) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, scale_num_dims_must_be_1) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5, 7};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, scale_channels_eq_query_channels) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {7};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, mask_num_dims_must_be_2) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, mask_query_tokens_eq_query_tokens) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {9, 7};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, mask_key_value_tokens_eq_key_value_tokens) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 9};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, output_dims_ge_4) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, output_dims_eq_query_dims) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {1, 2, 3, 7};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, output_batch_size_eq_query) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {2, 1, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, output_heads_eq_query) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {1, 7, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, output_tokens_eq_query) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {1, 2, 7, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotProductAttentionTest, output_channels_eq_value) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 11};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {1, 2, 3, 7};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotProductAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}
