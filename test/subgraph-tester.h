// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <unordered_map>
#include <numeric>
#include <random>
#include <vector>
#include <type_traits>

#include <gtest/gtest.h>

enum xnn_tensor_type {
  kStaticDense,
  kStaticSparse,
  kDynamic,
};

template <typename T>
class SubgraphTester {
 public:
  explicit SubgraphTester(uint32_t external_value_ids) {
    xnn_status status = xnn_initialize(nullptr);
    EXPECT_EQ(status, xnn_status_success);

    xnn_subgraph_t subgraph_ptr = nullptr;
    status = xnn_create_subgraph(external_value_ids, 0 /* flags */, &subgraph_ptr);
    EXPECT_EQ(status, xnn_status_success);
    subgraph_.reset(subgraph_ptr);

    std::random_device random_device;
    rng_ = std::mt19937(random_device());
  }

  inline SubgraphTester& AddTensor(const std::vector<size_t>& dims,
                                   xnn_tensor_type tensor_type,
                                   uint32_t external_id,
                                   uint32_t flags = 0) {
    void* data = nullptr;
    if (tensor_type == kStaticDense || tensor_type == kStaticSparse) {
      const size_t num_elements = NumElements(dims);
      static_data_.emplace_back(num_elements);
      std::vector<T>& weights = static_data_.back();
      if (tensor_type == kStaticDense) {
        InitializeWeights(weights);
      } else {
        // Create tensor with 90% sparsity in two steps:
        // 1. Generate non-zero elements in the beginning of the vector
        // 2. Randomize positions of non-zero elements
        const size_t num_nonzero_elements = num_elements / 10;
        std::generate(weights.begin(), weights.begin() + num_nonzero_elements, [&]() { return f32dist(rng_); });
        std::shuffle(weights.begin(), weights.end(), rng_);
      }
      data = weights.data();
    }
    uint32_t id_out = 0;
    const xnn_status status =
        xnn_define_tensor_value(subgraph_.get(), xnn_datatype_fp32, dims.size(),
                                dims.data(), data, external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);

    return *this;
  }

  inline SubgraphTester& AddInputTensor(const std::vector<size_t>& dims, uint32_t external_id) {
    AddTensor(dims, kDynamic, external_id, XNN_VALUE_FLAG_EXTERNAL_INPUT);
    auto input = std::vector<T>(NumElements(dims) + XNN_EXTRA_BYTES / sizeof(T));
    InitializeInput(input);
    auto it = external_tensors_.insert({ external_id, input});
    EXPECT_TRUE(it.second);
    return *this;
  }

  inline SubgraphTester& AddOutputTensor(const std::vector<size_t>& dims, uint32_t external_id) {
    AddTensor(dims, kDynamic, external_id, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
    output_id = external_id;
    auto output = std::vector<T>(NumElements(dims));
    InitializeOutput(output);
    auto it = external_tensors_.insert({external_id, output});
    EXPECT_TRUE(it.second);
    return *this;
  }

  inline SubgraphTester& AddConv(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t kernel_height, uint32_t kernel_width,
      uint32_t subsampling_height, uint32_t subsampling_width,
      uint32_t dilation_height, uint32_t dilation_width, uint32_t groups,
      size_t group_input_channels, size_t group_output_channels,
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
      uint32_t output_id) {
    const xnn_status status = xnn_define_convolution_2d(
        subgraph_.get(), input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, kernel_height, kernel_width,
        subsampling_height, subsampling_width, dilation_height, dilation_width,
        groups, group_input_channels, group_output_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddDepthwiseConv(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t kernel_height, uint32_t kernel_width,
      uint32_t subsampling_height, uint32_t subsampling_width,
      uint32_t dilation_height, uint32_t dilation_width,
      uint32_t depth_multiplier, size_t input_channels, uint32_t input_id,
      uint32_t filter_id, uint32_t bias_id, uint32_t output_id) {
    const xnn_status status = xnn_define_depthwise_convolution_2d(
        subgraph_.get(), input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, kernel_height, kernel_width,
        subsampling_height, subsampling_width, dilation_height, dilation_width,
        depth_multiplier, input_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddAddition(uint32_t input_id1, uint32_t input_id2, uint32_t output_id) {
    const xnn_status status =
        xnn_define_add2(subgraph_.get(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), input_id1,
                        input_id2, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddClamp(float output_min, float output_max, uint32_t input_id, uint32_t output_id) {
    const xnn_status status =
        xnn_define_clamp(subgraph_.get(), output_min, output_max, input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddGlobalAveragePooling(uint32_t input_id, uint32_t output_id) {
    const xnn_status status = xnn_define_global_average_pooling_2d(
        subgraph_.get(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& Optimize() {
    const xnn_status status = xnn_subgraph_optimize(subgraph_.get(), 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& Rewrite() {
    xnn_subgraph_rewrite_for_nchw(subgraph_.get());

    return *this;
  }

  inline xnn_layout_type GetLayout(uint32_t value_id) const {
    return subgraph_->values[value_id].layout;
  }

  inline const xnn_node* const Node(uint32_t node_index) const {
    return &subgraph_->nodes[node_index];
  }

 protected:
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_{nullptr, xnn_delete_subgraph};
  std::unordered_map<uint32_t, std::vector<T>> external_tensors_;
  uint32_t output_id;

 private:
  static inline size_t NumElements(const std::vector<size_t>& dims) {
    return std::accumulate(std::begin(dims), std::end(dims), size_t(1), std::multiplies<size_t>());
  }

  void InitializeInput(std::vector<float>& input) {
    std::generate(input.begin(), input.end(), [&]() { return f32dist(rng_); });
  }

  void InitializeWeights(std::vector<float> weights) {
    std::generate(weights.begin(), weights.end(), [&]() { return f32dist(rng_); });
  };

  void InitializeOutput(std::vector<float>& output) {
    std::fill(output.begin(), output.end(), std::nanf(""));
  }

  std::vector<std::vector<T>> static_data_;
  std::mt19937 rng_;
  std::uniform_real_distribution<float> f32dist = std::uniform_real_distribution<float>(-1.0f, +1.0f);

};

using SubgraphTesterF32 = SubgraphTester<float>;
