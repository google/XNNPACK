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
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

enum xnn_tensor_type {
  kStaticDense,
  kStaticSparse,
  kDynamic,
};

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

  inline SubgraphTester& add_tensor(const std::vector<size_t>& dims,
                                    xnn_tensor_type tensor_type,
                                    uint32_t external_id) {
    void* data = nullptr;
    if (tensor_type == kStaticDense || tensor_type == kStaticSparse) {
      const size_t num_elements = std::accumulate(std::begin(dims), std::end(dims), size_t(1), std::multiplies<size_t>());
      static_data_.emplace_back(num_elements);
      std::vector<float>& weights = static_data_.back();
      auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng_));
      if (tensor_type == kStaticDense) {
        std::generate(weights.begin(), weights.end(), std::ref(f32rng));
      } else {
        // Create tensor with 90% sparsity in two steps:
        // 1. Generate non-zero elements in the beginning of the vector
        // 2. Randomize positions of non-zero elements
        const size_t num_nonzero_elements = num_elements / 10;
        std::generate(weights.begin(), weights.begin() + num_nonzero_elements, std::ref(f32rng));
        std::shuffle(weights.begin(), weights.end(), rng_);
      }
      data = weights.data();
    }
    uint32_t id_out = 0;
    const xnn_status status =
        xnn_define_tensor_value(subgraph_.get(), xnn_datatype_fp32, dims.size(),
                                dims.data(), data, external_id, 0 /* flags */, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);

    return *this;
  }

  inline SubgraphTester& add_conv(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t kernel_height, uint32_t kernel_width,
      uint32_t subsampling_height, uint32_t subsampling_width,
      uint32_t dilation_height, uint32_t dilation_width, uint32_t groups,
      size_t group_input_channels, size_t group_output_channels,
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
      uint32_t output_id)
  {
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

  inline SubgraphTester& add_depthwise_conv(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t kernel_height, uint32_t kernel_width,
      uint32_t subsampling_height, uint32_t subsampling_width,
      uint32_t dilation_height, uint32_t dilation_width,
      uint32_t depth_multiplier, size_t input_channels, uint32_t input_id,
      uint32_t filter_id, uint32_t bias_id, uint32_t output_id)
  {
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

  inline SubgraphTester& add_addition(uint32_t input_id1, uint32_t input_id2, uint32_t output_id)
  {
    const xnn_status status =
        xnn_define_add2(subgraph_.get(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), input_id1,
                        input_id2, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& add_global_average_pooling(uint32_t input_id, uint32_t output_id)
  {
    const xnn_status status = xnn_define_global_average_pooling_2d(
        subgraph_.get(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& optimize() {
    const xnn_status status = xnn_subgraph_optimize(subgraph_.get(), 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& rewrite() {
    xnn_subgraph_rewrite_for_nchw(subgraph_.get());

    return *this;
  }

  inline xnn_layout_type get_layout(uint32_t value_id) const {
    return subgraph_->values[value_id].layout;
  }

 private:
  std::vector<std::vector<float>> static_data_;
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_{nullptr, xnn_delete_subgraph};
  std::mt19937 rng_;
};
