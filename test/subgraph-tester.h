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
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

enum xnn_tensor_type {
  kStatic = 0,
  kDynamic = 1,
};

class SubgraphTester {
 public:
  explicit SubgraphTester(uint32_t external_value_ids) {
    status_ = xnn_initialize(nullptr);
    assert(xnn_status_success == status_);

    const uint32_t flags = 0;
    status_ = xnn_create_subgraph(external_value_ids, flags, &subgraph_ptr_);
    assert(xnn_status_success == status_);

    std::random_device random_device;
    rng_ = std::mt19937(random_device());
  }

  ~SubgraphTester() {
    status_ = xnn_delete_subgraph(subgraph_ptr_);
    assert(xnn_status_success == status_);
  }

  inline SubgraphTester& add_tensor(const std::vector<size_t>& dims,
                                    xnn_tensor_type tensor_type,
                                    uint32_t external_id) {
    const uint32_t flags = 0;

    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                            std::ref(rng_));
    void* data = nullptr;
    if (tensor_type == kStatic) {
      auto num_elements = std::accumulate(std::begin(dims), std::end(dims), 1,
                                          std::multiplies<>());
      std::vector<float> weights(num_elements);
      std::generate(weights.begin(), weights.end(), std::ref(f32rng));
      data = weights.data();
    }
    uint32_t id_out = 0;
    status_ =
        xnn_define_tensor_value(subgraph_ptr_, xnn_datatype_fp32, dims.size(),
                                dims.data(), data, external_id, flags, &id_out);
    assert(xnn_status_success == status_);

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
      uint32_t output_id) {
    const uint32_t flags = 0;

    status_ = xnn_define_convolution_2d(
        subgraph_ptr_, input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, kernel_height, kernel_width,
        subsampling_height, subsampling_width, dilation_height, dilation_width,
        groups, group_input_channels, group_output_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, flags);
    assert(xnn_status_success == status_);

    return *this;
  }

  inline SubgraphTester& add_depthwise_conv(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t kernel_height, uint32_t kernel_width,
      uint32_t subsampling_height, uint32_t subsampling_width,
      uint32_t dilation_height, uint32_t dilation_width,
      uint32_t depth_multiplier, size_t input_channels, uint32_t input_id,
      uint32_t filter_id, uint32_t bias_id, uint32_t output_id) {
    const uint32_t flags = 0;

    status_ = xnn_define_depthwise_convolution_2d(
        subgraph_ptr_, input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, kernel_height, kernel_width,
        subsampling_height, subsampling_width, dilation_height, dilation_width,
        depth_multiplier, input_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, flags);
    assert(xnn_status_success == status_);

    return *this;
  }

  inline SubgraphTester& add_addition(uint32_t input_id1, uint32_t input_id2,
                                      uint32_t output_id) {
    const uint32_t flags = 0;

    status_ =
        xnn_define_add2(subgraph_ptr_, -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), input_id1,
                        input_id2, output_id, flags);
    assert(xnn_status_success == status_);

    return *this;
  }

  inline SubgraphTester& add_global_average_pooling(uint32_t input_id,
                                                    uint32_t output_id) {
    const uint32_t flags = 0;

    status_ = xnn_define_global_average_pooling_2d(
        subgraph_ptr_, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, output_id, flags);
    assert(xnn_status_success == status_);

    return *this;
  }

  inline SubgraphTester& optimize() {
    const uint32_t flags = 0;
    status_ = xnn_subgraph_optimize(subgraph_ptr_, flags);
    assert(xnn_status_success == status_);

    return *this;
  }

  inline SubgraphTester& rewrite() {
    xnn_subgraph_rewrite_for_nchw(subgraph_ptr_);

    return *this;
  }

  void CheckLayouts(
      std::map<uint32_t, std::pair<xnn_layout_type, xnn_layout_type>>&
          expected_layouts) const {
    for (auto const& item : expected_layouts) {
      xnn_node* node = &subgraph_ptr_->nodes[item.first];

      for (uint32_t i = 0; i < node->num_inputs; i++) {
        struct xnn_value* value = &subgraph_ptr_->values[node->inputs[i]];
        if (value->data != nullptr) {
          continue;
        }
        ASSERT_EQ(item.second.first, value->layout);
      }
      for (uint32_t i = 0; i < node->num_outputs; i++) {
        struct xnn_value* value = &subgraph_ptr_->values[node->outputs[i]];
        ASSERT_EQ(item.second.second, value->layout);
      }
    }
  }

 private:
  xnn_subgraph_t subgraph_ptr_ = nullptr;
  std::mt19937 rng_;
  xnn_status status_;
};
