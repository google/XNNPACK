// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#include <type_traits>

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>

namespace xnnpack {

enum class TensorType {
  kDense,
  kSparse,
};

struct Padding {
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
  uint32_t left;
};

struct HeightWidth {
  uint32_t height;
  uint32_t width;
};

using Kernel = HeightWidth;
using Subsampling = HeightWidth;
using Dilation = HeightWidth;
using Upsampling = HeightWidth;
using Adjustment = HeightWidth;

struct ConvolutionParams {
  Padding padding;
  Kernel kernel;
  Subsampling subsampling;
  Dilation dilation;
  uint32_t groups;
  uint32_t group_input_channels;
  uint32_t group_output_channels;
};

struct DeconvolutionParams {
  Padding padding;
  Adjustment adjustment;
  Kernel kernel;
  Upsampling upsampling;
  Dilation dilation;
  uint32_t groups;
  uint32_t group_input_channels;
  uint32_t group_output_channels;
};

struct DepthwiseConvolutionParams {
  Padding padding;
  Kernel kernel;
  Subsampling subsampling;
  Dilation dilation;
  uint32_t depth_multiplier;
  uint32_t input_channels;
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

  inline SubgraphTester& AddInternalDynamicTensorF32(const std::vector<size_t>& dims,
                                   uint32_t* id_out,
                                   uint32_t flags = 0) {
    const xnn_status status =
        xnn_define_tensor_value(subgraph_.get(), xnn_datatype_fp32, dims.size(),
                                dims.data(), nullptr, XNN_INVALID_VALUE_ID, flags, id_out);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddDynamicTensorF32(const std::vector<size_t>& dims,
                                   uint32_t external_id,
                                   uint32_t flags = 0) {
    uint32_t id_out = 0;
    const xnn_status status =
        xnn_define_tensor_value(subgraph_.get(), xnn_datatype_fp32, dims.size(),
                                dims.data(), nullptr, external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);

    return *this;
  }

  inline SubgraphTester& AddDynamicTensorQS8(
    int32_t zero_point,
    float scale,
    const std::vector<size_t>& dims,
    uint32_t external_id,
    uint32_t flags = 0)
  {
    uint32_t id_out = 0;
    const xnn_status status =
        xnn_define_quantized_tensor_value(subgraph_.get(), xnn_datatype_qint8,
                                          zero_point, scale,
                                          dims.size(),
                                dims.data(), nullptr, external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);

    return *this;
  }

  inline SubgraphTester& AddDynamicallyQuantizedTensor(
    const std::vector<size_t>& dims,
    uint32_t external_id,
    uint32_t flags = 0)
  {
    uint32_t id_out = 0;
    const xnn_status status =
        xnn_define_dynamically_quantized_tensor_value(subgraph_.get(), xnn_datatype_qdint8,
                                          dims.size(), 1,
                                dims.data(), external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);

    return *this;
  }

  inline SubgraphTester& AddStaticTensorQS8(const std::vector<size_t>& dims,
                                            TensorType tensor_type,
                                            const float* scale,
                                            uint32_t external_id,
                                            uint32_t flags = 0,
                                            int8_t* data = nullptr) {
    if (data == nullptr) {
      const size_t num_elements = NumElements(dims);
      static_data_.emplace_back(num_elements * sizeof(int8_t));
      data = reinterpret_cast<int8_t*>(static_data_.back().data());

      if (tensor_type == TensorType::kDense) {
        std::generate(data, data + num_elements, [&]() { return w8dist(rng_); });
      } else {
        // Create tensor with 90% sparsity in two steps:
        // 1. Generate non-zero elements in the beginning of the vector
        // 2. Randomize positions of non-zero elements
        const size_t num_nonzero_elements = num_elements / 10;
        std::generate(data, data + num_nonzero_elements, [&]() { return f32dist(rng_); });
        std::shuffle(data, data + num_elements, rng_);
      }
    }

    uint32_t id_out;
    const xnn_status status =
        xnn_define_channelwise_quantized_tensor_value(subgraph_.get(), xnn_datatype_qcint8, scale, dims.size(), 0,
                                dims.data(), data, external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);
    return *this;
  }


  inline SubgraphTester& AddStaticTensorF32(const std::vector<size_t>& dims,
                                            TensorType tensor_type,
                                            uint32_t external_id,
                                            uint32_t flags = 0,
                                            float* data = nullptr) {
    if (data == nullptr) {
      const size_t num_elements = NumElements(dims);
      static_data_.emplace_back(num_elements * sizeof(float));
      data = reinterpret_cast<float*>(static_data_.back().data());

      if (tensor_type == TensorType::kDense) {
        std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
      } else {
        // Create tensor with 90% sparsity in two steps:
        // 1. Generate non-zero elements in the beginning of the vector
        // 2. Randomize positions of non-zero elements
        const size_t num_nonzero_elements = num_elements / 10;
        std::generate(data, data + num_nonzero_elements, [&]() { return f32dist(rng_); });
        std::shuffle(data, data + num_elements, rng_);
      }
    }

    uint32_t id_out;
    const xnn_status status =
        xnn_define_tensor_value(subgraph_.get(), xnn_datatype_fp32, dims.size(),
                                dims.data(), data, external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);
    return *this;
  }

  inline SubgraphTester& AddInputTensorF32(const std::vector<size_t>& dims, uint32_t external_id) {
    AddDynamicTensorF32(dims, external_id, XNN_VALUE_FLAG_EXTERNAL_INPUT);
    size_t num_elements = NumElements(dims);
    auto input = std::vector<char>(num_elements * sizeof(float) + XNN_EXTRA_BYTES * sizeof(char));
    float* data = reinterpret_cast<float*>(input.data());
    std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
    auto it = external_tensors_.insert({external_id, input});
    EXPECT_TRUE(it.second);
    return *this;
  }

  inline SubgraphTester& AddInputTensorQS8(int32_t zero_point, float scale, const std::vector<size_t>& dims, uint32_t external_id) {
    AddDynamicTensorQS8(zero_point, scale, dims, external_id, XNN_VALUE_FLAG_EXTERNAL_INPUT);
    size_t num_elements = NumElements(dims);
    auto input = std::vector<char>(num_elements * sizeof(float) + XNN_EXTRA_BYTES * sizeof(char));
    float* data = reinterpret_cast<float*>(input.data());
    std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
    auto it = external_tensors_.insert({external_id, input});
    EXPECT_TRUE(it.second);
    return *this;
  }

  inline SubgraphTester& AddOutputTensorF32(const std::vector<size_t>& dims, uint32_t external_id) {
    output_id_ = external_id;
    AddDynamicTensorF32(dims, external_id, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
    size_t num_elements = NumElements(dims);
    auto output = std::vector<char>(num_elements * sizeof(float));
    float* data = reinterpret_cast<float*>(output.data());
    std::fill(data, data + num_elements, std::nanf(""));
    auto it = external_tensors_.insert({external_id, output});
    EXPECT_TRUE(it.second);
    return *this;
  }

  inline SubgraphTester& AddConcatenate2(size_t axis, uint32_t input1_id, uint32_t input2_id, uint32_t output_id) {
    const xnn_status status = xnn_define_concatenate2(
        subgraph_.get(), axis, input1_id, input2_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);
    return *this;
  }

  inline SubgraphTester& AddConstantPad(
      const size_t *pre_paddings, const size_t *post_paddings,
      float padding_value, uint32_t input_id, uint32_t output_id) {
    const xnn_status status = xnn_define_static_constant_pad(
        subgraph_.get(), pre_paddings, post_paddings, padding_value, input_id,
        output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);
    return *this;
  }

  inline SubgraphTester& AddConstantPad(
    const std::vector<size_t>& pre_paddings,
    const std::vector<size_t>& post_paddings,
    float padding_value,
    uint32_t input_id,
    uint32_t output_id)
  {
    const xnn_status status = xnn_define_static_constant_pad(
        subgraph_.get(), pre_paddings.data(), post_paddings.data(), padding_value, input_id,
        output_id, /*flags=*/0);
    EXPECT_EQ(status, xnn_status_success);
    return *this;
  }

  inline SubgraphTester& AddConvert(uint32_t input_id, uint32_t output_id) {
    const xnn_status status = xnn_define_convert(
        subgraph_.get(), input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);
    return *this;
  }

  inline SubgraphTester& AddConvolution2D(
      ConvolutionParams params,
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
      uint32_t output_id) {
    const xnn_status status = xnn_define_convolution_2d(
        subgraph_.get(), params.padding.top, params.padding.right,
        params.padding.bottom, params.padding.left, params.kernel.height, params.kernel.width,
        params.subsampling.height, params.subsampling.width, params.dilation.height, params.dilation.width,
        params.groups, params.group_input_channels, params.group_output_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddCopy(uint32_t input_id, uint32_t output_id) {
    const xnn_status status = xnn_define_copy(
        subgraph_.get(), input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);
    return *this;
  }

  inline SubgraphTester& AddDepthwiseConvolution2D(
      DepthwiseConvolutionParams params,
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id, uint32_t output_id) {
    const xnn_status status = xnn_define_depthwise_convolution_2d(
        subgraph_.get(), params.padding.top, params.padding.right,
        params.padding.bottom, params.padding.left, params.kernel.height, params.kernel.width,
        params.subsampling.height, params.subsampling.width, params.dilation.height, params.dilation.width,
        params.depth_multiplier, params.input_channels,
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

  inline SubgraphTester& AddAveragePooling2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
      uint32_t stride_width, uint32_t input_id, uint32_t output_id) {
    const xnn_status status = xnn_define_average_pooling_2d(
        subgraph_.get(), input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, pooling_height, pooling_width,
        stride_height, stride_width, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, output_id,
        0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddClamp(float output_min, float output_max, uint32_t input_id, uint32_t output_id) {
    const xnn_status status =
        xnn_define_clamp(subgraph_.get(), output_min, output_max, input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddDeconvolution2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t adjustment_height, uint32_t adjustment_width,
      uint32_t kernel_height, uint32_t kernel_width,
      uint32_t upsampling_height, uint32_t upsampling_width,
      uint32_t dilation_height, uint32_t dilation_width, uint32_t groups,
      size_t group_input_channels, size_t group_output_channels,
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
      uint32_t output_id) {
    const xnn_status status = xnn_define_deconvolution_2d(
        subgraph_.get(), input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, adjustment_height,
        adjustment_width, kernel_height, kernel_width, upsampling_height,
        upsampling_width, dilation_height, dilation_width, groups,
        group_input_channels, group_output_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddDeconvolution2D(
      DeconvolutionParams params,
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
      uint32_t output_id) {
    const xnn_status status = xnn_define_deconvolution_2d(
        subgraph_.get(), params.padding.top, params.padding.right,
        params.padding.bottom, params.padding.left, params.adjustment.height,
        params.adjustment.width, params.kernel.height, params.kernel.width, params.upsampling.height,
        params.upsampling.width, params.dilation.height, params.dilation.width, params.groups,
        params.group_input_channels, params.group_output_channels,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddDivide(uint32_t input_id1, uint32_t input_id2, uint32_t output_id) {
    const xnn_status status =
        xnn_define_divide(subgraph_.get(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), input_id1,
                        input_id2, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddEvenSplit2(size_t split_dim, uint32_t input_id, uint32_t output1_id, uint32_t output2_id) {
    const xnn_status status = xnn_define_even_split2(
        subgraph_.get(), split_dim, input_id, output1_id, output2_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);
    return *this;
  }

  inline SubgraphTester& AddFullyConnected(
      uint32_t input_id, uint32_t filter_id, uint32_t bias_id, uint32_t output_id, uint32_t flags = 0) {
    const xnn_status status = xnn_define_fully_connected(
        subgraph_.get(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
        output_id, flags);
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

  inline SubgraphTester& AddEvenSplit3(uint32_t input_id, uint32_t output_id0, uint32_t output_id1, uint32_t output_id2) {
    const xnn_status status = xnn_define_even_split3(
        subgraph_.get(), 0, input_id, output_id0, output_id1, output_id2, 0 /*flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddHardSwish(uint32_t input_id, uint32_t output_id) {
    const xnn_status status =
        xnn_define_hardswish(subgraph_.get(), input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddLeakyRelu(float negative_slope, uint32_t input_id, uint32_t output_id) {
    const xnn_status status =
        xnn_define_leaky_relu(subgraph_.get(), negative_slope, input_id, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddMaxPooling2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
      uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width, uint32_t input_id, uint32_t output_id) {
    const xnn_status status = xnn_define_max_pooling_2d(
        subgraph_.get(), input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, pooling_height, pooling_width,
        stride_height, stride_width, dilation_height, dilation_width, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), input_id, output_id,
        0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddMultiply(uint32_t input_id1, uint32_t input_id2, uint32_t output_id) {
    const xnn_status status =
        xnn_define_multiply2(subgraph_.get(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), input_id1,
                        input_id2, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddPrelu(uint32_t input_id, uint32_t slope_id, uint32_t output_id) {
    const xnn_status status = xnn_define_prelu(subgraph_.get(), input_id, slope_id, output_id, /*flags=*/0);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& AddSubtract(uint32_t input_id1, uint32_t input_id2, uint32_t output_id) {
    const xnn_status status =
        xnn_define_subtract(subgraph_.get(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), input_id1,
                        input_id2, output_id, 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& Optimize() {
    const xnn_status status = xnn_subgraph_optimize(subgraph_.get(), 0 /* flags */);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& InferShape() {
    const xnn_status status = xnn_subgraph_infer_shape(subgraph_.get(), /*flags=*/0);
    EXPECT_EQ(status, xnn_status_success);

    return *this;
  }

  inline SubgraphTester& RewriteForNchw() {
    xnn_subgraph_rewrite_for_nchw(subgraph_.get());

    return *this;
  }

  inline SubgraphTester& RewriteForFp16() {
    EXPECT_TRUE(xnn_subgraph_rewrite_for_fp16(subgraph_.get()));

    return *this;
  }

  inline SubgraphTester& RewriteForFp16WithFailure() {
    EXPECT_FALSE(xnn_subgraph_rewrite_for_fp16(subgraph_.get()));

    return *this;
  }

  inline xnn_layout_type GetLayout(uint32_t value_id) const {
    return subgraph_->values[value_id].layout;
  }

  inline const xnn_value* const Value(uint32_t value_id) const {
    return &subgraph_->values[value_id];
  }

  inline const xnn_node* const Node(uint32_t node_id) const {
    return &subgraph_->nodes[node_id];
  }

  inline size_t NumNodes() const {
    return subgraph_->num_nodes;
  }

  inline size_t NumValues() const {
    return subgraph_->num_values;
  }

  inline xnn_subgraph* Subgraph() const {
    return subgraph_.get();
  }

  float* GetExternalTensorDataF32(uint32_t external_id) {
    return reinterpret_cast<float*>(external_tensors_[external_id].data());
  }

  static inline size_t NumElements(const std::vector<size_t>& dims) {
    return std::accumulate(std::begin(dims), std::end(dims), size_t(1), std::multiplies<size_t>());
  }

 protected:
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_{nullptr, xnn_delete_subgraph};
  std::unordered_map<uint32_t, std::vector<char>> external_tensors_;
  uint32_t output_id_;
  std::mt19937 rng_;
  std::uniform_real_distribution<float> f32dist = std::uniform_real_distribution<float>(-1.0f, +1.0f);
  std::uniform_int_distribution<int32_t> w8dist = std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

 private:
  std::vector<std::vector<char>> static_data_;
};

}  // namespace xnnpack
