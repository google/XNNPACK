// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test/subgraph/subgraph-tester.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

namespace xnnpack {

SubgraphTester::SubgraphTester(uint32_t external_value_ids) {
  xnn_status status = xnn_initialize(nullptr);
  EXPECT_EQ(status, xnn_status_success);

  xnn_subgraph_t subgraph_ptr = nullptr;
  status = xnn_create_subgraph(external_value_ids, /*flags=*/0, &subgraph_ptr);
  EXPECT_EQ(status, xnn_status_success);
  subgraph_.reset(subgraph_ptr);
}

SubgraphTester& SubgraphTester::AddInternalDynamicTensorF32(
    const std::vector<size_t>& dims, uint32_t* id_out, uint32_t flags) {
  const xnn_status status = xnn_define_tensor_value(
      subgraph_.get(), xnn_datatype_fp32, dims.size(), dims.data(), nullptr,
      XNN_INVALID_VALUE_ID, flags, id_out);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddDynamicTensor(
    const std::vector<size_t>& dims, uint32_t external_id,
    xnn_datatype datatype, xnn_quantization_params quantization,
    uint32_t flags) {
  assert(external_id < subgraph_->external_value_ids);
  uint32_t id_out = 0;
  if (xnn_datatype_is_quantized(datatype)) {
    const xnn_status status = xnn_define_quantized_tensor_value(
        subgraph_.get(), datatype, quantization.zero_point, quantization.scale,
        dims.size(), dims.data(), nullptr, external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
  } else {
    const xnn_status status = xnn_define_tensor_value(
        subgraph_.get(), datatype, dims.size(), dims.data(), nullptr,
        external_id, flags, &id_out);
    EXPECT_EQ(status, xnn_status_success);
  }
  EXPECT_EQ(id_out, external_id);

  return *this;
}

SubgraphTester& SubgraphTester::AddDynamicTensor(
    const std::vector<size_t>& dims, uint32_t external_id,
    xnn_datatype datatype, uint32_t flags) {
  assert(!xnn_datatype_is_quantized(datatype));
  assert(external_id < subgraph_->external_value_ids);
  return AddDynamicTensor(dims, external_id, datatype, {}, flags);
}

std::vector<size_t> SubgraphTester::GetExternalTensorShape(
    uint32_t external_id) {
  assert(external_id < subgraph_->external_value_ids);
  std::vector<size_t> shape(XNN_MAX_TENSOR_DIMS);
  size_t rank = 0;
  const xnn_status status = xnn_get_external_value_shape(
      runtime_.get(), external_id, &rank, shape.data());
  EXPECT_EQ(status, xnn_status_success);
  shape.resize(rank);
  return shape;
}

SubgraphTester& SubgraphTester::AddDynamicallyQuantizedTensor(
    const std::vector<size_t>& dims, uint32_t external_id, uint32_t flags) {
  assert(external_id < subgraph_->external_value_ids);
  uint32_t id_out = 0;
  const xnn_status status = xnn_define_dynamically_quantized_tensor_value(
      subgraph_.get(), xnn_datatype_qdint8, dims.size(), 1, dims.data(),
      external_id, flags, &id_out);
  EXPECT_EQ(status, xnn_status_success);
  EXPECT_EQ(id_out, external_id);

  return *this;
}

SubgraphTester& SubgraphTester::AddStaticTensorQS8(
    const std::vector<size_t>& dims, size_t channel_dim, TensorType tensor_type,
    const float* scale, uint32_t external_id, uint32_t flags, int8_t* data) {
  assert(external_id < subgraph_->external_value_ids);
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
      std::fill_n(data, num_elements, 0);
      std::generate_n(data, num_nonzero_elements,
                      [&]() { return w8dist(rng_); });
      std::shuffle(data, data + num_elements, rng_);
    }
  }

  uint32_t id_out;
  const xnn_status status = xnn_define_channelwise_quantized_tensor_value(
      subgraph_.get(), xnn_datatype_qcint8, scale, dims.size(), channel_dim,
      dims.data(), data, external_id, flags, &id_out);
  EXPECT_EQ(status, xnn_status_success);
  EXPECT_EQ(id_out, external_id);
  return *this;
}

SubgraphTester& SubgraphTester::AddStaticTensorF32(
    const std::vector<size_t>& dims, TensorType tensor_type,
    uint32_t external_id, uint32_t flags, float* data) {
  assert(external_id < subgraph_->external_value_ids);
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
      std::fill_n(data, num_elements, 0.0f);
      std::generate_n(data, num_nonzero_elements,
                      [&]() { return f32dist(rng_); });
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

SubgraphTester& SubgraphTester::AddInputTensor(
    size_t rank, xnn_datatype datatype, xnn_quantization_params quantization,
    uint32_t external_id) {
  std::vector<size_t> dims(rank);
  AddDynamicTensor(dims, external_id, datatype, quantization,
                   XNN_VALUE_FLAG_EXTERNAL_INPUT);
  auto it = external_tensors_.insert({external_id, nullptr});
  EXPECT_TRUE(it.second);
  return *this;
}

SubgraphTester& SubgraphTester::AddInputTensorF32(
    const std::vector<size_t>& dims, uint32_t external_id) {
  AddDynamicTensorF32(dims, external_id, XNN_VALUE_FLAG_EXTERNAL_INPUT);
  size_t num_elements = NumElements(dims);
  xnnpack::Buffer<char> input(num_elements * sizeof(float) +
                              XNN_EXTRA_BYTES * sizeof(char));
  float* data = reinterpret_cast<float*>(input.data());
  std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
  auto it = external_tensors_.insert({external_id, data});
  buffers_[external_id] = std::move(input);
  EXPECT_TRUE(it.second);
  return *this;
}

SubgraphTester& SubgraphTester::AddInputTensorQS8(
    int32_t zero_point, float scale, const std::vector<size_t>& dims,
    uint32_t external_id) {
  AddDynamicTensorQS8(zero_point, scale, dims, external_id,
                      XNN_VALUE_FLAG_EXTERNAL_INPUT);
  size_t num_elements = NumElements(dims);
  xnnpack::Buffer<char> input(num_elements * sizeof(float) +
                              XNN_EXTRA_BYTES * sizeof(char));
  float* data = reinterpret_cast<float*>(input.data());
  std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
  auto it = external_tensors_.insert({external_id, data});
  buffers_[external_id] = std::move(input);
  EXPECT_TRUE(it.second);
  return *this;
}

SubgraphTester& SubgraphTester::AddOutputTensor(
    size_t rank, xnn_datatype datatype, xnn_quantization_params quantization,
    uint32_t external_id) {
  std::vector<size_t> dims(rank);
  AddDynamicTensor(dims, external_id, datatype, quantization,
                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  auto it = external_tensors_.insert({external_id, nullptr});
  EXPECT_TRUE(it.second);
  return *this;
}

SubgraphTester& SubgraphTester::AddOutputTensorF32(
    const std::vector<size_t>& dims, uint32_t external_id) {
  output_id_ = external_id;
  AddDynamicTensorF32(dims, external_id, XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  size_t num_elements = NumElements(dims);
  xnnpack::Buffer<char> output(num_elements * sizeof(float));
  auto it = external_tensors_.insert({external_id, output.data()});
  buffers_[external_id] = std::move(output);
  EXPECT_TRUE(it.second);
  return *this;
}

SubgraphTester& SubgraphTester::AddConcatenate(size_t axis,
                                               std::vector<uint32_t> input_ids,
                                               uint32_t output_id) {
  const xnn_status status =
      xnn_define_concatenate(subgraph_.get(), axis, input_ids.size(),
                             input_ids.data(), output_id, 0 /* flags */);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddConstantPad(const size_t* pre_paddings,
                                               const size_t* post_paddings,
                                               float padding_value,
                                               uint32_t input_id,
                                               uint32_t output_id) {
  const xnn_status status = xnn_define_static_constant_pad(
      subgraph_.get(), pre_paddings, post_paddings, padding_value, input_id,
      output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddConstantPad(
    const std::vector<size_t>& pre_paddings,
    const std::vector<size_t>& post_paddings, float padding_value,
    uint32_t input_id, uint32_t output_id) {
  const xnn_status status = xnn_define_static_constant_pad(
      subgraph_.get(), pre_paddings.data(), post_paddings.data(), padding_value,
      input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddTranspose(const std::vector<size_t>& perm,
                                             uint32_t input_id,
                                             uint32_t output_id) {
  const xnn_status status =
      xnn_define_static_transpose(subgraph_.get(), perm.size(), perm.data(),
                                  input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddExpandDims(
    const std::vector<size_t>& new_axes, uint32_t input_id,
    uint32_t output_id) {
  const xnn_status status = xnn_define_static_expand_dims(
      subgraph_.get(), new_axes.size(), new_axes.data(), input_id, output_id,
      /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddReshape(const std::vector<size_t>& new_dims,
                                           uint32_t input_id,
                                           uint32_t output_id) {
  const xnn_status status = xnn_define_static_reshape(
      subgraph_.get(), new_dims.size(), new_dims.data(), input_id, output_id,
      /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddResizeBilinear(size_t new_height,
                                                  size_t new_width,
                                                  uint32_t input_id,
                                                  uint32_t output_id,
                                                  uint32_t flags) {
  const xnn_status status = xnn_define_static_resize_bilinear_2d(
      subgraph_.get(), new_height, new_width, input_id, output_id, flags);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddFuseDims(size_t first_dim, size_t num_dims,
                                            uint32_t input_id,
                                            uint32_t output_id) {
  const xnn_status status = xnn_define_fuse_dims(subgraph_.get(), first_dim,
                                                 num_dims, input_id, output_id,
                                                 /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddSplitDim(size_t axis,
                                            const std::vector<size_t>& splits,
                                            uint32_t input_id,
                                            uint32_t output_id) {
  const xnn_status status = xnn_define_split_dim(
      subgraph_.get(), axis, splits.size(), splits.data(), input_id, output_id,
      /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddSpaceToDepth2D(size_t block_size,
                                                  uint32_t input_id,
                                                  uint32_t output_id) {
  const xnn_status status = xnn_define_space_to_depth_2d(
      subgraph_.get(), block_size, input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddDepthToSpace2D(size_t block_size,
                                                  uint32_t input_id,
                                                  uint32_t output_id) {
  const xnn_status status = xnn_define_depth_to_space_2d(
      subgraph_.get(), block_size, input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddSlice(const std::vector<int64_t>& begins,
                                         const std::vector<int64_t>& ends,
                                         const std::vector<int64_t>& strides,
                                         uint32_t input_id,
                                         uint32_t output_id) {
  const xnn_status status = xnn_define_static_slice_v3(
      subgraph_.get(), begins.size(), begins.data(), ends.data(),
      strides.data(), input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddConvert(uint32_t input_id,
                                           uint32_t output_id) {
  const xnn_status status =
      xnn_define_unary(subgraph_.get(), xnn_unary_convert, /*params=*/nullptr,
                       input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddBinary(xnn_binary_operator op,
                                          xnn_binary_params* params,
                                          uint32_t input1_id,
                                          uint32_t input2_id,
                                          uint32_t output_id) {
  const xnn_status status =
      xnn_define_binary(subgraph_.get(), op, params, input1_id, input2_id,
                        output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddUnary(xnn_unary_operator op,
                                         xnn_unary_params* params,
                                         uint32_t input_id,
                                         uint32_t output_id) {
  const xnn_status status = xnn_define_unary(subgraph_.get(), op, params,
                                             input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddConvolution2D(ConvolutionParams params,
                                                 uint32_t input_id,
                                                 uint32_t filter_id,
                                                 uint32_t bias_id,
                                                 uint32_t output_id) {
  const xnn_status status = xnn_define_convolution_2d(
      subgraph_.get(), params.padding.top, params.padding.right,
      params.padding.bottom, params.padding.left, params.kernel.height,
      params.kernel.width, params.subsampling.height, params.subsampling.width,
      params.dilation.height, params.dilation.width, params.groups,
      params.group_input_channels, params.group_output_channels,
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
      output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddCopy(uint32_t input_id, uint32_t output_id) {
  const xnn_status status =
      xnn_define_copy(subgraph_.get(), input_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddDepthwiseConvolution2D(
    DepthwiseConvolutionParams params, uint32_t input_id, uint32_t filter_id,
    uint32_t bias_id, uint32_t output_id) {
  const xnn_status status = xnn_define_depthwise_convolution_2d(
      subgraph_.get(), params.padding.top, params.padding.right,
      params.padding.bottom, params.padding.left, params.kernel.height,
      params.kernel.width, params.subsampling.height, params.subsampling.width,
      params.dilation.height, params.dilation.width, params.depth_multiplier,
      params.input_channels, params.output_min, params.output_max, input_id,
      filter_id, bias_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddAddition(uint32_t input_id1,
                                            uint32_t input_id2,
                                            uint32_t output_id) {
  struct xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
  const xnn_status status =
      xnn_define_binary(subgraph_.get(), xnn_binary_add, &params, input_id1,
                        input_id2, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddAveragePooling2D(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t input_id, uint32_t output_id) {
  const xnn_status status = xnn_define_average_pooling_2d(
      subgraph_.get(), input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, pooling_height, pooling_width,
      stride_height, stride_width, -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, output_id,
      /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddClamp(float output_min, float output_max,
                                         uint32_t input_id,
                                         uint32_t output_id) {
  xnn_unary_params params;
  params.clamp.min = output_min;
  params.clamp.max = output_max;
  const xnn_status status =
      xnn_define_unary(subgraph_.get(), xnn_unary_clamp, &params, input_id,
                       output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddRoPE(uint32_t input_id1, uint32_t input_id2,
                                        uint32_t output_id) {
  const xnn_status status = xnn_define_rope(subgraph_.get(), 0, input_id1,
                                            input_id2, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddDeconvolution2D(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t adjustment_height, uint32_t adjustment_width,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t upsampling_height,
    uint32_t upsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, uint32_t input_id, uint32_t filter_id,
    uint32_t bias_id, uint32_t output_id) {
  const xnn_status status = xnn_define_deconvolution_2d(
      subgraph_.get(), input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, adjustment_height,
      adjustment_width, kernel_height, kernel_width, upsampling_height,
      upsampling_width, dilation_height, dilation_width, groups,
      group_input_channels, group_output_channels,
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
      output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddDeconvolution2D(DeconvolutionParams params,
                                                   uint32_t input_id,
                                                   uint32_t filter_id,
                                                   uint32_t bias_id,
                                                   uint32_t output_id) {
  const xnn_status status = xnn_define_deconvolution_2d(
      subgraph_.get(), params.padding.top, params.padding.right,
      params.padding.bottom, params.padding.left, params.adjustment.height,
      params.adjustment.width, params.kernel.height, params.kernel.width,
      params.upsampling.height, params.upsampling.width, params.dilation.height,
      params.dilation.width, params.groups, params.group_input_channels,
      params.group_output_channels, -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
      output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddDivide(uint32_t input_id1,
                                          uint32_t input_id2,
                                          uint32_t output_id) {
  struct xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
  const xnn_status status =
      xnn_define_binary(subgraph_.get(), xnn_binary_divide, &params, input_id1,
                        input_id2, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddEvenSplit(size_t split_dim,
                                             uint32_t input_id,
                                             std::vector<uint32_t> output_ids) {
  const xnn_status status = xnn_define_even_split(
      subgraph_.get(), split_dim, input_id, output_ids.size(),
      output_ids.data(), 0 /* flags */);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddFullyConnected(uint32_t input_id,
                                                  uint32_t filter_id,
                                                  uint32_t bias_id,
                                                  uint32_t output_id,
                                                  uint32_t flags) {
  const xnn_status status = xnn_define_fully_connected(
      subgraph_.get(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, filter_id, bias_id,
      output_id, flags);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddBatchMatrixMultiply(uint32_t input_a_id,
                                                       uint32_t input_b_id,
                                                       uint32_t output_id,
                                                       uint32_t flags) {
  const xnn_status status = xnn_define_batch_matrix_multiply(
      subgraph_.get(), input_a_id, input_b_id, output_id, flags);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddGlobalAveragePooling(uint32_t input_id,
                                                        uint32_t output_id) {
  int64_t reduction_axes[2] = {1, 2};
  const xnn_status status = xnn_define_static_reduce_v2(
      subgraph_.get(), xnn_reduce_mean, 2, &reduction_axes[0], input_id,
      output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddHardSwish(uint32_t input_id,
                                             uint32_t output_id) {
  const xnn_status status =
      xnn_define_unary(subgraph_.get(), xnn_unary_hardswish, nullptr, input_id,
                       output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddLeakyRelu(float negative_slope,
                                             uint32_t input_id,
                                             uint32_t output_id) {
  xnn_unary_params params;
  params.leaky_relu.negative_slope = negative_slope;
  const xnn_status status =
      xnn_define_unary(subgraph_.get(), xnn_unary_leaky_relu, &params, input_id,
                       output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddMaxPooling2D(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
    uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
    uint32_t input_id, uint32_t output_id) {
  const xnn_status status = xnn_define_max_pooling_2d(
      subgraph_.get(), input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, pooling_height, pooling_width,
      stride_height, stride_width, dilation_height, dilation_width,
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), input_id, output_id,
      /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddArgMaxPooling2D(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t input_id,
    uint32_t output_value_id, uint32_t output_index_id) {
  const xnn_status status = xnn_define_argmax_pooling_2d(
      subgraph_.get(), input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, pooling_height, pooling_width,
      input_id, output_value_id, output_index_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddUnpooling2D(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t pooling_height, uint32_t pooling_width, uint32_t input_value_id,
    uint32_t input_index_id, uint32_t output_id) {
  const xnn_status status = xnn_define_unpooling_2d(
      subgraph_.get(), input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, pooling_height, pooling_width,
      input_value_id, input_index_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddMultiply(uint32_t input_id1,
                                            uint32_t input_id2,
                                            uint32_t output_id) {
  struct xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
  const xnn_status status =
      xnn_define_binary(subgraph_.get(), xnn_binary_multiply, &params,
                        input_id1, input_id2, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddPrelu(uint32_t input_id, uint32_t slope_id,
                                         uint32_t output_id) {
  const xnn_status status =
      xnn_define_binary(subgraph_.get(), xnn_binary_prelu, /*params=*/nullptr,
                        input_id, slope_id, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddSubtract(uint32_t input_id1,
                                            uint32_t input_id2,
                                            uint32_t output_id) {
  struct xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
  const xnn_status status =
      xnn_define_binary(subgraph_.get(), xnn_binary_subtract, &params,
                        input_id1, input_id2, output_id, /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::AddReduce(
    xnn_reduce_operator reduce_operator,
    const std::vector<int64_t>& reduction_axes, uint32_t input_id,
    uint32_t output_id, uint32_t flags) {
  const xnn_status status = xnn_define_static_reduce_v2(
      subgraph_.get(), reduce_operator, reduction_axes.size(),
      reduction_axes.data(), input_id, output_id, flags);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::AddSoftmax(uint32_t input_id,
                                           uint32_t output_id, uint32_t flags) {
  const xnn_status status =
      xnn_define_softmax(subgraph_.get(), input_id, output_id, flags);
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::Optimize() {
  const xnn_status status = xnn_subgraph_optimize(subgraph_.get(), /*flags=*/0);
  EXPECT_EQ(status, xnn_status_success);

  return *this;
}

SubgraphTester& SubgraphTester::RewriteForNchw() {
  xnn_subgraph_rewrite_for_nchw(subgraph_.get());

  return *this;
}

SubgraphTester& SubgraphTester::RewriteForFp16() {
  EXPECT_TRUE(xnn_subgraph_rewrite_for_fp16(subgraph_.get()));

  return *this;
}

SubgraphTester& SubgraphTester::RewriteForFp16WithFailure() {
  EXPECT_FALSE(xnn_subgraph_rewrite_for_fp16(subgraph_.get()));

  return *this;
}

xnn_status SubgraphTester::CreateRuntime(xnn_weights_cache_t weights_cache,
                                         xnn_workspace_t workspace,
                                         pthreadpool_t threadpool,
                                         uint32_t flags) {
  EXPECT_EQ(runtime_, nullptr);
  xnn_runtime_t runtime = nullptr;
  const xnn_status status = xnn_create_runtime_v4(
      subgraph_.get(), weights_cache, workspace, threadpool, flags, &runtime);
  runtime_.reset(runtime);
  return status;
}

SubgraphTester& SubgraphTester::ReshapeRuntime() {
  const xnn_status status = xnn_reshape_runtime(runtime_.get());
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::SetupRuntime() {
  std::vector<xnn_external_value> values;
  values.reserve(external_tensors_.size());
  for (const std::pair<uint32_t, void*> i : external_tensors_) {
    values.push_back({i.first, i.second});
  }
  const xnn_status status =
      xnn_setup_runtime_v2(runtime_.get(), values.size(), values.data());
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

SubgraphTester& SubgraphTester::InvokeRuntime() {
  const xnn_status status = xnn_invoke_runtime(runtime_.get());
  EXPECT_EQ(status, xnn_status_success);
  return *this;
}

}  // namespace xnnpack
