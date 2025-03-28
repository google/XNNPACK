// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_TEST_SUBGRAPH_SUBGRAPH_TESTER_H_
#define THIRD_PARTY_XNNPACK_TEST_SUBGRAPH_SUBGRAPH_TESTER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/runtime-flags.h"
#include <pthreadpool.h>

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
  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();
};

class SubgraphTester {
 public:
  explicit SubgraphTester(uint32_t external_value_ids);

  SubgraphTester& AddInternalDynamicTensorF32(const std::vector<size_t>& dims,
                                              uint32_t* id_out,
                                              uint32_t flags = 0);

  SubgraphTester& AddDynamicTensor(const std::vector<size_t>& dims,
                                   uint32_t external_id, xnn_datatype datatype,
                                   xnn_quantization_params quantization,
                                   uint32_t flags = 0);

  SubgraphTester& AddDynamicTensor(const std::vector<size_t>& dims,
                                   uint32_t external_id, xnn_datatype datatype,
                                   uint32_t flags = 0);
  template <typename T>
  SubgraphTester& ReshapeExternalTensor(const std::vector<size_t>& dims,
                                        T* data, uint32_t external_id) {
    assert(external_id < subgraph_->external_value_ids);
    const xnn_status status = xnn_reshape_external_value(
        runtime_.get(), external_id, dims.size(), dims.data());
    EXPECT_EQ(status, xnn_status_success);
    external_tensors_[external_id] = data;

    return *this;
  }

  template <typename T>
  SubgraphTester& SetupExternalTensor(T* data, uint32_t external_id) {
    assert(external_id < subgraph_->external_value_ids);
    external_tensors_[external_id] = data;
    return *this;
  }

  std::vector<size_t> GetExternalTensorShape(uint32_t external_id);

  SubgraphTester& AddDynamicTensorF32(const std::vector<size_t>& dims,
                                      uint32_t external_id,
                                      uint32_t flags = 0) {
    return AddDynamicTensor(dims, external_id, xnn_datatype_fp32, flags);
  }

  template <typename T>
  inline SubgraphTester& AddStaticTensor(const std::vector<size_t>& dims,
                                         uint32_t external_id, T* data,
                                         uint32_t flags = 0) {
    return AddStaticTensor(dims, external_id, data, {0, 1.0f}, flags);
  }

  template <typename T>
  inline SubgraphTester& AddStaticTensor(const std::vector<size_t>& dims,
                                         uint32_t external_id, T* data,
                                         xnn_quantization_params quantization,
                                         uint32_t flags = 0) {
    uint32_t id_out = 0;
    xnn_status status;
    if (xnn_datatype_is_quantized(xnn_datatype_of<T>())) {
      status = xnn_define_quantized_tensor_value(
          subgraph_.get(), xnn_datatype_of<T>(), quantization.zero_point,
          quantization.scale, dims.size(), dims.data(), data, external_id,
          flags, &id_out);
    } else {
      status = xnn_define_tensor_value(subgraph_.get(), xnn_datatype_of<T>(),
                                       dims.size(), dims.data(), data,
                                       external_id, flags, &id_out);
    }
    EXPECT_EQ(status, xnn_status_success);
    EXPECT_EQ(id_out, external_id);

    return *this;
  }

  SubgraphTester& AddStaticTensorF32(const std::vector<size_t>& dims,
                                     uint32_t external_id, float* data,
                                     uint32_t flags = 0) {
    return AddStaticTensor(dims, external_id, data, flags);
  }

  SubgraphTester& AddStaticTensorF16(const std::vector<size_t>& dims,
                                     uint32_t external_id, xnn_float16* data,
                                     uint32_t flags = 0) {
    return AddStaticTensor(dims, external_id, data, flags);
  }

  SubgraphTester& AddDynamicTensorQS8(int32_t zero_point, float scale,
                                      const std::vector<size_t>& dims,
                                      uint32_t external_id,
                                      uint32_t flags = 0) {
    return AddDynamicTensor(dims, external_id, xnn_datatype_qint8,
                            {zero_point, scale}, flags);
  }

  SubgraphTester& AddDynamicallyQuantizedTensor(const std::vector<size_t>& dims,
                                                uint32_t external_id,
                                                uint32_t flags = 0);

  SubgraphTester& AddStaticTensorQS8(const std::vector<size_t>& dims,
                                     size_t channel_dim, TensorType tensor_type,
                                     const float* scale, uint32_t external_id,
                                     uint32_t flags = 0,
                                     int8_t* data = nullptr);

  SubgraphTester& AddStaticTensorF32(const std::vector<size_t>& dims,
                                     TensorType tensor_type,
                                     uint32_t external_id, uint32_t flags = 0,
                                     float* data = nullptr);

  template <typename T>
  SubgraphTester& AddInputTensor(const std::vector<size_t>& dims, T* data,
                                 xnn_quantization_params quantization,
                                 uint32_t external_id) {
    AddDynamicTensor(dims, external_id, xnn_datatype_of<T>(), quantization,
                     XNN_VALUE_FLAG_EXTERNAL_INPUT);
    auto it = external_tensors_.insert({external_id, data});
    EXPECT_TRUE(it.second);
    return *this;
  }

  template <typename T>
  SubgraphTester& AddInputTensor(const std::vector<size_t>& dims, T* data,
                                 uint32_t external_id) {
    assert(!xnn_datatype_is_quantized(xnn_datatype_of<T>()));
    return AddInputTensor(dims, data, {}, external_id);
  }

  SubgraphTester& AddInputTensor(size_t rank, xnn_datatype datatype,
                                 xnn_quantization_params quantization,
                                 uint32_t external_id);

  SubgraphTester& AddInputTensor(size_t rank, xnn_datatype datatype,
                                 uint32_t external_id) {
    return AddInputTensor(rank, datatype, {}, external_id);
  }

  SubgraphTester& AddInputTensorF32(const std::vector<size_t>& dims,
                                    uint32_t external_id);

  SubgraphTester& AddInputTensorQS8(int32_t zero_point, float scale,
                                    const std::vector<size_t>& dims,
                                    uint32_t external_id);

  template <typename T>
  SubgraphTester& AddOutputTensor(const std::vector<size_t>& dims, T* data,
                                  xnn_quantization_params quantization,
                                  uint32_t external_id) {
    AddDynamicTensor(dims, external_id, xnn_datatype_of<T>(), quantization,
                     XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
    auto it = external_tensors_.insert({external_id, data});
    EXPECT_TRUE(it.second);
    return *this;
  }

  template <typename T>
  SubgraphTester& AddOutputTensor(const std::vector<size_t>& dims, T* data,
                                  uint32_t external_id) {
    return AddOutputTensor(dims, data, {}, external_id);
  }

  SubgraphTester& AddOutputTensor(size_t rank, xnn_datatype datatype,
                                  xnn_quantization_params quantization,
                                  uint32_t external_id);

  SubgraphTester& AddOutputTensor(size_t rank, xnn_datatype datatype,
                                  uint32_t external_id) {
    assert(!xnn_datatype_is_quantized(datatype));
    return AddOutputTensor(rank, datatype, {}, external_id);
  }

  SubgraphTester& AddOutputTensorF32(const std::vector<size_t>& dims,
                                     uint32_t external_id);

  SubgraphTester& AddConcatenate(size_t axis, std::vector<uint32_t> input_ids,
                                 uint32_t output_id);

  SubgraphTester& AddConstantPad(const size_t* pre_paddings,
                                 const size_t* post_paddings,
                                 float padding_value, uint32_t input_id,
                                 uint32_t output_id);

  SubgraphTester& AddConstantPad(const std::vector<size_t>& pre_paddings,
                                 const std::vector<size_t>& post_paddings,
                                 float padding_value, uint32_t input_id,
                                 uint32_t output_id);

  SubgraphTester& AddTranspose(const std::vector<size_t>& perm,
                               uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddExpandDims(const std::vector<size_t>& new_axes,
                                uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddReshape(const std::vector<size_t>& new_dims,
                             uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddResizeBilinear(size_t new_height, size_t new_width,
                                    uint32_t input_id, uint32_t output_id,
                                    uint32_t flags = 0);

  SubgraphTester& AddFuseDims(size_t first_dim, size_t num_dims,
                              uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddSplitDim(size_t axis, const std::vector<size_t>& splits,
                              uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddSpaceToDepth2D(size_t block_size, uint32_t input_id,
                                    uint32_t output_id);

  SubgraphTester& AddDepthToSpace2D(size_t block_size, uint32_t input_id,
                                    uint32_t output_id);

  SubgraphTester& AddSlice(const std::vector<int64_t>& begins,
                           const std::vector<int64_t>& ends,
                           const std::vector<int64_t>& strides,
                           uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddConvert(uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddBinary(xnn_binary_operator op, xnn_binary_params* params,
                            uint32_t input1_id, uint32_t input2_id,
                            uint32_t output_id);

  SubgraphTester& AddUnary(xnn_unary_operator op, xnn_unary_params* params,
                           uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddConvolution2D(ConvolutionParams params, uint32_t input_id,
                                   uint32_t filter_id, uint32_t bias_id,
                                   uint32_t output_id);

  SubgraphTester& AddCopy(uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddDepthwiseConvolution2D(DepthwiseConvolutionParams params,
                                            uint32_t input_id,
                                            uint32_t filter_id,
                                            uint32_t bias_id,
                                            uint32_t output_id);

  SubgraphTester& AddAddition(uint32_t input_id1, uint32_t input_id2,
                              uint32_t output_id);

  SubgraphTester& AddAveragePooling2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
      uint32_t stride_width, uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddClamp(float output_min, float output_max,
                           uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddRoPE(uint32_t input_id1, uint32_t input_id2,
                          uint32_t output_id);

  SubgraphTester& AddDeconvolution2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t adjustment_height, uint32_t adjustment_width,
      uint32_t kernel_height, uint32_t kernel_width, uint32_t upsampling_height,
      uint32_t upsampling_width, uint32_t dilation_height,
      uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
      size_t group_output_channels, uint32_t input_id, uint32_t filter_id,
      uint32_t bias_id, uint32_t output_id);

  SubgraphTester& AddDeconvolution2D(DeconvolutionParams params,
                                     uint32_t input_id, uint32_t filter_id,
                                     uint32_t bias_id, uint32_t output_id);

  SubgraphTester& AddDivide(uint32_t input_id1, uint32_t input_id2,
                            uint32_t output_id);

  SubgraphTester& AddEvenSplit(size_t split_dim, uint32_t input_id,
                               std::vector<uint32_t> output_ids);

  SubgraphTester& AddFullyConnected(uint32_t input_id, uint32_t filter_id,
                                    uint32_t bias_id, uint32_t output_id,
                                    uint32_t flags = 0);

  SubgraphTester& AddBatchMatrixMultiply(uint32_t input_a_id,
                                         uint32_t input_b_id,
                                         uint32_t output_id,
                                         uint32_t flags = 0);

  SubgraphTester& AddGlobalAveragePooling(uint32_t input_id,
                                          uint32_t output_id);

  SubgraphTester& AddHardSwish(uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddLeakyRelu(float negative_slope, uint32_t input_id,
                               uint32_t output_id);

  SubgraphTester& AddMaxPooling2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t pooling_height, uint32_t pooling_width, uint32_t stride_height,
      uint32_t stride_width, uint32_t dilation_height, uint32_t dilation_width,
      uint32_t input_id, uint32_t output_id);

  SubgraphTester& AddArgMaxPooling2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t pooling_height, uint32_t pooling_width, uint32_t input_id,
      uint32_t output_value_id, uint32_t output_index_id);

  SubgraphTester& AddUnpooling2D(
      uint32_t input_padding_top, uint32_t input_padding_right,
      uint32_t input_padding_bottom, uint32_t input_padding_left,
      uint32_t pooling_height, uint32_t pooling_width, uint32_t input_value_id,
      uint32_t input_index_id, uint32_t output_id);

  SubgraphTester& AddMultiply(uint32_t input_id1, uint32_t input_id2,
                              uint32_t output_id);

  SubgraphTester& AddPrelu(uint32_t input_id, uint32_t slope_id,
                           uint32_t output_id);

  SubgraphTester& AddSubtract(uint32_t input_id1, uint32_t input_id2,
                              uint32_t output_id);

  SubgraphTester& AddReduce(xnn_reduce_operator reduce_operator,
                            const std::vector<int64_t>& reduction_axes,
                            uint32_t input_id, uint32_t output_id,
                            uint32_t flags = 0);

  SubgraphTester& AddSoftmax(uint32_t input_id, uint32_t output_id,
                             uint32_t flags = 0);

  SubgraphTester& Optimize();

  SubgraphTester& RewriteForNchw();

  SubgraphTester& RewriteForFp16();

  SubgraphTester& RewriteForFp16WithFailure();

  xnn_status CreateRuntime(xnn_weights_cache_t weights_cache,
                           xnn_workspace_t workspace, pthreadpool_t threadpool,
                           uint32_t flags);

  SubgraphTester& ReshapeRuntime();

  SubgraphTester& SetupRuntime();

  xnn_status CreateRuntime(pthreadpool_t threadpool = nullptr,
                           uint32_t flags = xnn_test_runtime_flags()) {
    return CreateRuntime(nullptr, nullptr, threadpool, flags);
  }

  SubgraphTester& InvokeRuntime();

  xnn_layout_type GetLayout(uint32_t value_id) const {
    return subgraph_->values[value_id].layout;
  }

  const xnn_value* Value(uint32_t value_id) const {
    return &subgraph_->values[value_id];
  }

  const xnn_node* Node(uint32_t node_id) const {
    return &subgraph_->nodes[node_id];
  }

  size_t NumNodes() const { return subgraph_->num_nodes; }

  size_t NumValues() const { return subgraph_->num_values; }

  xnn_subgraph* Subgraph() const { return subgraph_.get(); }

  template <typename T>
  float* GetExternalTensorData(uint32_t external_id) {
    assert(external_id < subgraph_->external_value_ids);
    return reinterpret_cast<T*>(external_tensors_[external_id]);
  }

  float* GetExternalTensorDataF32(uint32_t external_id) {
    assert(external_id < subgraph_->external_value_ids);
    return GetExternalTensorData<float>(external_id);
  }

  static size_t NumElements(const std::vector<size_t>& dims) {
    return std::accumulate(std::begin(dims), std::end(dims), size_t(1),
                           std::multiplies<size_t>());
  }

 protected:
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_{
      nullptr, xnn_delete_subgraph};
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr, xnn_delete_runtime};
  std::unordered_map<uint32_t, xnnpack::Buffer<char>> buffers_;
  std::unordered_map<uint32_t, void*> external_tensors_;
  uint32_t output_id_;
  xnnpack::ReplicableRandomDevice rng_;
  std::uniform_real_distribution<float> f32dist =
      std::uniform_real_distribution<float>(-1.0f, +1.0f);
  std::uniform_int_distribution<int32_t> w8dist =
      std::uniform_int_distribution<int32_t>(
          -std::numeric_limits<int8_t>::max(),
          std::numeric_limits<int8_t>::max());

 private:
  std::vector<xnnpack::Buffer<char>> static_data_;
};

}  // namespace xnnpack

#endif  // THIRD_PARTY_XNNPACK_TEST_SUBGRAPH_SUBGRAPH_TESTER_H_
