// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::min.
#include <array>      // For std::array.
#include <cassert>    // For std::cassert.
#include <cmath>      // For std::lrintf.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <functional>
#include <limits>   // For std::numeric_limits.
#include <memory>   // For std::unique_ptr.
#include <numeric>  // For std::accumulate.
#include <random>   // For std::uniform_real_distribution.
#include <vector>   // For std::vector.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/requantization.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/runtime-flags.h"

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::Pointwise;

struct FullyConnectedTestParam {
  explicit FullyConnectedTestParam(bool use_bias_) : use_bias(use_bias_) {}
  bool use_bias;
};

template <class InputType, class KernelType = InputType,
          class BiasType = InputType, class OutputType = InputType,
          int WeightsPerElement = 1>
class FullyConnectedTestBase
    : public ::testing::TestWithParam<FullyConnectedTestParam> {
 protected:
  FullyConnectedTestBase() {
    f32dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    scale_dist = std::uniform_real_distribution<float>(1.0f, 5.0f);
    i32dist = std::uniform_int_distribution<int32_t>(-10000, 10000);
    auto shape_dist = std::uniform_int_distribution<size_t>(2, 3);
    dim_dist = std::uniform_int_distribution<size_t>(5, 15);
    i8dist = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    w8dist = std::uniform_int_distribution<int32_t>(
        -std::numeric_limits<uint8_t>::max(),
        std::numeric_limits<uint8_t>::max());

    output_min = f32dist(rng);
    if (output_min > 0.0) {
      output_min = -std::numeric_limits<float>::infinity();
    }
    output_max = f32dist(rng);
    if (output_max < 0.0) {
      output_max = std::numeric_limits<float>::infinity();
    }

    size_t num_input_dims = shape_dist(rng);
    input_dims = RandomShape(num_input_dims);
    assert(input_dims.size() >= 2);
    output_channels = dim_dist(rng);
    input_channels = input_dims.back();
    input_channels = round_up_po2(input_channels, WeightsPerElement);
    input_dims.back() = input_channels;
    kernel_dims = {output_channels, input_channels};
    kernel_dims_tranposed = {input_channels, output_channels};
    bias_dims = {output_channels};
    output_dims = input_dims;
    output_dims[output_dims.size() - 1] = output_channels;

    batch_size = NumElements(input_dims) / input_channels;

    input = xnnpack::Buffer<InputType>(XNN_EXTRA_BYTES / sizeof(InputType) +
                                       NumElements(input_dims));
    kernel = xnnpack::Buffer<KernelType>(input_channels * output_channels /
                                         WeightsPerElement);
    kernel_fp16 =
        xnnpack::Buffer<xnn_float16>(input_channels * output_channels);
    bias = xnnpack::Buffer<BiasType>(output_channels);
    bias_fp16 = xnnpack::Buffer<xnn_float16>(output_channels);
    operator_output = xnnpack::Buffer<OutputType>(NumElements(output_dims));
    subgraph_output = xnnpack::Buffer<OutputType>(operator_output.size());
    accumulators = xnnpack::Buffer<int32_t>(batch_size * output_channels);
  }

  std::vector<size_t> RandomShape(size_t num_dims) {
    std::vector<size_t> dims(num_dims);
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  size_t NumElements(std::vector<size_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t(1),
                           std::multiplies<size_t>());
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;
  std::uniform_int_distribution<int32_t> w8dist;

  uint32_t batch_size;
  size_t input_channels;
  size_t output_channels;

  float output_min;
  float output_max;

  std::vector<size_t> input_dims;
  std::vector<size_t> kernel_dims;
  std::vector<size_t> kernel_dims_tranposed;
  std::vector<size_t> bias_dims;
  std::vector<size_t> output_dims;

  xnnpack::Buffer<InputType> input;
  xnnpack::Buffer<KernelType> kernel;
  xnnpack::Buffer<BiasType> bias;
  xnnpack::Buffer<xnn_float16> kernel_fp16;
  xnnpack::Buffer<xnn_float16> bias_fp16;
  xnnpack::Buffer<OutputType> operator_output;
  xnnpack::Buffer<OutputType> subgraph_output;
  xnnpack::Buffer<int32_t> accumulators;
};

template <class T>
class QuantizedFullyConnectedTestBase
    : public FullyConnectedTestBase<T, T, int32_t> {
 protected:
  void initialize_accumulators_from_bias() {
    for (size_t i = 0; i < this->batch_size; i++) {
      for (size_t oc = 0; oc < this->output_channels; oc++) {
        this->accumulators[i * this->output_channels + oc] = this->bias[oc];
      }
    }
  }
};

class FullyConnectedTestF32QC4W
    : public FullyConnectedTestBase<float, uint8_t, float, float, 2> {};

class FullyConnectedTestF32QC8W
    : public FullyConnectedTestBase<float, int8_t, float> {};

using FullyConnectedTestQC8 = QuantizedFullyConnectedTestBase<int8_t>;
using FullyConnectedTestQS8 = QuantizedFullyConnectedTestBase<int8_t>;
using FullyConnectedTestQU8 = QuantizedFullyConnectedTestBase<uint8_t>;
using FullyConnectedTestF16 = FullyConnectedTestBase<xnn_float16, float, float>;
using FullyConnectedTestF32 = FullyConnectedTestBase<float>;
using FullyConnectedTestBF16F32 =
    FullyConnectedTestBase<xnn_bfloat16, xnn_bfloat16, float, float>;
using DynamicFullyConnectedTestF32 = FullyConnectedTestBase<float>;

TEST_F(FullyConnectedTestQC8, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, 0, 1.0f, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  xnnpack::Buffer<float> scale(output_channels, 1.0f);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, scale.data(), kernel_dims.size(),
                0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint32, scale.data(), bias_dims.size(),
                0, bias_dims.data(), bias.data(),
                /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, 0, 1.0f, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestQS8, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, 0, 1.0f, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, 0, 1.0f, kernel_dims.size(),
                kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint32, 0, 1.0f, bias_dims.size(),
                bias_dims.data(), bias.data(),
                /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, 0, 1.0f, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestQU8, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, 0, 1.0f, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, 0, 1.0f, kernel_dims.size(),
                kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint32, 0, 1.0f, bias_dims.size(),
                bias_dims.data(), bias.data(),
                /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, 0, 1.0f, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       input_id, kernel_id, bias_id, output_id,
                                       /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_P(FullyConnectedTestF16, define) {
  bool use_bias = GetParam().use_bias;
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    kernel_dims.size(), kernel_dims.data(),
                                    kernel.data(), /*external_id=*/1,
                                    /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (use_bias) {
    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                                bias_dims.data(), bias.data(),
                                /*external_id=*/2, /*flags=*/0, &bias_id));
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, use_bias ? 3 : 2);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestF32, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    kernel_dims.size(), kernel_dims.data(),
                                    kernel.data(), /*external_id=*/1,
                                    /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestF32QC4W, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_channelwise_quantized_tensor_value_v2(
          subgraph, xnn_datatype_qcint4, /*zero_point=*/8,
          requantization_scales.data(), kernel_dims.size(), /*channel_dim=*/0,
          kernel_dims.data(), kernel.data(), /*external_id=*/1,
          /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestF32QC4W, define_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_channelwise_quantized_tensor_value_v2(
          subgraph, xnn_datatype_qcint4, /*zero_point=*/8,
          requantization_scales.data(), kernel_dims.size(), /*channel_dim=*/0,
          kernel_dims.data(), kernel.data(), /*external_id=*/1,
          /*flags=*/0, &kernel_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(
                subgraph, output_min, output_max, input_id, kernel_id,
                XNN_INVALID_VALUE_ID, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 2);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], XNN_INVALID_VALUE_ID);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestF32QC8W, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestF32QC8W, define_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(
                subgraph, output_min, output_max, input_id, kernel_id,
                XNN_INVALID_VALUE_ID, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 2);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], XNN_INVALID_VALUE_ID);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(DynamicFullyConnectedTestF32, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, kernel_dims.size(),
                              kernel_dims.data(), nullptr, /*external_id=*/1,
                              /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    bias_dims.size(), bias_dims.data(), nullptr,
                                    /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
  } else {
    const struct xnn_node* pack_lh_node = &subgraph->nodes[0];
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);
}

TEST_F(FullyConnectedTestQC8, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  const int8_t input_zero_point = -1;
  const float input_scale = scale_dist(rng);
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);
  std::generate(requantization_scales.begin(), requantization_scales.end(),
                [&]() { return std::abs(f32dist(rng)); });

  // Compute reference results, without renormalization.
  initialize_accumulators_from_bias();
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oc = 0; oc < output_channels; oc++) {
      for (size_t ic = 0; ic < input_channels; ic++) {
        accumulators[i * output_channels + oc] +=
            (static_cast<int32_t>(input[i * input_channels + ic]) -
             static_cast<int32_t>(input_zero_point)) *
            static_cast<int32_t>(kernel[oc * input_channels + ic]);
      }
    }
  }

  // Compute renormalization parameters.
  const int32_t accumulated_min =
      *std::min_element(accumulators.cbegin(), accumulators.cend());
  const int32_t accumulated_max =
      *std::max_element(accumulators.cbegin(), accumulators.cend());

  float output_scale = static_cast<double>(static_cast<uint32_t>(
                           accumulated_max - accumulated_min)) /
                       255.0;
  int8_t output_zero_point = static_cast<int8_t>(std::max(
      std::min(
          lrint(-0.5 -
                0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                    output_scale),
          static_cast<long>(std::numeric_limits<int8_t>::max())),
      static_cast<long>(std::numeric_limits<int8_t>::min())));
  const int8_t quantized_output_min =
      xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max =
      xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_qs8_qc8w(
      input_channels, output_channels, input_channels, output_channels,
      input_zero_point, input_scale, requantization_scales.data(),
      kernel.data(), bias.data(), output_zero_point, output_scale,
      quantized_output_min, quantized_output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qs8_qc8w(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_qs8_qc8w(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, input_zero_point, input_scale,
                input_dims.size(), input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), 0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint32, requantization_scales.data(),
                bias_dims.size(), 0, bias_dims.data(), bias.data(),
                /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, output_zero_point, output_scale,
                output_dims.size(), output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(subgraph_output, ElementsAreArray(operator_output));
}

TEST_F(FullyConnectedTestQS8, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  const int8_t input_zero_point = -1;
  const float input_scale = scale_dist(rng);
  const float kernel_scale = scale_dist(rng);

  // Compute reference results, without renormalization.
  initialize_accumulators_from_bias();
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oc = 0; oc < output_channels; oc++) {
      for (size_t ic = 0; ic < input_channels; ic++) {
        accumulators[i * output_channels + oc] +=
            (static_cast<int32_t>(input[i * input_channels + ic]) -
             static_cast<int32_t>(input_zero_point)) *
            static_cast<int32_t>(kernel[oc * input_channels + ic]);
      }
    }
  }

  // Compute renormalization parameters.
  const int32_t accumulated_min =
      *std::min_element(accumulators.cbegin(), accumulators.cend());
  const int32_t accumulated_max =
      *std::max_element(accumulators.cbegin(), accumulators.cend());

  float output_scale = static_cast<double>(static_cast<uint32_t>(
                           accumulated_max - accumulated_min)) /
                       255.0;
  int8_t output_zero_point = static_cast<int8_t>(std::max(
      std::min(
          lrint(-0.5 -
                0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                    output_scale),
          static_cast<long>(std::numeric_limits<int8_t>::max())),
      static_cast<long>(std::numeric_limits<int8_t>::min())));
  const int8_t quantized_output_min =
      xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max =
      xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_qs8(
      input_channels, output_channels, input_channels, output_channels,
      input_zero_point, input_scale, kernel_scale, kernel.data(), bias.data(),
      output_zero_point, output_scale, quantized_output_min,
      quantized_output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qs8(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_qs8(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, input_zero_point, input_scale,
                input_dims.size(), input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, 0, kernel_scale,
                kernel_dims.size(), kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint32, 0, kernel_scale,
                bias_dims.size(), bias_dims.data(), bias.data(),
                /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, output_zero_point, output_scale,
                output_dims.size(), output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(subgraph_output, ElementsAreArray(operator_output));
}

TEST_F(FullyConnectedTestQU8, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  const uint8_t input_zero_point = u8dist(rng);
  const uint8_t kernel_zero_point = 0;
  const float input_scale = scale_dist(rng);
  const float kernel_scale = scale_dist(rng);

  // Compute reference results, without renormalization.
  initialize_accumulators_from_bias();
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oc = 0; oc < output_channels; oc++) {
      for (size_t ic = 0; ic < input_channels; ic++) {
        accumulators[i * output_channels + oc] +=
            (static_cast<int32_t>(input[i * input_channels + ic]) -
             static_cast<int32_t>(input_zero_point)) *
            (static_cast<int32_t>(kernel[oc * input_channels + ic]) -
             static_cast<int32_t>(kernel_zero_point));
      }
    }
  }

  // Compute renormalization parameters.
  const int32_t accumulated_min =
      *std::min_element(accumulators.cbegin(), accumulators.cend());
  const int32_t accumulated_max =
      *std::max_element(accumulators.cbegin(), accumulators.cend());

  const double output_scale = static_cast<double>(static_cast<uint32_t>(
                                  accumulated_max - accumulated_min)) /
                              255.0;
  const uint8_t output_zero_point = static_cast<uint8_t>(std::max(
      std::min(
          lrint(127.5 -
                0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                    output_scale),
          static_cast<long>(std::numeric_limits<uint8_t>::max())),
      static_cast<long>(std::numeric_limits<uint8_t>::min())));
  const uint8_t quantized_output_min =
      xnn_qu8_quantize(output_min, output_scale, output_zero_point);
  const uint8_t quantized_output_max =
      xnn_qu8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_qu8(
      input_channels, output_channels, input_channels, output_channels,
      input_zero_point, input_scale, kernel_zero_point, kernel_scale,
      kernel.data(), bias.data(), output_zero_point, output_scale,
      quantized_output_min, quantized_output_max, /*flags=*/0, nullptr, nullptr,
      &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qu8(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_qu8(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, input_zero_point, input_scale,
                input_dims.size(), input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, 0, kernel_scale,
                kernel_dims.size(), kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint32, 0, kernel_scale,
                bias_dims.size(), bias_dims.data(), bias.data(),
                /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, output_zero_point, output_scale,
                output_dims.size(), output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(subgraph_output, ElementsAreArray(operator_output));
}

INSTANTIATE_TEST_SUITE_P(
    UseBias, FullyConnectedTestF16,
    testing::ValuesIn<FullyConnectedTestParam>({FullyConnectedTestParam(false),
                                                FullyConnectedTestParam(true)}),
    [](const testing::TestParamInfo<FullyConnectedTestParam>& info) {
      if (info.param.use_bias) {
        return "with_bias";
      }
      return "no_bias";
    });

TEST_P(FullyConnectedTestF16, matches_operator_api) {
  bool use_bias = GetParam().use_bias;
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  if (use_bias) {
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  }

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f16(
      input_channels, output_channels, input_channels, output_channels,
      kernel.data(), use_bias ? bias.data() : nullptr, output_min, output_max,
      XNN_FLAG_FP32_STATIC_WEIGHTS, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f16(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f16(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, kernel_dims.size(),
                              kernel_dims.data(), kernel.data(),
                              /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (use_bias) {
    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                                bias_dims.data(), bias.data(),
                                /*external_id=*/2, /*flags=*/0, &bias_id));
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(subgraph_output,
              Pointwise(FloatNear(1.0e-2),
                        std::vector<xnn_float16>(operator_output.begin(),
                                                 operator_output.end())));
}

TEST_P(FullyConnectedTestF16, matches_operator_api_f16_weights) {
  bool use_bias = GetParam().use_bias;
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(),
                [&]() { return xnn_float16_from_float(f32dist(rng)); });
  std::generate(kernel_fp16.begin(), kernel_fp16.end(),
                [&]() { return xnn_float16_from_float(f32dist(rng)); });
  if (use_bias) {
    std::generate(bias_fp16.begin(), bias_fp16.end(),
                  [&]() { return xnn_float16_from_float(f32dist(rng)); });
  }

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f16(
      input_channels, output_channels, input_channels, output_channels,
      kernel_fp16.data(), use_bias ? bias_fp16.data() : nullptr, output_min,
      output_max, /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f16(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f16(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, kernel_dims.size(),
                              kernel_dims.data(), kernel_fp16.data(),
                              /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (use_bias) {
    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, xnn_datatype_fp16, bias_dims.size(),
                                bias_dims.data(), bias_fp16.data(),
                                /*external_id=*/2, /*flags=*/0, &bias_id));
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(subgraph_output,
              Pointwise(FloatNear(1.0e-2),
                        std::vector<xnn_float16>(operator_output.begin(),
                                                 operator_output.end())));
}

TEST_F(FullyConnectedTestBF16F32, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  int counter = 0;
  std::generate(input.begin(), input.end(),
                [&]() { return static_cast<float>(counter++ % 10); });
  std::generate(kernel.begin(), kernel.end(),
                [&]() { return static_cast<float>(counter++ % 10); });
  std::generate(bias.begin(), bias.end(),
                [&]() { return static_cast<float>(counter++ % 10); });

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_bf16_f32(
      input_channels, output_channels, input_channels, output_channels,
      kernel.data(), bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_bf16_f32(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_bf16_f32(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_bf16, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_bf16, kernel_dims.size(),
                              kernel_dims.data(), kernel.data(),
                              /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(FullyConnectedTestF32QC4W, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return i8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  xnnpack::Buffer<float> requantization_scales(output_channels);
  std::generate(requantization_scales.begin(), requantization_scales.end(),
                [&]() { return scale_dist(rng); });
  uint8_t kernel_zero_point = 8;

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f32_qc4w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_zero_point, requantization_scales.data(), kernel.data(),
      bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32_qc4w(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32_qc4w(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                requantization_scales.data(), kernel_dims.size(),
                /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(FullyConnectedTestF32QC4W, matches_operator_api_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return i8dist(rng); });
  xnnpack::Buffer<float> requantization_scales(output_channels);
  std::generate(requantization_scales.begin(), requantization_scales.end(),
                [&]() { return scale_dist(rng); });
  uint8_t kernel_zero_point = 8;

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f32_qc4w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_zero_point, requantization_scales.data(), kernel.data(), nullptr,
      output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32_qc4w(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32_qc4w(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                requantization_scales.data(), kernel_dims.size(),
                /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(
                subgraph, output_min, output_max, input_id, kernel_id,
                XNN_INVALID_VALUE_ID, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(FullyConnectedTestF32QC8W, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return i8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  xnnpack::Buffer<float> requantization_scales(output_channels);
  std::generate(requantization_scales.begin(), requantization_scales.end(),
                [&]() { return scale_dist(rng); });

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f32_qc8w(
      input_channels, output_channels, input_channels, output_channels,
      requantization_scales.data(), kernel.data(), bias.data(), output_min,
      output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32_qc8w(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32_qc8w(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(FullyConnectedTestF32QC8W, matches_operator_api_without_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return i8dist(rng); });
  xnnpack::Buffer<float> requantization_scales(output_channels);
  std::generate(requantization_scales.begin(), requantization_scales.end(),
                [&]() { return scale_dist(rng); });

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f32_qc8w(
      input_channels, output_channels, input_channels, output_channels,
      requantization_scales.data(), kernel.data(),
      /*bias=*/nullptr, output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32_qc8w(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32_qc8w(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(
                subgraph, output_min, output_max, input_id, kernel_id,
                XNN_INVALID_VALUE_ID, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

INSTANTIATE_TEST_SUITE_P(
    UseBias, DynamicFullyConnectedTestF32,
    testing::ValuesIn<FullyConnectedTestParam>({FullyConnectedTestParam(false),
                                                FullyConnectedTestParam(true)}),
    [](const testing::TestParamInfo<FullyConnectedTestParam>& info) {
      if (info.param.use_bias) {
        return "with_dynamic_bias";
      }
      return "no_bias";
    });

TEST_P(DynamicFullyConnectedTestF32, matches_operator_api_dynamic_kernel) {
  bool use_bias = GetParam().use_bias;
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  if (use_bias) {
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  }

  // Call operator API.
  const xnn_status status = xnn_create_dynamic_fully_connected_nc_f32(
      output_min, output_max, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_dynamic_fully_connected_nc_f32(
                op, batch_size, input_channels, output_channels, input_channels,
                output_channels, &workspace_size, &workspace_alignment,
                /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success,
            xnn_setup_dynamic_fully_connected_nc_f32(
                op, workspace.data(), input.data(), kernel.data(),
                use_bias ? bias.data() : nullptr, operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, kernel_dims.size(),
                kernel_dims.data(), nullptr,
                /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &kernel_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (use_bias) {
    ASSERT_EQ(xnn_status_success,
              xnn_define_tensor_value(
                  subgraph, xnn_datatype_fp32, bias_dims.size(),
                  bias_dims.data(), /*data=*/nullptr,
                  /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_INPUT, &bias_id));
  }

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::vector<xnn_external_value> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{kernel_id, kernel.data()},
      xnn_external_value{output_id, subgraph_output.data()},
  };
  if (use_bias) {
    external.push_back(xnn_external_value{bias_id, bias.data()});
  }
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(DynamicFullyConnectedTestF32, matches_operator_api_dynamic_bias) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(),
                [&]() { return 1.0f + 0.0f * f32dist(rng); });

  // Call operator API.
  const xnn_status status = xnn_create_dynamic_fully_connected_nc_f32(
      output_min, output_max, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_dynamic_fully_connected_nc_f32(
                op, batch_size, input_channels, output_channels, input_channels,
                output_channels, &workspace_size, &workspace_alignment,
                /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success,
            xnn_setup_dynamic_fully_connected_nc_f32(
                op, workspace.data(), input.data(), kernel.data(), bias.data(),
                operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, kernel_dims.size(),
                              kernel_dims.data(), kernel.data(),
                              /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    bias_dims.size(), bias_dims.data(), nullptr,
                                    /*external_id=*/2,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{bias_id, bias.data()},
      xnn_external_value{output_id, subgraph_output.data()},
  };
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(FullyConnectedTestF32QC8W, matches_operator_api_transposed_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return i8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  xnnpack::Buffer<float> requantization_scales(output_channels);
  std::generate(requantization_scales.begin(), requantization_scales.end(),
                [&]() { return scale_dist(rng); });

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f32_qc8w(
      input_channels, output_channels, input_channels, output_channels,
      requantization_scales.data(), kernel.data(), bias.data(), output_min,
      output_max, XNN_FLAG_TRANSPOSE_WEIGHTS, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32_qc8w(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32_qc8w(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims_tranposed.size(),
                /*channel_dim=*/1, kernel_dims_tranposed.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       input_id, kernel_id, bias_id, output_id,
                                       XNN_FLAG_TRANSPOSE_WEIGHTS));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  EXPECT_THAT(
      subgraph_output,
      Pointwise(FloatNear(1.0e-5), std::vector<float>(operator_output.begin(),
                                                      operator_output.end())));
}

TEST_F(FullyConnectedTestF32QC8W, non_static_kernel_is_invalid_parameter) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                nullptr, /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_invalid_parameter,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));
}

TEST_F(FullyConnectedTestF32QC8W, non_static_bias_is_invalid_parameter) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    bias_dims.size(), bias_dims.data(), nullptr,
                                    /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_invalid_parameter,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));
}

TEST_F(FullyConnectedTestF32QC8W, invalid_channel_dimension) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  const size_t channel_dim = 1;
  xnnpack::Buffer<float> requantization_scales(kernel_dims[channel_dim], 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), channel_dim, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_invalid_parameter,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));
}

TEST_F(FullyConnectedTestF32QC8W,
       transposed_weights_invalid_channel_dimension) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  const size_t channel_dim = 0;
  xnnpack::Buffer<float> requantization_scales(
      kernel_dims_tranposed[channel_dim], 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_dims.size(),
                              input_dims.data(), nullptr,
                              /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims_tranposed.size(), channel_dim,
                kernel_dims_tranposed.data(), kernel.data(), /*external_id=*/1,
                /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_invalid_parameter,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       input_id, kernel_id, bias_id, output_id,
                                       XNN_FLAG_TRANSPOSE_WEIGHTS));
}

class FullyConnectedTestQD8F16QC4W
    : public FullyConnectedTestBase<int8_t, int8_t, float, xnn_float16, 2> {};

TEST_F(FullyConnectedTestQD8F16QC4W, define) {
  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  const uint8_t kernel_zero_point = 8;

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                requantization_scales.data(), kernel_dims.size(),
                /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(FullyConnectedTestQD8F16QC4W,
       internally_allocated_dynamic_quantization_parameters) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<xnn_float16> convert_input(
      batch_size * input_channels + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<xnn_float16> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_float16> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  const uint8_t kernel_zero_point = 8;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f16_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f16_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f16_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f16_qc4w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_zero_point, kernel_scale.data(), kernel.data(), bias.data(),
      output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f16_qc4w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_channelwise_quantized_tensor_value_v2(
          subgraph, xnn_datatype_qcint4, kernel_zero_point, kernel_scale.data(),
          kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
          kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  EXPECT_THAT(subgraph_output,
              Pointwise(FloatNear(1.0e-3),
                        std::vector<xnn_float16>(operator_output.begin(),
                                                 operator_output.end())));
}

class FullyConnectedTestQD8F16QB4W
    : public FullyConnectedTestBase<int8_t, uint8_t, float, xnn_float16, 2> {};

TEST_F(FullyConnectedTestQD8F16QB4W, define) {
  size_t block_size = 32;
  input_channels = round_up_po2(input_channels, block_size);

  input_dims[input_dims.size() - 1] = input_channels;
  kernel_dims[kernel_dims.size() - 1] = input_channels;

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  // We adjusted input_channels above, reallocate the kernel.
  kernel = xnnpack::Buffer<uint8_t>(output_channels * input_channels);
  const uint8_t kernel_zero_point = 8;
  xnnpack::Buffer<xnn_bfloat16> kernel_scale(output_channels * block_size);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_blockwise_quantized_tensor_value(
                subgraph, xnn_datatype_qbint4, kernel_zero_point,
                reinterpret_cast<const uint16_t*>(kernel_scale.data()),
                kernel_dims.size(),
                /*channel_dim=*/0, block_size, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(FullyConnectedTestQD8F16QB4W,
       internally_allocated_dynamic_quantization_parameters) {
  size_t block_size = 32;
  input_channels = round_up_po2(input_channels, block_size);

  input_dims[input_dims.size() - 1] = input_channels;
  kernel_dims[kernel_dims.size() - 1] = input_channels;

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<xnn_float16> convert_input(
      batch_size * input_channels + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<xnn_float16> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_float16> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<xnn_bfloat16> kernel_scale(output_channels * block_size);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });

  // We adjusted input_channels above, reallocate the kernel.
  kernel = xnnpack::Buffer<uint8_t>(output_channels * input_channels);

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  const uint8_t kernel_zero_point = 8;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f16_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f16_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f16_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f16_qb4w(
      input_channels, output_channels, input_channels, output_channels,
      block_size, kernel_zero_point,
      reinterpret_cast<const uint16_t*>(kernel_scale.data()), kernel.data(),
      bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f16_qb4w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f16_qb4w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_blockwise_quantized_tensor_value(
                subgraph, xnn_datatype_qbint4, kernel_zero_point,
                reinterpret_cast<const uint16_t*>(kernel_scale.data()),
                kernel_dims.size(),
                /*channel_dim=*/0, block_size, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
}

class FullyConnectedTestQD8F16QC8W
    : public FullyConnectedTestBase<int8_t, int8_t, float, xnn_float16> {};

TEST_F(FullyConnectedTestQD8F16QC8W, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(FullyConnectedTestQD8F16QC8W,
       internally_allocated_dynamic_quantization_parameters) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<xnn_float16> convert_input(
      batch_size * input_channels + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<xnn_float16> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_float16> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f16_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f16_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f16_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f16_qc8w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_scale.data(), kernel.data(), bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f16_qc8w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, kernel_scale.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

class FullyConnectedTestQD8F32QC8W
    : public FullyConnectedTestBase<int8_t, int8_t, float, float> {};

TEST_F(FullyConnectedTestQD8F32QC8W, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, requantization_scales.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(FullyConnectedTestQD8F32QC8W,
       internally_allocated_dynamic_quantization_parameters) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<float> convert_input(batch_size * input_channels +
                                       XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<float> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<float> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f32_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f32_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f32_qc8w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_scale.data(), kernel.data(), bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f32_qc8w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, kernel_scale.data(),
                kernel_dims.size(), /*channel_dim=*/0, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Don't look for exact matches since the result may differ if the subgraph
  // gets automatically converted from `qd8` to `qp8`.
  float max_abs_val = 1.0f;
  for (float val : operator_output) {
    max_abs_val = std::max(max_abs_val, std::abs(val));
  }
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(subgraph_output[i], operator_output[i], max_abs_val * 1e-3);
  }
}

class FullyConnectedTestQD8F32QC4W
    : public FullyConnectedTestBase<int8_t, uint8_t, float, float, 2> {};

TEST_F(FullyConnectedTestQD8F32QC4W, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnnpack::Buffer<float> requantization_scales(output_channels, 1.0f);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  const uint8_t kernel_zero_point = 8;
  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                kernel_scale.data(), kernel_dims.size(),
                /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(FullyConnectedTestQD8F32QC4W,
       internally_allocated_dynamic_quantization_parameters) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<float> convert_input(batch_size * input_channels +
                                       XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<float> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<float> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  const uint8_t kernel_zero_point = 8;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f32_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f32_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_zero_point, kernel_scale.data(), kernel.data(), bias.data(),
      output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f32_qc4w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                kernel_scale.data(), kernel_dims.size(),
                /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  float max_abs_val = 0.0f;
  for (size_t i = 0; i < operator_output.size(); i++) {
    max_abs_val = std::max(max_abs_val, std::abs(operator_output[i]));
  }
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(operator_output[i], subgraph_output[i], max_abs_val * 1e-2);
  }
}

TEST_F(FullyConnectedTestQD8F32QC4W,
       internally_allocated_dynamic_quantization_parameters_with_reshape) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  std::vector<size_t> subgraph_input_dims(input_dims);
  std::vector<size_t> subgraph_output_dims(output_dims);
  subgraph_input_dims[0] += 4;
  subgraph_output_dims[0] += 4;
  const size_t subgraph_batch_size =
      NumElements(subgraph_input_dims) / input_channels;

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<float> convert_input(subgraph_batch_size * input_channels +
                                       XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<float> subgraph_output(subgraph_batch_size * output_channels);
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<float> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  const uint8_t kernel_zero_point = 8;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f32_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f32_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_zero_point, kernel_scale.data(), kernel.data(), nullptr,
      output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f32_qc4w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  //
  // dim[0] size increments:
  //
  //  0..........2......3.......4
  //  ^          ^      ^       ^
  //  `..Input1  |      |       |
  //     Input2..'      |       |
  //     Subgraph.......'       |
  //     Input3.................'

  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, subgraph_input_dims.size(),
                subgraph_input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, subgraph_input_dims.size(),
                /*num_nonbatch_dims=*/1, subgraph_input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                kernel_scale.data(), kernel_dims.size(),
                /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, subgraph_output_dims.size(),
                subgraph_output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(
                subgraph, output_min, output_max, dq_quantized_id, kernel_id,
                XNN_INVALID_NODE_ID, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_convert);

  // 1st inference: lets start smaller than we planned memory for
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_dims.size(),
                                       input_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  float max_abs_val = 0.0f;
  for (size_t i = 0; i < operator_output.size(); i++) {
    max_abs_val = std::max(max_abs_val, std::abs(operator_output[i]));
  }
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(operator_output[i], subgraph_output[i], max_abs_val * 1e-2);
  }

  // 2nd inference: The dq-params should be properly allocated to handle a
  // resize without memory retrigger
  input_dims[0] += 2;
  size_t batch_size2 =
      std::accumulate(input_dims.begin(), input_dims.end() - 1,
                      static_cast<size_t>(1), std::multiplies<size_t>());
  xnnpack::Buffer<float> convert_input2(batch_size2 * input_channels +
                                        XNN_EXTRA_BYTES / sizeof(float));
  std::generate(convert_input2.begin(), convert_input2.end(),
                [&]() { return f32dist(rng); });
  xnnpack::Buffer<float> subgraph_output2(batch_size2 * output_channels);
  std::array<xnn_external_value, 2> external2 = {
      xnn_external_value{input_id, convert_input2.data()},
      xnn_external_value{output_id, subgraph_output2.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_dims.size(),
                                       input_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external2.size(), external2.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // 3rd inference: The dq-params should be properly allocated even with memory
  // retrigger
  input_dims[0] += 2;  // +4 total
  size_t batch_size3 =
      std::accumulate(input_dims.begin(), input_dims.end() - 1,
                      static_cast<size_t>(1), std::multiplies<size_t>());
  xnnpack::Buffer<float> convert_input3(batch_size3 * input_channels +
                                        XNN_EXTRA_BYTES / sizeof(float));
  std::generate(convert_input3.begin(), convert_input3.end(),
                [&]() { return f32dist(rng); });
  xnnpack::Buffer<float> subgraph_output3(batch_size3 * output_channels);
  std::array<xnn_external_value, 2> external3 = {
      xnn_external_value{input_id, convert_input3.data()},
      xnn_external_value{output_id, subgraph_output3.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_dims.size(),
                                       input_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external3.size(), external3.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
}

// TODO(b/381381604) - Re-enable once fixed.
TEST_F(
    FullyConnectedTestQD8F32QC4W,
    DISABLED_internally_allocated_dynamic_quantization_parameters_transposed_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<float> convert_input(batch_size * input_channels +
                                       XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<float> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<float> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  // Adjust number of kernel elements for QC4W. input_channels should be padded
  // to byte boundary, hence even.
  const size_t rounded_output_channels = round_up_po2(output_channels, 2);
  kernel = xnnpack::Buffer<uint8_t>(input_channels * rounded_output_channels);

  xnnpack::Buffer<float> kernel_scale(output_channels);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  const uint8_t kernel_zero_point = 8;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f32_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f32_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
      input_channels, output_channels, input_channels, output_channels,
      kernel_zero_point, kernel_scale.data(), kernel.data(), bias.data(),
      output_min, output_max, XNN_FLAG_TRANSPOSE_WEIGHTS, nullptr, nullptr,
      &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f32_qc4w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value_v2(
                subgraph, xnn_datatype_qcint4, kernel_zero_point,
                kernel_scale.data(), kernel_dims_tranposed.size(),
                /*channel_dim=*/1, kernel_dims_tranposed.data(), kernel.data(),
                /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, XNN_FLAG_TRANSPOSE_WEIGHTS));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  float max_abs_val = 0.0f;
  for (size_t i = 0; i < operator_output.size(); i++) {
    max_abs_val = std::max(max_abs_val, std::abs(operator_output[i]));
  }
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_NEAR(operator_output[i], subgraph_output[i], max_abs_val * 0.0625);
  }
}

class FullyConnectedTestQD8F32QB4W
    : public FullyConnectedTestBase<int8_t, uint8_t, float, float, 2> {};

TEST_F(FullyConnectedTestQD8F32QB4W, define) {
  size_t block_size = 32;
  input_channels = round_up_po2(input_channels, block_size);

  input_dims[input_dims.size() - 1] = input_channels;
  kernel_dims[kernel_dims.size() - 1] = input_channels;

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  // We adjusted input_channels above, reallocate the kernel.
  kernel = xnnpack::Buffer<uint8_t>(output_channels * input_channels);
  const uint8_t kernel_zero_point = 8;
  xnnpack::Buffer<xnn_bfloat16> kernel_scale(output_channels * block_size);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_blockwise_quantized_tensor_value(
                subgraph, xnn_datatype_qbint4, kernel_zero_point,
                reinterpret_cast<const uint16_t*>(kernel_scale.data()),
                kernel_dims.size(),
                /*channel_dim=*/0, block_size, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 3);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->inputs[1], kernel_id);
  ASSERT_EQ(node->inputs[2], bias_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(FullyConnectedTestQD8F32QB4W,
       internally_allocated_dynamic_quantization_parameters) {
  size_t block_size = 32;
  input_channels = round_up_po2(input_channels, block_size);

  input_dims[input_dims.size() - 1] = input_channels;
  kernel_dims[kernel_dims.size() - 1] = input_channels;

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/4,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  xnnpack::Buffer<float> convert_input(batch_size * input_channels +
                                       XNN_EXTRA_BYTES / sizeof(float));
  xnnpack::Buffer<int8_t> operator_dq_data(batch_size * input_channels +
                                           XNN_EXTRA_BYTES);
  xnnpack::Buffer<float> subgraph_output(batch_size * output_channels);
  xnnpack::Buffer<float> operator_output(batch_size * output_channels);
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size + XNN_EXTRA_QUANTIZATION_PARAMS);

  xnnpack::Buffer<xnn_bfloat16> kernel_scale(output_channels * block_size);
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return scale_dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(convert_input.begin(), convert_input.end(),
                [&]() { return f32dist(rng); });
  std::generate(quantization_params.begin(), quantization_params.end(), [&]() {
    return xnn_quantization_params{w8dist(rng), f32dist(rng)};
  });

  // We adjusted input_channels above, reallocate the kernel.
  kernel = xnnpack::Buffer<uint8_t>(output_channels * input_channels);

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  const uint8_t kernel_zero_point = 8;

  // Call operator API.
  xnn_operator_t convert_op = nullptr;
  xnn_operator_t fc_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f32_qd8(
                convert_op, batch_size, input_channels, input_channels,
                input_channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_convert_nc_f32_qd8(convert_op, convert_input.data(),
                                         operator_dq_data.data(),
                                         quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
      input_channels, output_channels, input_channels, output_channels,
      block_size, kernel_zero_point,
      reinterpret_cast<const uint16_t*>(kernel_scale.data()), kernel.data(),
      bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &fc_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fc_op(
      fc_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, fc_op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
                                    fc_op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_fully_connected_nc_qd8_f32_qb4w(
                fc_op, operator_dq_data.data(), operator_output.data(),
                quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(fc_op, /*threadpool=*/nullptr));

  // Call subgraph API.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input_dims.size(),
                /*num_nonbatch_dims=*/1, input_dims.data(),
                XNN_INVALID_VALUE_ID, /*flags=*/0, &dq_quantized_id));
  ASSERT_NE(dq_quantized_id, XNN_INVALID_NODE_ID);
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_blockwise_quantized_tensor_value(
                subgraph, xnn_datatype_qbint4, kernel_zero_point,
                reinterpret_cast<const uint16_t*>(kernel_scale.data()),
                kernel_dims.size(),
                /*channel_dim=*/0, block_size, kernel_dims.data(),
                kernel.data(), /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             input_id, dq_quantized_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_fully_connected(subgraph, output_min, output_max,
                                       dq_quantized_id, kernel_id, bias_id,
                                       output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, convert_input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
}

TEST_F(FullyConnectedTestF32, reshape) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    kernel_dims.size(), kernel_dims.data(),
                                    kernel.data(), /*external_id=*/1,
                                    /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  ASSERT_LE(subgraph->num_nodes, 2);
  const struct xnn_node* fc_node = nullptr;
  const struct xnn_node* pack_lh_node = nullptr;
  int fc_offset = 0;
  if (subgraph->num_nodes == 1) {
    fc_node = &subgraph->nodes[0];
    ASSERT_EQ(fc_node->inputs[0], input_id);
    ASSERT_EQ(subgraph->num_values, 4);
  } else {
    pack_lh_node = &subgraph->nodes[0];
    fc_offset = 1;
    fc_node = &subgraph->nodes[1];
    ASSERT_EQ(pack_lh_node->type, xnn_node_type_pack_lh);
    ASSERT_EQ(pack_lh_node->inputs[0], input_id);
    ASSERT_EQ(fc_node->inputs[0], pack_lh_node->outputs[0]);
    ASSERT_EQ(subgraph->num_values, 5);
  }
  ASSERT_EQ(fc_node->type, xnn_node_type_fully_connected);
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->num_inputs, 3);
  ASSERT_EQ(fc_node->inputs[1], kernel_id);
  ASSERT_EQ(fc_node->inputs[2], bias_id);
  ASSERT_EQ(fc_node->num_outputs, 1);
  ASSERT_EQ(fc_node->outputs[0], output_id);
  ASSERT_EQ(fc_node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  std::vector<size_t> new_input_dims(input_dims.begin(), input_dims.end());
  const xnn_shape* output_shape = &runtime->values[fc_node->outputs[0]].shape;
  const xnn_shape* kernel_shape = &runtime->values[fc_node->inputs[1]].shape;

  // case 1 : no change in input dims
  size_t prev_workspace_size = runtime->workspace->size;
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  if (pack_lh_node) {
    ASSERT_EQ(runtime->workspace->size, prev_workspace_size);
  }

  // case 2: Resize input to a smaller size. This should not require memory
  // planning
  for (size_t i = 0; i < new_input_dims.size() - 1; ++i) {
    new_input_dims[i] = input_dims[i] - 1;
  }

  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, new_input_dims.size(),
                                       new_input_dims.data()));

  prev_workspace_size = runtime->workspace->size;
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  if (pack_lh_node) {
    ASSERT_EQ(runtime->workspace->size, prev_workspace_size);
  }

  // Check that the output shape is correct
  for (size_t i = 0; i < output_shape->num_dims - 1; ++i) {
    ASSERT_EQ(output_shape->dim[i], new_input_dims[i]);
  }
  ASSERT_EQ(output_shape->dim[output_shape->num_dims - 1],
            kernel_shape->dim[0]);

  // case 3: Resize input to a larger size. This should require memory planning
  for (size_t i = 0; i < new_input_dims.size() - 1; ++i) {
    new_input_dims[i] = input_dims[i] + XNN_MAX_MR;
  }

  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, new_input_dims.size(),
                                       new_input_dims.data()));

  prev_workspace_size = runtime->workspace->size;
  size_t prev_output_size = runtime->values[fc_node->outputs[0]].size;
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  if (pack_lh_node) {
    ASSERT_GT(runtime->workspace->size, prev_workspace_size);
  }
  ASSERT_GT(runtime->values[fc_node->outputs[0]].size, prev_output_size);

  // Check that the output shape is correct
  for (size_t i = 0; i < output_shape->num_dims - 1; ++i) {
    ASSERT_EQ(output_shape->dim[i], new_input_dims[i]);
  }
  ASSERT_EQ(output_shape->dim[output_shape->num_dims - 1],
            kernel_shape->dim[0]);

  size_t num_output_elements =
      std::accumulate(new_input_dims.begin(), new_input_dims.end() - 1,
                      size_t{1}, std::multiplies<size_t>()) *
      kernel_shape->dim[0];
  ASSERT_EQ(runtime->values[fc_node->outputs[0]].size,
            num_output_elements * sizeof(float));
}

TEST_F(FullyConnectedTestF32, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

  // Call operator API.
  const xnn_status status = xnn_create_fully_connected_nc_f32(
      input_channels, output_channels, input_channels, output_channels,
      kernel.data(), bias.data(), output_min, output_max,
      /*flags=*/0, nullptr, nullptr, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_f32(
                                    op, batch_size, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_f32(
                                    op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, kernel_dims.size(),
                              kernel_dims.data(), kernel.data(),
                              /*external_id=*/1, /*flags=*/0, &kernel_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                 kernel_id, bias_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v3(subgraph, nullptr, nullptr,
                                  xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  float max_abs_output = 0.0f;
  for (const float val : operator_output) {
    max_abs_output = std::max(max_abs_output, std::abs(val));
  }
  EXPECT_THAT(subgraph_output,
              Pointwise(FloatNear(1.0e-6 * max_abs_output),
                        std::vector<float>(operator_output.begin(),
                                           operator_output.end())));
}
