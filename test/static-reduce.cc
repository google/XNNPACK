// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/common.h"
#include "xnnpack/datatype.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

struct Param {
  using TupleT = std::tuple<xnn_datatype, xnn_reduce_operator, bool>;
  explicit Param(TupleT p)
      : datatype(std::get<0>(p)),
        reduce_operator(std::get<1>(p)),
        keep_dims(std::get<2>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    switch (reduce_operator) {
      case xnn_reduce_mean:
        sstr << "mean";
        break;
      case xnn_reduce_sum:
        sstr << "sum";
        break;
      case xnn_reduce_invalid:
        sstr << "invalid";
        break;
    }
    sstr << "_" << xnn_datatype_to_string(datatype);
    if (keep_dims) {
      sstr << "_keep_dims";
    }
    return sstr.str();
  }

  xnn_datatype datatype;
  xnn_reduce_operator reduce_operator;
  bool keep_dims;
};

namespace {
constexpr xnn_compute_type GetComputeType(xnn_datatype t) {
  switch (t) {
    case xnn_datatype_fp16:
      return xnn_compute_type_fp16;
    case xnn_datatype_fp32:
      return xnn_compute_type_fp32;
    case xnn_datatype_qint8:
      return xnn_compute_type_qs8;
    case xnn_datatype_quint8:
      return xnn_compute_type_qu8;
    default:
      XNN_UNREACHABLE;
  }
}
}  // namespace

namespace xnnpack {
template <class T>
class ReduceTestBase : public ::testing::TestWithParam<Param> {
 protected:
  void SetUp() override {
    const Param p = GetParam();
    auto num_input_dim_dist =
        std::uniform_int_distribution<size_t>(2, XNN_MAX_TENSOR_DIMS);
    const size_t num_input_dims = num_input_dim_dist(rng);
    auto num_reduction_axes_dist =
        std::uniform_int_distribution<size_t>(1, num_input_dims);
    const size_t num_reduction_axes = num_reduction_axes_dist(rng);

    auto axes_dist =
        std::uniform_int_distribution<size_t>(0, num_input_dims - 1);
    reduction_axes.resize(num_reduction_axes);
    std::generate(reduction_axes.begin(), reduction_axes.end(),
                  [&]() { return axes_dist(rng); });
    std::sort(reduction_axes.begin(), reduction_axes.end());
    auto end = std::unique(reduction_axes.begin(), reduction_axes.end());
    reduction_axes.erase(end, reduction_axes.end());

    auto shape_dist = std::uniform_int_distribution<size_t>(2, 15);
    input_shape.resize(num_input_dims);
    std::generate(input_shape.begin(), input_shape.end(),
                  [&]() { return shape_dist(rng); });
    num_input_elements =
        std::accumulate(input_shape.cbegin(), input_shape.cend(), size_t(1),
                        std::multiplies<size_t>());

    output_shape = input_shape;
    for (size_t axis : reduction_axes) {
      output_shape[axis] = 1;
    }
    num_output_elements =
        std::accumulate(output_shape.cbegin(), output_shape.cend(), size_t(1),
                        std::multiplies<size_t>());

    input = xnnpack::Buffer<char>(XNN_EXTRA_BYTES / sizeof(char) +
                                  num_input_elements * xnn_datatype_size_bytes(p.datatype));
    operator_output =
        xnnpack::Buffer<char>(num_output_elements * xnn_datatype_size_bytes(p.datatype));
    subgraph_output =
        xnnpack::Buffer<char>(num_output_elements * xnn_datatype_size_bytes(p.datatype));
  }

  struct QuantizationParams {
    float input_scale;
    float output_scale;
    int32_t input_zero_point;
    int32_t output_zero_point;

    constexpr bool IsQuantized() const { return input_scale != 0; }
  };

  QuantizationParams RandomQuantizationParams(xnn_datatype t) {
    QuantizationParams qp;
    switch (t) {
      case xnn_datatype_qint8:
        qp.input_scale = scale_dist(rng);
        qp.output_scale = scale_dist(rng);
        qp.input_zero_point = i8dist(rng);
        qp.output_zero_point = i8dist(rng);
        break;
      case xnn_datatype_quint8:
        qp.input_scale = scale_dist(rng);
        qp.output_scale = scale_dist(rng);
        qp.input_zero_point = u8dist(rng);
        qp.output_zero_point = u8dist(rng);
        break;
      default:
        qp.input_scale = 0;
        qp.output_scale = 0;
        qp.input_zero_point = 0;
        qp.output_zero_point = 0;
    }
    return qp;
  }

  void SetUpInputOutput(xnn_subgraph_t subgraph, const QuantizationParams& qp,
                        uint32_t& input_id, uint32_t& output_id) {
    const Param p = GetParam();
    if (qp.IsQuantized()) {
      ASSERT_EQ(xnn_status_success,
                xnn_define_quantized_tensor_value(
                    subgraph, p.datatype, qp.input_zero_point, qp.input_scale,
                    input_shape.size(), input_shape.data(), nullptr,
                    /*external_id=*/0,
                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
      ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

      ASSERT_EQ(xnn_status_success,
                xnn_define_quantized_tensor_value(
                    subgraph, p.datatype, qp.output_zero_point, qp.output_scale,
                    output_shape.size(), output_shape.data(), nullptr,
                    /*external_id=*/1,
                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
      ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
    } else {
      ASSERT_EQ(xnn_status_success,
                xnn_define_tensor_value(
                    subgraph, p.datatype, input_shape.size(),
                    input_shape.data(), nullptr,
                    /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &input_id));
      ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

      ASSERT_EQ(xnn_status_success,
                xnn_define_tensor_value(
                    subgraph, p.datatype, output_shape.size(),
                    output_shape.data(), nullptr, /*external_id=*/1,
                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
      ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
    }
  }

  template <class Datatype, class Dist>
  void GenerateRandomInput(Dist& dist) {
    Datatype* beg = reinterpret_cast<Datatype*>(input.data());
    Datatype* end = reinterpret_cast<Datatype*>(input.data() + input.size());
    std::generate(beg, end, [&]() { return dist(rng); });
  }

  void GenerateRandomInput(xnn_datatype t) {
    switch (t) {
      case xnn_datatype_fp16:
        GenerateRandomInput<xnn_float16>(f32dist);
        break;
      case xnn_datatype_fp32:
        GenerateRandomInput<float>(f32dist);
        break;
      case xnn_datatype_qint8:
        GenerateRandomInput<int8_t>(i8dist);
        break;
      case xnn_datatype_quint8:
        GenerateRandomInput<uint8_t>(u8dist);
        break;
      default:
        XNN_UNREACHABLE;
    }
  }

  template <class Datatype>
  void CompareOutputsImpl() {
    const Datatype* subgraph_out_ptr =
        reinterpret_cast<const Datatype*>(subgraph_output.data());
    const Datatype* operator_out_ptr =
        reinterpret_cast<const Datatype*>(operator_output.data());
    const size_t output_size = subgraph_output.size() / sizeof(Datatype);
    for (size_t i = 0; i < output_size;
         i++, ++subgraph_out_ptr, ++operator_out_ptr) {
      const Datatype sub_out = *subgraph_out_ptr;
      const Datatype op_out = *operator_out_ptr;
      ASSERT_NEAR(sub_out, op_out, std::abs(0.05f * std::min(sub_out, op_out)));
    }
  }

  void CompareOutputs(xnn_datatype t) {
    switch (t) {
      case xnn_datatype_fp16:
        CompareOutputsImpl<xnn_float16>();
        break;
      case xnn_datatype_fp32:
        CompareOutputsImpl<float>();
        break;
      case xnn_datatype_qint8:
        CompareOutputsImpl<int8_t>();
        break;
      case xnn_datatype_quint8:
        CompareOutputsImpl<uint8_t>();
        break;
      default:
        XNN_UNREACHABLE;
    }
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> scale_dist =
      std::uniform_real_distribution<float>(0.0f, 1.0f);
  std::uniform_real_distribution<float> f32dist =
      std::uniform_real_distribution<float>(-1.0f, 1.0f);
  std::uniform_int_distribution<int32_t> i8dist =
      std::uniform_int_distribution<int32_t>(
          std::numeric_limits<int8_t>::min(),
          std::numeric_limits<int8_t>::max());
  std::uniform_int_distribution<int32_t> u8dist =
      std::uniform_int_distribution<int32_t>(
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max());

  std::vector<size_t> reduction_axes;
  std::vector<size_t> input_shape;
  size_t num_input_elements;
  std::vector<size_t> output_shape;
  size_t num_output_elements;

  xnnpack::Buffer<char> input;
  xnnpack::Buffer<char> operator_output;
  xnnpack::Buffer<char> subgraph_output;
};

using ReduceTest = ReduceTestBase<void>;

using ReduceTestF16 = ReduceTestBase<xnn_float16>;
using ReduceTestF32 = ReduceTestBase<float>;
using ReduceTestQS8 = ReduceTestBase<int8_t>;
using ReduceTestQU8 = ReduceTestBase<uint8_t>;

using ::testing::Bool;
using ::testing::Combine;
using ::testing::Values;

INSTANTIATE_TEST_SUITE_P(ReduceTest, ReduceTest,
                         testing::ConvertGenerator<Param::TupleT>(Combine(
                             Values(xnn_datatype_fp16, xnn_datatype_fp32,
                                    xnn_datatype_qint8, xnn_datatype_quint8),
                             Values(xnn_reduce_sum, xnn_reduce_mean), Bool())),
                         [](auto p) { return p.param.Name(); });

TEST_P(ReduceTest, define) {
  const Param p = GetParam();
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  uint32_t output_id = XNN_INVALID_NODE_ID;

  SetUpInputOutput(subgraph, RandomQuantizationParams(p.datatype), input_id,
                   output_id);

  ASSERT_EQ(xnn_status_success,
            xnn_define_static_reduce(
                subgraph, p.reduce_operator, reduction_axes.size(),
                reduction_axes.data(), input_id, output_id,
                /*flags=*/p.keep_dims ? XNN_FLAG_KEEP_DIMS : 0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_reduce_operator_to_node_type(p.reduce_operator));
  ASSERT_EQ(node->compute_type, GetComputeType(p.datatype));
  ASSERT_EQ(node->params.reduce.num_reduction_axes, reduction_axes.size());
  for (size_t i = 0; i < reduction_axes.size(); i++) {
    ASSERT_EQ(node->params.reduce.reduction_axes[i], reduction_axes[i]);
  }
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, p.keep_dims ? XNN_FLAG_KEEP_DIMS : 0);
}

TEST_P(ReduceTest, matches_operator_api) {
  const Param p = GetParam();
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  GenerateRandomInput(p.datatype);

  const uint32_t flags = p.keep_dims ? XNN_FLAG_KEEP_DIMS : 0;
  const QuantizationParams qp = RandomQuantizationParams(p.datatype);
  // Call operator API.
  const float scale =
      qp.output_scale == 0 ? 0 : qp.input_scale / qp.output_scale;
  const xnn_status status = xnn_create_reduce_nd(
      p.reduce_operator, p.datatype, scale, qp.input_zero_point,
      qp.output_zero_point, flags, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_reduce_nd(op, p.datatype, reduction_axes.size(),
                                  reduction_axes.data(), input_shape.size(),
                                  input_shape.data(), &workspace_size,
                                  &workspace_alignment,
                                  /*threadpool=*/nullptr));

  ASSERT_NE(workspace_size, SIZE_MAX);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace;
  void* workspace_ptr = nullptr;
  if (p.datatype != xnn_datatype_fp32) {
    workspace = xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT>(workspace_size);
    workspace_ptr = workspace.data();
  }
  ASSERT_EQ(xnn_status_success,
            xnn_setup_reduce_nd(op, workspace_ptr, input.data(),
                                operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  uint32_t output_id = XNN_INVALID_NODE_ID;

  int output_num_dims = input_shape.size();
  if (!p.keep_dims) {
    output_num_dims -= reduction_axes.size();
  }
  if (qp.IsQuantized()) {
    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, p.datatype, qp.input_zero_point, qp.input_scale,
                  input_shape.size(), input_shape.data(), nullptr,
                  /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
    ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

    ASSERT_EQ(
        xnn_status_success,
        xnn_define_quantized_tensor_value(
            subgraph, p.datatype, qp.output_zero_point, qp.output_scale,
            output_shape.size(), output_shape.data(), nullptr,
            /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
    ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  } else {
    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, p.datatype, input_shape.size(),
                                input_shape.data(), nullptr, /*external_id=*/0,
                                XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
    ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, p.datatype, output_num_dims,
                                output_shape.data(), nullptr, /*external_id=*/1,
                                XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
    ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(xnn_status_success,
            xnn_define_static_reduce(
                subgraph, p.reduce_operator, reduction_axes.size(),
                reduction_axes.data(), input_id, output_id, flags));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  const std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  CompareOutputs(p.datatype);
}

TEST_P(ReduceTest, reshape) {
  const Param p = GetParam();
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  GenerateRandomInput(p.datatype);

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  uint32_t output_id = XNN_INVALID_NODE_ID;
  QuantizationParams qp = RandomQuantizationParams(p.datatype);
  const int output_num_dims = p.keep_dims
                                  ? output_shape.size()
                                  : input_shape.size() - reduction_axes.size();
  if (qp.IsQuantized()) {
    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, p.datatype, qp.input_zero_point, qp.input_scale,
                  input_shape.size(), input_shape.data(), nullptr,
                  /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
    ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

    ASSERT_EQ(
        xnn_status_success,
        xnn_define_quantized_tensor_value(
            subgraph, p.datatype, qp.output_zero_point, qp.output_scale,
            output_num_dims, output_shape.data(), nullptr, /*external_id=*/1,
            XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
    ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  } else {
    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, p.datatype, input_shape.size(),
                                input_shape.data(), nullptr, /*external_id=*/0,
                                XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
    ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

    ASSERT_EQ(
        xnn_status_success,
        xnn_define_tensor_value(subgraph, p.datatype, output_num_dims,
                                output_shape.data(), nullptr, /*external_id=*/1,
                                XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
    ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(xnn_define_static_reduce(
                subgraph, p.reduce_operator, reduction_axes.size(),
                reduction_axes.data(), input_id, output_id,
                /*flags=*/p.keep_dims ? XNN_FLAG_KEEP_DIMS : 0),
            xnn_status_success);

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(status, xnn_status_success);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  const std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  input_shape[0] += 2;
  input_shape[1] += 4;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_shape.size(),
                                       input_shape.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  std::vector<size_t> unique_reduction_axes = reduction_axes;
  std::sort(unique_reduction_axes.begin(), unique_reduction_axes.end());
  auto end =
      std::unique(unique_reduction_axes.begin(), unique_reduction_axes.end());
  unique_reduction_axes.erase(end, unique_reduction_axes.end());
  // There are too many parameters which influence the workspace size so
  // knowing if reallocation is required or not is messy.
  node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values,
                /*threadpool=*/nullptr);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  size_t current_axes = 0;
  size_t current_dim = 0;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (unique_reduction_axes[current_axes] == i) {
      if (p.keep_dims) {
        ASSERT_EQ(output_shape->dim[current_dim], 1);
        ++current_dim;
      }
      ++current_axes;
      if (current_axes == unique_reduction_axes.size()) {
        break;
      }
    } else {
      ASSERT_EQ(output_shape->dim[current_dim], input_shape[i]);
      ++current_dim;
    }
  }

  input_shape[0] -= 1;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_shape.size(),
                                       input_shape.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_success);
  current_axes = 0;
  current_dim = 0;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (unique_reduction_axes[current_axes] == i) {
      if (p.keep_dims) {
        ASSERT_EQ(output_shape->dim[current_dim], 1);
        ++current_dim;
      }
      ++current_axes;
      if (current_axes == unique_reduction_axes.size()) {
        break;
      }
    } else {
      ASSERT_EQ(output_shape->dim[current_dim], input_shape[i]);
      ++current_dim;
    }
  }
}

}  // namespace xnnpack
