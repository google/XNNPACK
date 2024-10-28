// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/datatype.h"
#include "xnnpack/buffer.h"
#include "xnnpack/log.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"
#include "unary-ops.h"

struct Param {
  using UnaryT = std::tuple<xnn_unary_operator, xnn_datatype>;
  using ConvertT = std::tuple<xnn_unary_operator, xnn_datatype, xnn_datatype>;
  explicit Param(UnaryT p)
      : unary_operator(std::get<0>(p)),
        input_datatype(std::get<1>(p)),
        output_datatype(std::get<1>(p)) {}
  explicit Param(ConvertT p)
      : unary_operator(std::get<0>(p)),
        input_datatype(std::get<1>(p)),
        output_datatype(std::get<2>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    sstr << xnn_unary_operator_to_string(unary_operator) << "_"
         << xnn_datatype_to_string(input_datatype);
    if (input_datatype != output_datatype) {
      sstr << "_" << xnn_datatype_to_string(output_datatype);
    }
    std::string s = sstr.str();
    // Test names must be alphanumeric with no spaces
    std::replace(s.begin(), s.end(), ' ', '_');
    std::replace(s.begin(), s.end(), '(', '_');
    std::replace(s.begin(), s.end(), ')', '_');
    return s;
  }

  xnn_unary_operator unary_operator;
  xnn_datatype input_datatype;
  xnn_datatype output_datatype;
};

class UnaryTest : public testing::TestWithParam<Param> {
 public:
  xnnpack::ReplicableRandomDevice rng_;
};

TEST_P(UnaryTest, matches_operator_api) {
  const xnn_unary_operator unary_operator = GetParam().unary_operator;
  const xnn_datatype input_datatype = GetParam().input_datatype;
  const xnn_datatype output_datatype = GetParam().output_datatype;

  const size_t sizeof_input = xnn_datatype_size_bytes(input_datatype);
  const size_t sizeof_output = xnn_datatype_size_bytes(output_datatype);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  std::uniform_int_distribution<> rank_dist(0, XNN_MAX_TENSOR_DIMS);
  std::uniform_int_distribution<> dim_dist(1, 10);
  std::vector<size_t> dims(rank_dist(rng_));
  std::generate(dims.begin(), dims.end(), [&]() { return dim_dist(rng_); });

  size_t size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
  size_t channels = dims.empty() ? 1 : dims.back();
  size_t batch_size = size / channels;

  xnnpack::Buffer<char> input(size * sizeof_input + XNN_EXTRA_BYTES);
  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng_);

  xnnpack::Buffer<char> subgraph_output(size * sizeof_output);
  xnnpack::Buffer<char> operator_output(size * sizeof_output);

  const UnaryOpInfo* op_info = GetUnaryOpInfo(unary_operator);
  xnn_unary_params params = op_info->DefaultParams();
  const xnn_quantization_params input_quantization =
      op_info->InputQuantizationParams(input_datatype);
  const xnn_quantization_params output_quantization =
      op_info->OutputQuantizationParams(output_datatype);

  // Call operator API.
  const xnn_status status = xnn_run_unary_elementwise_nc(
      unary_operator, input_datatype, output_datatype, &params,
      &input_quantization, &output_quantization, /*flags=*/0, batch_size,
      channels, channels, channels, /*thread_pool=*/nullptr, input.data(),
      operator_output.data());
  if (status == xnn_status_unsupported_parameter) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  if (xnn_datatype_is_quantized(input_datatype)) {
    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, input_datatype, input_quantization.zero_point,
                  input_quantization.scale, dims.size(), dims.data(), nullptr,
                  /*external_id=*/0,
                  /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  } else {
    ASSERT_EQ(xnn_status_success,
              xnn_define_tensor_value(subgraph, input_datatype, dims.size(),
                                      dims.data(), nullptr, /*external_id=*/0,
                                      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                      &input_id));
  }
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  if (xnn_datatype_is_quantized(output_datatype)) {
    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, output_datatype, output_quantization.zero_point,
                  output_quantization.scale, dims.size(), dims.data(), nullptr,
                  /*external_id=*/1,
                  /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  } else {
    ASSERT_EQ(xnn_status_success,
              xnn_define_tensor_value(subgraph, output_datatype, dims.size(),
                                      dims.data(), nullptr, /*external_id=*/1,
                                      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                      &output_id));
  }
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_unary(subgraph, unary_operator, &params, input_id,
                             output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

xnn_unary_operator all_unary_ops[] = {
    xnn_unary_clamp,
    xnn_unary_abs,
    xnn_unary_bankers_rounding,
    xnn_unary_ceiling,
    xnn_unary_elu,
    xnn_unary_exp,
    xnn_unary_floor,
    xnn_unary_gelu,
    xnn_unary_hardswish,
    xnn_unary_leaky_relu,
    xnn_unary_log,
    xnn_unary_negate,
    xnn_unary_sigmoid,
    xnn_unary_square,
    xnn_unary_square_root,
    xnn_unary_reciprocal_square_root,
    xnn_unary_tanh,
};

xnn_datatype all_datatypes[] = {
    xnn_datatype_quint8, xnn_datatype_qint8, xnn_datatype_fp16,
    xnn_datatype_fp32,   xnn_datatype_int32,
};

INSTANTIATE_TEST_SUITE_P(
    UnaryTest, UnaryTest,
    testing::ConvertGenerator<Param::UnaryT>(testing::Combine(
        testing::ValuesIn(all_unary_ops), testing::ValuesIn(all_datatypes))),
    [](const auto& info) { return info.param.Name(); });

INSTANTIATE_TEST_SUITE_P(
    ConvertTest, UnaryTest,
    testing::ConvertGenerator<Param::ConvertT>(testing::Combine(
        testing::Values(xnn_unary_convert), testing::ValuesIn(all_datatypes),
        testing::ValuesIn(all_datatypes))),
    [](const auto& info) { return info.param.Name(); });


TEST(AbsTest, reshape) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> dims{2, 3, 4};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_unary(subgraph, xnn_unary_abs, /*params=*/nullptr, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_unary_elementwise);
  ASSERT_EQ(node->unary_operator, xnn_unary_abs);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

  dims[0] = 7;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, 0, dims.size(), dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  const size_t num_input_elements = std::accumulate(dims.cbegin(), dims.cend(), size_t{1}, std::multiplies<size_t>());
  ASSERT_EQ(output_shape->dim[0], dims[0]);
  ASSERT_EQ(runtime->values[node->outputs[0]].size, num_input_elements * sizeof(float));
}
