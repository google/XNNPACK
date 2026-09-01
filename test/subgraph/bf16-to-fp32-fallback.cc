// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/subgraph/rewrites/cvt_to_fp32.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/subgraph.h"
#include "test/subgraph/rewrites/subgraph_matcher.h"
#include "litert/tensor/arithmetic.h"
#include "litert/tensor/backends/xnnpack/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/matchers.h"

void PrintTo(const enum xnn_node_type type, std::ostream* os) {
  *os << '"' << xnn_node_type_to_string(type) << '"';
}

void PrintTo(const struct xnn_node& node, std::ostream* os) {
  *os << "<xnn_node>";
}

void PrintTo(const enum xnn_unary_operator type, std::ostream* os) {
  *os << '"' << xnn_unary_operator_to_string(type) << '"';
}

namespace litert::tensor {
void PrintTo(const XnnpackGraph& graph, std::ostream* os) {
  PrintTo(graph.subgraph(), os);
}
}  // namespace litert::tensor

namespace {

using XnnTensor = litert::tensor::Tensor<litert::tensor::XnnpackMixinTag>;

using litert::tensor::BuildXnnpackGraph;
using litert::tensor::OwningCpuBuffer;
using litert::tensor::Type;
using litert::tensor::XnnpackGraph;
using testing::Eq;
using xnnpack::IsIsomorphicTo;

struct InlineQd8Bf16Qb4wFullyConnected {
  xnn_subgraph_t subgraph = nullptr;
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t convert_output_id = XNN_INVALID_VALUE_ID;
  uint32_t weights_id = XNN_INVALID_VALUE_ID;
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  std::vector<uint16_t> weights_scale = {0x3F80, 0x3F80};
  std::vector<uint8_t> weights_data = std::vector<uint8_t>(32, 0x88);

  void Build(enum xnn_datatype input_datatype,
             enum xnn_datatype output_datatype = xnn_datatype_bf16,
             enum xnn_datatype scale_datatype = xnn_datatype_bf16) {
    ASSERT_EQ(
        xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph),
        xnn_status_success);

    const std::vector<size_t> input_dims = {3, 32};
    const std::vector<size_t> weights_dims = {2, 32};
    const std::vector<size_t> output_dims = {3, 2};

    ASSERT_EQ(xnn_define_tensor_value(
                  subgraph, input_datatype, input_dims.size(),
                  input_dims.data(), /*data=*/nullptr, /*external_id=*/0,
                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
              xnn_status_success);
    ASSERT_EQ(
        xnn_define_dynamically_quantized_tensor_value(
            subgraph, xnn_datatype_qdint8, input_dims.size(),
            /*num_nonbatch_dims=*/1, input_dims.data(), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &convert_output_id),
        xnn_status_success);
    ASSERT_EQ(xnn_define_unary(subgraph, xnn_unary_convert,
                               /*params=*/nullptr, input_id, convert_output_id,
                               /*flags=*/0),
              xnn_status_success);
    ASSERT_EQ(
        xnn_define_blockwise_quantized_tensor_value_v2(
            subgraph, xnn_datatype_qbint4, /*zero_point=*/8,
            weights_scale.data(), weights_dims.size(), /*channel_dim=*/0,
            /*block_size=*/32, weights_dims.data(), weights_data.data(),
            XNN_INVALID_VALUE_ID, /*flags=*/0, scale_datatype, &weights_id),
        xnn_status_success);
    ASSERT_EQ(xnn_define_tensor_value(
                  subgraph, output_datatype, output_dims.size(),
                  output_dims.data(), /*data=*/nullptr, /*external_id=*/1,
                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
              xnn_status_success);
    ASSERT_EQ(xnn_define_fully_connected(
                  subgraph, -INFINITY, INFINITY, convert_output_id, weights_id,
                  XNN_INVALID_VALUE_ID, output_id, /*flags=*/0),
              xnn_status_success);

    ASSERT_EQ(subgraph->num_nodes, 2);
    struct xnn_node& fully_connected = subgraph->nodes[1];
    ASSERT_EQ(fully_connected.type, xnn_node_type_fully_connected);
    // Match the graph state produced by packed-LHS fusion while retaining the
    // original input datatype used by the inline packer.
    fully_connected.inputs[0] = input_id;
    fully_connected.flags |= XNN_FLAG_INLINE_LHS_PACKING;
    fully_connected.packed_input_datatype = xnn_datatype_qdint8;
  }
};

const struct xnn_node* FindFullyConnected(const xnn_subgraph_t subgraph) {
  for (size_t i = 0; i < subgraph->num_nodes; i++) {
    if (subgraph->nodes[i].type == xnn_node_type_fully_connected) {
      return &subgraph->nodes[i];
    }
  }
  return nullptr;
}

class Bf16ToFp32FallbackTest : public testing::Test {
 public:
  void SetUp() override {
    // Use an empty config to disable all bf16 support (including the bf16
    // GEMM), so every bf16 op falls back to fp32.
    xnn_set_hardware_config(&mock_config_);
  }

  void TearDown() override { xnn_reset_hardware_config(); }

  xnn_hardware_config mock_config_{};
};

TEST_F(Bf16ToFp32FallbackTest, OpChainRewrite) {
  // - An op chain rewrite should add convert bf16 operations to fp32 and insert
  //   conversions from bf16 inputs and to bf16 outputs.
  // - The intermediate values should stay as fp32.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Abs(a);
    a = Sqrt(a);
    XnnTensor b({.type = Type::kBF16, .shape = {3, 4}});
    a = Add(a, b);
    XnnTensor c({.type = Type::kBF16, .shape = {3, 4}});
    a = Mul(a, c);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor b({.type = Type::kBF16, .shape = {3, 4}});
    b = Cast(b, Type::kFP32);
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Sqrt(a);
    a = Add(a, b);
    XnnTensor c({.type = Type::kBF16, .shape = {3, 4}});
    c = Cast(c, Type::kFP32);
    a = Mul(a, c);
    a = Cast(a, Type::kBF16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Bf16ToFp32FallbackTest, NativeUnaryOpsStayBf16) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Square(a);
    a = Rsqrt(a);
    a = Logistic(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Square(a);
    a = Rsqrt(a);
    a = Logistic(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(
                  graph->subgraph(), /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Bf16ToFp32FallbackTest, ReshapeAllowsBf16Inputs) {
  // Reshape is transparent: if its inputs are bf16, it isn't rewritten.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Reshape(a, {6, 2});
    a = Abs(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Reshape(a, {6, 2});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Cast(a, Type::kBF16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Bf16ToFp32FallbackTest, ReshapeHandlesRewrittenInputs) {
  // Reshape is transparent: if its inputs have been converted from bf16 to
  // fp32, it is rewritten to output fp32.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Abs(a);
    a = Reshape(a, {6, 2});
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Reshape(a, {6, 2});
    a = Cast(a, Type::kBF16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Bf16ToFp32FallbackTest, DontInsertConvertFp32Fp32) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Abs(a);
    a = Cast(a, Type::kFP32);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Bf16ToFp32FallbackTest, ReuseConvertedFp32ValueForMultipleConsumers) {
  // If a bf16 input is consumed by multiple rewritten ops, the convert node to
  // fp32 should be reused.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    XnnTensor b = Abs(a);
    XnnTensor c = Sqrt(a);
    XnnTensor d = Add(b, c);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({d}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    XnnTensor b = Abs(a);
    XnnTensor c = Sqrt(a);
    XnnTensor d = Add(b, c);
    d = Cast(d, Type::kBF16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({d}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Bf16ToFp32FallbackTest, BinaryRewrite) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    XnnTensor b({.type = Type::kBF16, .shape = {3, 4}});
    XnnTensor output = Add(a, b);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kBF16, .shape = {3, 4}});
    XnnTensor a_fp32 = Cast(a, Type::kFP32);
    XnnTensor b({.type = Type::kBF16, .shape = {3, 4}});
    XnnTensor b_fp32 = Cast(b, Type::kFP32);
    XnnTensor output_fp32 = Add(a_fp32, b_fp32);
    XnnTensor output = Cast(output_fp32, Type::kBF16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

class Bf16ToFp32OptimizedQd8Test : public testing::Test {
 public:
  void SetUp() override {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ENABLE_ARM_DOTPROD
    mock_config_.arch_flags |= xnn_arch_arm_neon_dot;
#else
    GTEST_SKIP();
#endif
#else
    GTEST_SKIP();
#endif
    xnn_set_hardware_config(&mock_config_);
  }

  void TearDown() override { xnn_reset_hardware_config(); }

  xnn_hardware_config mock_config_{};
};

TEST_F(Bf16ToFp32OptimizedQd8Test, KeepsBf16InputNative) {
  InlineQd8Bf16Qb4wFullyConnected builder;
  builder.Build(xnn_datatype_bf16);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(
                  builder.subgraph, /*optimization_flags=*/0),
              Eq(xnn_status_success));

  const struct xnn_node* fully_connected =
      FindFullyConnected(builder.subgraph);
  ASSERT_NE(fully_connected, nullptr);
  EXPECT_EQ(builder.subgraph->values[fully_connected->inputs[0]].datatype,
            xnn_datatype_bf16);
  EXPECT_EQ(builder.subgraph->values[fully_connected->outputs[0]].datatype,
            xnn_datatype_bf16);
  EXPECT_EQ(fully_connected->outputs[0], builder.output_id);
}

TEST_F(Bf16ToFp32OptimizedQd8Test, RewritesFp32InputForSafety) {
  InlineQd8Bf16Qb4wFullyConnected builder;
  builder.Build(xnn_datatype_fp32);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(
                  builder.subgraph, /*optimization_flags=*/0),
              Eq(xnn_status_success));

  const struct xnn_node* fully_connected =
      FindFullyConnected(builder.subgraph);
  ASSERT_NE(fully_connected, nullptr);
  EXPECT_EQ(builder.subgraph->values[fully_connected->inputs[0]].datatype,
            xnn_datatype_fp32);
  EXPECT_EQ(builder.subgraph->values[fully_connected->outputs[0]].datatype,
            xnn_datatype_fp32);
  EXPECT_EQ(builder.subgraph->values[builder.output_id].datatype,
            xnn_datatype_bf16);
  EXPECT_NE(fully_connected->outputs[0], builder.output_id);
}

TEST_F(Bf16ToFp32OptimizedQd8Test, RewritesFp32OutputForSafety) {
  InlineQd8Bf16Qb4wFullyConnected builder;
  builder.Build(xnn_datatype_bf16, xnn_datatype_fp32);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(
                  builder.subgraph, /*optimization_flags=*/0),
              Eq(xnn_status_success));

  const struct xnn_node* fully_connected =
      FindFullyConnected(builder.subgraph);
  ASSERT_NE(fully_connected, nullptr);
  EXPECT_EQ(builder.subgraph->values[fully_connected->inputs[0]].datatype,
            xnn_datatype_fp32);
  EXPECT_EQ(builder.subgraph->values[fully_connected->outputs[0]].datatype,
            xnn_datatype_fp32);
}

TEST_F(Bf16ToFp32OptimizedQd8Test, RewritesFp16WeightScalesForSafety) {
  InlineQd8Bf16Qb4wFullyConnected builder;
  builder.Build(xnn_datatype_bf16, xnn_datatype_bf16, xnn_datatype_fp16);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(
                  builder.subgraph, /*optimization_flags=*/0),
              Eq(xnn_status_success));

  const struct xnn_node* fully_connected =
      FindFullyConnected(builder.subgraph);
  ASSERT_NE(fully_connected, nullptr);
  EXPECT_EQ(builder.subgraph->values[fully_connected->inputs[0]].datatype,
            xnn_datatype_fp32);
  EXPECT_EQ(builder.subgraph->values[fully_connected->outputs[0]].datatype,
            xnn_datatype_fp32);
}

TEST_F(Bf16ToFp32FallbackTest, RewritesScalarQd8Fallback) {
  InlineQd8Bf16Qb4wFullyConnected builder;
  builder.Build(xnn_datatype_bf16);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(
                  builder.subgraph, /*optimization_flags=*/0),
              Eq(xnn_status_success));

  const struct xnn_node* fully_connected =
      FindFullyConnected(builder.subgraph);
  ASSERT_NE(fully_connected, nullptr);
  EXPECT_EQ(builder.subgraph->values[fully_connected->inputs[0]].datatype,
            xnn_datatype_fp32);
  EXPECT_EQ(builder.subgraph->values[fully_connected->outputs[0]].datatype,
            xnn_datatype_fp32);
  EXPECT_EQ(builder.subgraph->values[builder.output_id].datatype,
            xnn_datatype_bf16);
  EXPECT_NE(fully_connected->outputs[0], builder.output_id);
}

}  // namespace
