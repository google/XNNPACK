// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

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

}  // namespace
