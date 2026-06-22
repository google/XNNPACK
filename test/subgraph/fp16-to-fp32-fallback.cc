// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functional>
#include <memory>
#include <ostream>
#include <string>
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

class Fp16ToFp32FallbackTest : public testing::Test {
 public:
  void SetUp() override {
    // Use an empty config to disable FP16 support.
    xnn_set_hardware_config(&mock_config_);
  }

  void TearDown() override { xnn_reset_hardware_config(); }

  xnn_hardware_config mock_config_{};
};

TEST_F(Fp16ToFp32FallbackTest, SingleOpRewrite) {
  // A single op rewrite should add convert nodes for the fp32 inputs (to fp32)
  // and outputs (from fp32) that are fp16.
  //
  // Before:
  //
  //                         ┌─────────────────────────┐
  //                 (input) │     000: FP16[3, 4]     │
  //                         └─────────────────────────┘
  //                           │
  //                           │
  //      (weights)            ▼
  // ┌─────────────────┐     ┌─────────────────────────┐
  // │ 002: FP16[2, 4] │     │  #000: Fully Connected  │
  // │                 │ ──▶ │ (FP16, FP16, FP16, goi) │
  // └─────────────────┘     └─────────────────────────┘
  //                           │
  //                           │
  //                           ▼
  //                         ┌─────────────────────────┐
  //                         │     001: FP16[3, 2]     │
  //                         └─────────────────────────┘
  //
  //
  // After:
  //                         ┌─────────────────────────┐
  //                 (input) │     000: FP16[3, 4]     │
  //                         └─────────────────────────┘
  //                           │
  //                           │
  //                           ▼
  //                         ┌─────────────────────────┐
  //                         │ #000: Unary Elementwise │
  //                         │     (convert, FP16)     │
  //                         └─────────────────────────┘
  //                           │
  // (statically converted     │ v004: FP32[3, 4]
  //       weights)            ▼
  // ┌─────────────────┐     ┌─────────────────────────┐
  // │ 005: FP32[2, 4] │     │  #001: Fully Connected  │
  // │                 │ ──▶ │ (FP32, FP32, FP32, goi) │
  // └─────────────────┘     └─────────────────────────┘
  //                           │
  //                           │ v003: FP32[3, 2]
  //                           ▼
  //                         ┌─────────────────────────┐
  //                         │ #002: Unary Elementwise │
  //                         │     (convert, FP32)     │
  //                         └─────────────────────────┘
  //                           │
  //                           │
  //                           ▼
  //                         ┌─────────────────────────┐
  //                         │     001: FP16[3, 2]     │
  //                         └─────────────────────────┘

  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor weights({.type = Type::kFP16,
                       .shape = {2, 4},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                           {1, 2, 3, 4, 5, 6, 7, 8})});
    XnnTensor output = FullyConnected(input, weights);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    input = Cast(input, Type::kFP32);
    XnnTensor weights({.type = Type::kFP32,
                       .shape = {2, 4},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                           {1, 2, 3, 4, 5, 6, 7, 8})});
    XnnTensor output = FullyConnected(input, weights);
    output = Cast(output, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }
  xnn_subgraph_t subgraph = graph->subgraph();

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(subgraph,
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, OpChainRewrite) {
  // - An op chain rewrite should add convert fp16 operations to fp32 and insert
  //   conversions from fp16 inputs and to fp16 outputs.
  // - The intermediate values should stay as fp32.

  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Abs(a);
    a = Sqrt(a);
    XnnTensor b({.type = Type::kFP16, .shape = {3, 4}});
    a = Add(a, b);
    XnnTensor c({.type = Type::kFP16, .shape = {3, 4}});
    a = Mul(a, c);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor b({.type = Type::kFP16, .shape = {3, 4}});
    b = Cast(b, Type::kFP32);
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Sqrt(a);
    a = Add(a, b);
    XnnTensor c({.type = Type::kFP16, .shape = {3, 4}});
    c = Cast(c, Type::kFP32);
    a = Mul(a, c);
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, ReshapeAllowsFp16Inputs) {
  // Reshape is transparent: if its inputs are fp16, it isn't rewritten.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Reshape(a, {6, 2});
    a = Abs(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Reshape(a, {6, 2});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, ReshapeHandlesRewrittenInputs) {
  // Reshape is transparent: if its inputs have been converted from fp16 to
  // fp32, it is rewritten to output fp32.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Abs(a);
    a = Reshape(a, {6, 2});
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Reshape(a, {6, 2});
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, DontInsertConvertFp32Fp32) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Abs(a);
    a = Cast(a, Type::kFP32);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, Fp16ToFp16HandleExternalInput) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP16);
    a = Abs(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, Fp16ToFp16HandleExternalOutput) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Abs(a);
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest,
       DontElideConvertFp16ToFp16WhenExternalInputToExternalOutput) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, HandleExternalOutputThatIsReused) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Abs(a);
    XnnTensor b = Cast(a, Type::kFP32);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a, b}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    XnnTensor b = Abs(a);
    a = Cast(b, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a, b}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, TransposeAllowsFp16Inputs) {
  // Transpose is transparent: if its inputs are fp16, it isn't rewritten.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Transpose(a, {1, 0});
    a = Abs(a);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Transpose(a, {1, 0});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, TransposeHandlesRewrittenInputs) {
  // Transpose is transparent: if its inputs have been converted from fp16 to
  // fp32, it is rewritten to output fp32.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Abs(a);
    a = Transpose(a, {1, 0});
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({a}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    a = Abs(a);
    a = Transpose(a, {1, 0});
    a = Cast(a, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({a}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, ReuseConvertedFp32ValueForMultipleConsumers) {
  // If an fp16 input is consumed by multiple rewritten ops, the convert node to
  // fp32 should be reused.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor b = Abs(a);
    XnnTensor c = Sqrt(a);
    XnnTensor d = Add(b, c);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({d}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    a = Cast(a, Type::kFP32);
    XnnTensor b = Abs(a);
    XnnTensor c = Sqrt(a);
    XnnTensor d = Add(b, c);
    d = Cast(d, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph, BuildXnnpackGraph({d}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, SplitAllowsFp16Inputs) {
  // Split is transparent: if its inputs are fp16, it isn't rewritten.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {2, 4}});
    XnnTensor axis({.type = Type::kI32,
                    .shape = {1},
                    .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});
    std::vector<XnnTensor> outputs = Split(input, axis, 2);
    XnnTensor out0 = Abs(outputs[0]);
    XnnTensor out1 = Abs(outputs[1]);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({out0, out1}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {2, 4}});
    XnnTensor axis({.type = Type::kI32,
                    .shape = {1},
                    .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});

    std::vector<XnnTensor> outputs = Split(input, axis, 2);

    XnnTensor out0 = Cast(outputs[0], Type::kFP32);
    out0 = Abs(out0);
    out0 = Cast(out0, Type::kFP16);

    XnnTensor out1 = Cast(outputs[1], Type::kFP32);
    out1 = Abs(out1);
    out1 = Cast(out1, Type::kFP16);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({out0, out1}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, SplitHandlesRewrittenInputs) {
  // Split is transparent: if its inputs have been converted from fp16 to
  // fp32, it is rewritten to output fp32.
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {2, 4}});
    input = Abs(input);
    XnnTensor axis({.type = Type::kI32,
                    .shape = {1},
                    .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});
    std::vector<XnnTensor> outputs = Split(input, axis, 2);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        graph, BuildXnnpackGraph({outputs[0], outputs[1]}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {2, 4}});
    input = Cast(input, Type::kFP32);
    input = Abs(input);
    XnnTensor axis({.type = Type::kI32,
                    .shape = {1},
                    .buffer = OwningCpuBuffer::Copy<Type::kI32>({0})});

    std::vector<XnnTensor> outputs = Split(input, axis, 2);

    XnnTensor out0 = Cast(outputs[0], Type::kFP16);
    XnnTensor out1 = Cast(outputs[1], Type::kFP16);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({out0, out1}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, FullyConnectedWithBias) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor weights({.type = Type::kFP16,
                       .shape = {2, 4},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                           {1, 2, 3, 4, 5, 6, 7, 8})});
    XnnTensor bias({.type = Type::kFP16,
                    .shape = {2},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP16>({1, 2})});
    XnnTensor output = FullyConnected(input, weights, bias);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    input = Cast(input, Type::kFP32);
    XnnTensor weights({.type = Type::kFP32,
                       .shape = {2, 4},
                       .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                           {1, 2, 3, 4, 5, 6, 7, 8})});
    XnnTensor bias(
        {.type = Type::kFP32,
         .shape = {2},
         .buffer = OwningCpuBuffer::Copy<Type::kFP32>({1.0f, 2.0f})});
    XnnTensor output = FullyConnected(input, weights, bias);
    output = Cast(output, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, BatchMatMul) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor x({.type = Type::kFP16, .shape = {1, 3, 4}});
    XnnTensor y({.type = Type::kFP16, .shape = {1, 4, 2}});
    XnnTensor output = BatchMatMul(x, y);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor x({.type = Type::kFP16, .shape = {1, 3, 4}});
    XnnTensor y({.type = Type::kFP16, .shape = {1, 4, 2}});
    XnnTensor x_fp32 = Cast(x, Type::kFP32);
    XnnTensor y_fp32 = Cast(y, Type::kFP32);
    XnnTensor output = BatchMatMul(x_fp32, y_fp32);
    output = Cast(output, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, TransposeConv2D) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor filter({.type = Type::kFP32,
                      .shape = {8, 3, 3, 3},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                          std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
    XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
    XnnTensor bias({.type = Type::kFP16,
                    .shape = {8},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                        std::vector<float>(8, 1.0f))});
    XnnTensor output = TransposeConv2D(filter, input, bias, {1, 5, 5, 8},
                                       litert::tensor::kPaddingSame,
                                       /*stride_h=*/1, /*stride_w=*/1);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor filter({.type = Type::kFP32,
                      .shape = {8, 3, 3, 3},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                          std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
    XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
    XnnTensor input_fp32 = Cast(input, Type::kFP32);
    XnnTensor bias({.type = Type::kFP32,
                    .shape = {8},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                        std::vector<float>(8, 1.0f))});
    XnnTensor output = TransposeConv2D(filter, input_fp32, bias, {1, 5, 5, 8},
                                       litert::tensor::kPaddingSame,
                                       /*stride_h=*/1, /*stride_w=*/1);
    output = Cast(output, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FallbackTest, Conv2D) {
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
    XnnTensor filter({.type = Type::kFP32,
                      .shape = {8, 3, 3, 3},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                          std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
    XnnTensor bias({.type = Type::kFP32,
                    .shape = {8},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                        std::vector<float>(8, 1.0f))});
    XnnTensor output = Conv2D(input, filter, bias, /*stride_h=*/1,
                              /*stride_w=*/1, litert::tensor::kPaddingSame);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
    XnnTensor input_fp32 = Cast(input, Type::kFP32);
    XnnTensor filter({.type = Type::kFP32,
                      .shape = {8, 3, 3, 3},
                      .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                          std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
    XnnTensor bias({.type = Type::kFP32,
                    .shape = {8},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                        std::vector<float>(8, 1.0f))});
    XnnTensor output = Conv2D(input_fp32, filter, bias, /*stride_h=*/1,
                              /*stride_w=*/1, litert::tensor::kPaddingSame);
    output = Cast(output, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

class Fp16ToFp32FineGrainedOpSupportTest : public testing::Test {
 public:
  void SetUp() override {
#if XNN_ARCH_ARM64
    mock_config_.arch_flags |= xnn_arch_arm_neon_fp16_arith;
#elif XNN_ARCH_X86_64
    mock_config_.arch_flags |= xnn_arch_x86_sse2;
    mock_config_.arch_flags |= xnn_arch_x86_avx;
    mock_config_.arch_flags |= xnn_arch_x86_f16c;
    mock_config_.arch_flags |= xnn_arch_x86_fma3;
    mock_config_.arch_flags |= xnn_arch_x86_avx2;
#else
    GTEST_SKIP();
#endif
  }

  void TearDown() override { xnn_reset_hardware_config(); }

  xnn_hardware_config mock_config_{};
};

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, AbsAndElu) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor output = Abs(input);
        output = Elu(output);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor output = Abs(input);
        output = Elu(output);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest,
       TransparentAbsHandlesRewrittenInputs) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor output = Elu(input);
        output = Abs(output);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor output = Elu(input);
        output = Abs(output);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, NegLogisticAdd) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor b({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor a_neg = Neg(a);
        XnnTensor a_sig = Logistic(a_neg);
        XnnTensor b_neg = Neg(b);
        XnnTensor output = Add(a_sig, b_neg);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, BuildXnnpackGraph([] {
        XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor b({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor a_neg = Neg(a);
        XnnTensor a_sig = Logistic(a_neg);
        XnnTensor b_neg = Neg(b);
        XnnTensor output = Add(a_sig, b_neg);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, SoftmaxTest) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph, [] {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor output = Softmax(input);
    return BuildXnnpackGraph({output});
  }());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, [] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor output = Softmax(input);
        return BuildXnnpackGraph({output});
      }());

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, FullyConnected) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor weights({.type = Type::kFP16,
                           .shape = {2, 4},
                           .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                               {1, 2, 3, 4, 5, 6, 7, 8})});
        XnnTensor output = FullyConnected(input, weights);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        XnnTensor weights({.type = Type::kFP16,
                           .shape = {2, 4},
                           .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                               {1, 2, 3, 4, 5, 6, 7, 8})});
        XnnTensor output = FullyConnected(input, weights);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, BatchMatMul) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor x({.type = Type::kFP16, .shape = {1, 3, 4}});
        XnnTensor y({.type = Type::kFP16, .shape = {1, 4, 2}});
        XnnTensor output = BatchMatMul(x, y);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, BuildXnnpackGraph([] {
        XnnTensor x({.type = Type::kFP16, .shape = {1, 3, 4}});
        XnnTensor y({.type = Type::kFP16, .shape = {1, 4, 2}});
        XnnTensor output = BatchMatMul(x, y);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, TransposeConv2D) {
  xnn_set_hardware_config(&mock_config_);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor filter({.type = Type::kFP32,
                          .shape = {8, 3, 3, 3},
                          .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                              std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
        XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
        XnnTensor bias({.type = Type::kFP16,
                        .shape = {8},
                        .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                            std::vector<float>(8, 1.0f))});
        XnnTensor output = TransposeConv2D(filter, input, bias, {1, 5, 5, 8},
                                           litert::tensor::kPaddingSame,
                                           /*stride_h=*/1, /*stride_w=*/1);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, BuildXnnpackGraph([] {
        XnnTensor filter({.type = Type::kFP32,
                          .shape = {8, 3, 3, 3},
                          .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                              std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
        XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
        XnnTensor bias({.type = Type::kFP16,
                        .shape = {8},
                        .buffer = OwningCpuBuffer::Copy<Type::kFP16>(
                            std::vector<float>(8, 1.0f))});
        XnnTensor output = TransposeConv2D(filter, input, bias, {1, 5, 5, 8},
                                           litert::tensor::kPaddingSame,
                                           /*stride_h=*/1, /*stride_w=*/1);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, Conv2D) {
  xnn_set_hardware_config(&mock_config_);

  const bool vmulcaddc_supported = xnn_init_f16_vmulcaddc_config() != nullptr;

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> graph, BuildXnnpackGraph([] {
        XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
        XnnTensor filter({.type = Type::kFP32,
                          .shape = {8, 3, 3, 3},
                          .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                              std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
        XnnTensor bias({.type = Type::kFP32,
                        .shape = {8},
                        .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                            std::vector<float>(8, 1.0f))});
        XnnTensor output = Conv2D(input, filter, bias, /*stride_h=*/1,
                                  /*stride_w=*/1, litert::tensor::kPaddingSame);
        return std::vector<litert::tensor::TensorHandle>({output});
      }()));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph,
      BuildXnnpackGraph([vmulcaddc_supported] {
        XnnTensor input({.type = Type::kFP16, .shape = {1, 5, 5, 3}});
        XnnTensor filter({.type = Type::kFP32,
                          .shape = {8, 3, 3, 3},
                          .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                              std::vector<float>(8 * 3 * 3 * 3, 1.0f))});
        XnnTensor bias({.type = Type::kFP32,
                        .shape = {8},
                        .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                            std::vector<float>(8, 1.0f))});
        if (vmulcaddc_supported) {
          XnnTensor output =
              Conv2D(input, filter, bias, /*stride_h=*/1,
                     /*stride_w=*/1, litert::tensor::kPaddingSame);
          return std::vector<litert::tensor::TensorHandle>({output});
        } else {
          XnnTensor input_fp32 = Cast(input, Type::kFP32);
          XnnTensor output = Conv2D(input_fp32, filter, bias,
                                    /*stride_h=*/1, /*stride_w=*/1,
                                    litert::tensor::kPaddingSame);
          output = Cast(output, Type::kFP16);
          return std::vector<litert::tensor::TensorHandle>({output});
        }
      }()));

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

struct UnaryOpParam {
  std::string name;
  std::function<XnnTensor(XnnTensor)> op_builder;
};

class Fp16ToFp32FallbackUnaryOpTest
    : public Fp16ToFp32FallbackTest,
      public testing::WithParamInterface<UnaryOpParam> {};

TEST_P(Fp16ToFp32FallbackUnaryOpTest, Rewrite) {
  const auto& param = GetParam();
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph, [&] {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor output = param.op_builder(input);
    return BuildXnnpackGraph({output});
  }());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XnnpackGraph> expected_graph, [&] {
        XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
        // These ops have software emulation fallbacks and will therefore not be
        // converted.
        if (param.name == "Cos" || param.name == "Sin") {
          XnnTensor output = param.op_builder(input);
          return BuildXnnpackGraph({output});
        } else {
          XnnTensor input_fp32 = Cast(input, Type::kFP32);
          XnnTensor output_fp32 = param.op_builder(input_fp32);
          XnnTensor output = Cast(output_fp32, Type::kFP16);
          return BuildXnnpackGraph({output});
        }
      }());

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

INSTANTIATE_TEST_SUITE_P(
    UnaryOps, Fp16ToFp32FallbackUnaryOpTest,
    testing::ValuesIn<UnaryOpParam>({
        {"Abs", [](XnnTensor x) { return Abs(x); }},
        {"Relu", [](XnnTensor x) { return Relu(x); }},
        {"Elu", [](XnnTensor x) { return Elu(x); }},
        {"ApproxGelu",
         [](XnnTensor x) { return Gelu(x, /*approximate=*/true); }},
        {"Cos", [](XnnTensor x) { return Cos(x); }},
        {"Exp", [](XnnTensor x) { return Exp(x); }},
        {"Gelu", [](XnnTensor x) { return Gelu(x, /*approximate=*/false); }},
        {"HardSwish", [](XnnTensor x) { return HardSwish(x); }},
        {"LeakyRelu", [](XnnTensor x) { return LeakyRelu(x); }},
        {"Log", [](XnnTensor x) { return Log(x); }},
        {"Neg", [](XnnTensor x) { return Neg(x); }},
        {"Logistic", [](XnnTensor x) { return Logistic(x); }},
        {"Sin", [](XnnTensor x) { return Sin(x); }},
        {"Square", [](XnnTensor x) { return Square(x); }},
        {"Sqrt", [](XnnTensor x) { return Sqrt(x); }},
        {"Tanh", [](XnnTensor x) { return Tanh(x); }},
        {"Rsqrt", [](XnnTensor x) { return Rsqrt(x); }},
        {"Ceil", [](XnnTensor x) { return Ceil(x); }},
        {"Floor", [](XnnTensor x) { return Floor(x); }},
        {"Round", [](XnnTensor x) { return Round(x); }},
    }),
    [](const testing::TestParamInfo<Fp16ToFp32FallbackUnaryOpTest::ParamType>&
           info) { return info.param.name; });

struct BinaryOpParam {
  std::string name;
  std::function<XnnTensor(XnnTensor, XnnTensor)> op_builder;
};

class Fp16ToFp32FallbackBinaryOpTest
    : public Fp16ToFp32FallbackTest,
      public testing::WithParamInterface<BinaryOpParam> {};

TEST_P(Fp16ToFp32FallbackBinaryOpTest, Rewrite) {
  const auto& param = GetParam();
  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor b({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor output = param.op_builder(a, b);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor a({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor a_fp32 = Cast(a, Type::kFP32);
    XnnTensor b({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor b_fp32 = Cast(b, Type::kFP32);
    XnnTensor output_fp32 = param.op_builder(a_fp32, b_fp32);
    XnnTensor output = Cast(output_fp32, Type::kFP16);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(graph->subgraph(),
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, NoRewriteAfterFp16Rewrite) {
  xnn_set_hardware_config(&mock_config_);

  std::unique_ptr<XnnpackGraph> graph;
  {
    XnnTensor input({.type = Type::kFP32, .shape = {3, 4}});
    XnnTensor output = Abs(input);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
  }

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP32, .shape = {3, 4}});
    XnnTensor output = Abs(input);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  // We expect the graph to be in the FP16 state after rewrite.
  // So we run rewrite on expected_graph too.
  ASSERT_TRUE(xnn_subgraph_rewrite_for_fp16(expected_graph->subgraph()));

  xnn_subgraph_t subgraph = graph->subgraph();

  // Rewrite to FP16.
  ASSERT_TRUE(xnn_subgraph_rewrite_for_fp16(subgraph));

  // Run fallback. It should be a no-op because native FP16 is supported.
  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(subgraph,
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  // The graph should match the rewritten FP16 graph.
  EXPECT_THAT(graph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

struct Qd8FcSubgraph {
  xnn_subgraph_t subgraph = nullptr;
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t convert_output_id = XNN_INVALID_VALUE_ID;
  uint32_t weights_id = XNN_INVALID_VALUE_ID;
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  std::vector<float> weights_scale = {1.0f, 1.0f};
  std::vector<int8_t> weights_data = {1, 2, 3, 4, 5, 6, 7, 8};

  void Build(enum xnn_datatype input_datatype,
             enum xnn_datatype output_datatype) {
    ASSERT_EQ(
        xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph),
        xnn_status_success);

    std::vector<size_t> input_dims = {3, 4};
    std::vector<size_t> weights_dims = {2, 4};
    std::vector<size_t> output_dims = {3, 2};

    // Define Input (external)
    ASSERT_EQ(xnn_define_tensor_value(subgraph, input_datatype,
                                      input_dims.size(), input_dims.data(),
                                      /*data=*/nullptr, /*external_id=*/0,
                                      XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
              xnn_status_success);

    // Define Temp (qdint8, internal)
    ASSERT_EQ(
        xnn_define_dynamically_quantized_tensor_value(
            subgraph, xnn_datatype_qdint8, input_dims.size(),
            /*num_nonbatch_dims=*/1, input_dims.data(), XNN_INVALID_VALUE_ID,
            /*flags=*/0, &convert_output_id),
        xnn_status_success);

    // Define Convert Node (type 6)
    ASSERT_EQ(xnn_define_unary(subgraph, xnn_unary_convert,
                               /*params=*/nullptr, input_id, convert_output_id,
                               /*flags=*/0),
              xnn_status_success);

    // Define Weights (static, qcint8)
    ASSERT_EQ(xnn_define_channelwise_quantized_tensor_value(
                  subgraph, xnn_datatype_qcint8, weights_scale.data(),
                  weights_dims.size(), /*channel_dim=*/0, weights_dims.data(),
                  weights_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0,
                  &weights_id),
              xnn_status_success);

    // Define Output (external)
    ASSERT_EQ(
        xnn_define_tensor_value(subgraph, output_datatype, output_dims.size(),
                                output_dims.data(),
                                /*data=*/nullptr, /*external_id=*/1,
                                XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
        xnn_status_success);

    // Define Fully Connected Node
    ASSERT_EQ(xnn_define_fully_connected(
                  subgraph, -INFINITY, INFINITY, convert_output_id, weights_id,
                  XNN_INVALID_VALUE_ID, output_id, /*flags=*/0),
              xnn_status_success);
  }
};

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, Fp16ToQdint8Convert) {
  xnn_set_hardware_config(&mock_config_);

  Qd8FcSubgraph builder;
  builder.Build(xnn_datatype_fp16, xnn_datatype_fp16);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  Qd8FcSubgraph expected_builder;
  expected_builder.Build(xnn_datatype_fp16, xnn_datatype_fp16);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
      expected_subgraph_guard(expected_builder.subgraph, xnn_delete_subgraph);

  // Run fallback. Since native FP16 is supported (via mock_config_),
  // and all ops in the graph (Convert FP16->qd8, FC qd8->f16) are supported,
  // it should be a no-op (0 changes).
  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(builder.subgraph,
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  // The graph should match the expected FP16 graph.
  EXPECT_THAT(builder.subgraph, IsIsomorphicTo(expected_builder.subgraph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FallbackTest, Fp16ToQdint8ConvertFallback) {
  Qd8FcSubgraph builder;
  builder.Build(xnn_datatype_fp16, xnn_datatype_fp16);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      builder.subgraph, xnn_delete_subgraph);

  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor temp_fp32_in = Cast(input, Type::kFP32);

    XnnTensor weights(
        {.type = Type::kI8,
         .shape = {2, 4},
         .buffer = OwningCpuBuffer::Copy<Type::kI8>({1, 2, 3, 4, 5, 6, 7, 8})});
    weights.SetQuantization(
        std::make_shared<litert::tensor::PerChannelAffineQuantization>(
            /*scales=*/std::vector<float>{1.0f, 1.0f},
            /*zero_points=*/std::vector<int64_t>{0, 0},
            /*quantized_dimension=*/0));

    XnnTensor temp_fp32_out = FullyConnected(temp_fp32_in, weights);
    XnnTensor output = Cast(temp_fp32_out, Type::kFP16);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  // Run fallback. Since FP16 is disabled, it should rewrite it.
  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(builder.subgraph,
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  // The graph should match the expected rewritten graph.
  EXPECT_THAT(builder.subgraph, IsIsomorphicTo(expected_graph));
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, ElideType6Convert) {
  xnn_set_hardware_config(&mock_config_);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(
      xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph),
      xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      subgraph, xnn_delete_subgraph);

  std::vector<size_t> dims = {3, 4};

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp16, dims.size(),
                                    dims.data(),
                                    /*data=*/nullptr, /*external_id=*/0,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
            xnn_status_success);

  uint32_t temp_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp16, dims.size(),
                                    dims.data(),
                                    /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                    /*flags=*/0, &temp_id),
            xnn_status_success);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp16, dims.size(),
                                    dims.data(),
                                    /*data=*/nullptr, /*external_id=*/1,
                                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            xnn_status_success);

  // Node 0: Convert (FP16->FP16, type 6)
  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  ASSERT_NE(node, nullptr);
  xnn_init_convert_node(node, input_id, temp_id, /*flags=*/0);

  // Node 1: Abs (FP16)
  ASSERT_EQ(xnn_define_unary(subgraph, xnn_unary_abs,
                             /*params=*/nullptr, temp_id, output_id,
                             /*flags=*/0),
            xnn_status_success);

  // Expected graph: Input -> Abs -> Output (no convert)
  std::unique_ptr<XnnpackGraph> expected_graph;
  {
    XnnTensor input({.type = Type::kFP16, .shape = {3, 4}});
    XnnTensor output = Abs(input);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(expected_graph,
                                    BuildXnnpackGraph({output}));
  }

  // Run fallback. Convert should be elided.
  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(subgraph,
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  // The graph should match the expected graph (only Abs).
  EXPECT_THAT(subgraph, IsIsomorphicTo(expected_graph));

  xnn_reset_hardware_config();
}

TEST_F(Fp16ToFp32FineGrainedOpSupportTest, DontElideType6ConvertWhenExternal) {
  xnn_set_hardware_config(&mock_config_);

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(
      xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph),
      xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph_guard(
      subgraph, xnn_delete_subgraph);

  std::vector<size_t> dims = {3, 4};

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp16, dims.size(),
                                    dims.data(),
                                    /*data=*/nullptr, /*external_id=*/0,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
            xnn_status_success);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(subgraph, xnn_datatype_fp16, dims.size(),
                                    dims.data(),
                                    /*data=*/nullptr, /*external_id=*/1,
                                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            xnn_status_success);

  // Node 0: Convert (FP16->FP16, type 6)
  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  ASSERT_NE(node, nullptr);
  xnn_init_convert_node(node, input_id, output_id, /*flags=*/0);

  // Expected graph: same as starting graph (must keep the convert)
  xnn_subgraph_t expected_subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0,
                                &expected_subgraph),
            xnn_status_success);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
      expected_subgraph_guard(expected_subgraph, xnn_delete_subgraph);

  uint32_t exp_input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(
                expected_subgraph, xnn_datatype_fp16, dims.size(), dims.data(),
                /*data=*/nullptr, /*external_id=*/0,
                XNN_VALUE_FLAG_EXTERNAL_INPUT, &exp_input_id),
            xnn_status_success);

  uint32_t exp_output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(
                expected_subgraph, xnn_datatype_fp16, dims.size(), dims.data(),
                /*data=*/nullptr, /*external_id=*/1,
                XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &exp_output_id),
            xnn_status_success);

  struct xnn_node* exp_node = xnn_subgraph_new_node(expected_subgraph);
  ASSERT_NE(exp_node, nullptr);
  xnn_init_convert_node(exp_node, exp_input_id, exp_output_id, /*flags=*/0);

  // Run fallback. Convert should NOT be elided.
  ASSERT_THAT(xnn_subgraph_fallback_from_fp16_to_fp32(subgraph,
                                                      /*optimization_flags=*/0),
              Eq(xnn_status_success));

  // The graph should still match the expected graph (with convert).
  EXPECT_THAT(subgraph, IsIsomorphicTo(expected_subgraph));

  xnn_reset_hardware_config();
}

INSTANTIATE_TEST_SUITE_P(
    BinaryOps, Fp16ToFp32FallbackBinaryOpTest,
    testing::ValuesIn<BinaryOpParam>({
        {"Add", [](XnnTensor a, XnnTensor b) { return Add(a, b); }},
        {"Sub", [](XnnTensor a, XnnTensor b) { return Sub(a, b); }},
        {"Mul", [](XnnTensor a, XnnTensor b) { return Mul(a, b); }},
        {"Div", [](XnnTensor a, XnnTensor b) { return Div(a, b); }},
        {"Maximum", [](XnnTensor a, XnnTensor b) { return Maximum(a, b); }},
        {"Minimum", [](XnnTensor a, XnnTensor b) { return Minimum(a, b); }},
        {"PRelu", [](XnnTensor a, XnnTensor b) { return PRelu(a, b); }},
    }),
    [](const testing::TestParamInfo<Fp16ToFp32FallbackBinaryOpTest::ParamType>&
           info) { return info.param.name; });

}  // namespace
