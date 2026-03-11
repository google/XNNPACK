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
#include "src/subgraph/rewrites/fp16_to_fp32.h"
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

TEST(Fp16ToFp32FallbackTest, SingleOpRewrite) {
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

TEST(Fp16ToFp32FallbackTest, OpChainRewrite) {
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

TEST(Fp16ToFp32FallbackTest, ReshapeAllowsFp16Inputs) {
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

TEST(Fp16ToFp32FallbackTest, ReshapeHandlesRewrittenInputs) {
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

TEST(Fp16ToFp32FallbackTest, DontInsertConvertFp32Fp32) {
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

TEST(Fp16ToFp32FallbackTest, Fp16ToFp16HandleExternalInput) {
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

TEST(Fp16ToFp32FallbackTest, Fp16ToFp16HandleExternalOutput) {
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

TEST(Fp16ToFp32FallbackTest,
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

TEST(Fp16ToFp32FallbackTest, HandleExternalOutputThatIsReused) {
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

TEST(Fp16ToFp32FallbackTest, TransposeAllowsFp16Inputs) {
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

TEST(Fp16ToFp32FallbackTest, TransposeHandlesRewrittenInputs) {
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

TEST(Fp16ToFp32FallbackTest, ReuseConvertedFp32ValueForMultipleConsumers) {
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

TEST(Fp16ToFp32FallbackTest, SplitAllowsFp16Inputs) {
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
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(
        graph, BuildXnnpackGraph({out0, out1}));
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

TEST(Fp16ToFp32FallbackTest, SplitHandlesRewrittenInputs) {
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

TEST(Fp16ToFp32FallbackTest, FullyConnectedWithBias) {
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

}  // namespace
