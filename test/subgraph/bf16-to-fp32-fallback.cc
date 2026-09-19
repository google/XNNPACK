// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/subgraph/rewrites/cvt_to_fp32.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/subgraph.h"
#include "test/subgraph/rewrites/subgraph_matcher.h"
#include "litert/tensor/arithmetic.h"
#include "litert/tensor/backends/xnnpack/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/backends/xnnpack/graph.h"
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
  PrintTo(graph.GetSubgraph(), os);
}
}  // namespace litert::tensor

namespace {

using XnnTensor = litert::tensor::Tensor<litert::tensor::XnnpackMixinTag>;

using litert::tensor::BuildXnnpackGraph;
using litert::tensor::Type;
using litert::tensor::XnnpackGraph;
using testing::Eq;
using xnnpack::IsIsomorphicTo;

struct InlineQd8Bf16Qb4wFullyConnected {
  std::unique_ptr<XnnpackGraph> graph;
  xnn_subgraph_t subgraph = nullptr;
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  // BF16 1.0 scales backing the patched BF16 scale pointer below. The Tensor
  // API only produces FP16 blockwise scales.
  std::vector<uint16_t> bf16_scales = {0x3F80, 0x3F80};

  void Build(enum xnn_datatype input_datatype,
             enum xnn_datatype output_datatype = xnn_datatype_bf16,
             enum xnn_datatype scale_datatype = xnn_datatype_bf16) {
    // The Tensor backend only emits a valid qd8_qb4w graph for FP32 inputs
    // (it inserts a dynamic-quantize convert to qdint8). A BF16 input would
    // reach xnn_define_fully_connected as BF16, which is rejected with
    // xnn_status_invalid_parameter. Always build with FP32, then patch the
    // input datatype to the requested type below.
    XnnTensor input({.type = Type::kFP32, .shape = {3, 32}});
    // 2x32 int4 weights (64 nibbles, 32 bytes). {-8, -8} packs to 0x88,
    // which decodes to all-zero weights with zero_point=8.
    std::vector<litert::tensor::int4_t> weights_data(
        32, litert::tensor::int4_t{-8, -8});
    auto quantization = std::make_shared<litert::tensor::BlockwiseQuantization>(
        std::vector<float>{1.0f, 1.0f}, std::vector<int64_t>{8, 8},
        /*block_size=*/32, /*quantized_dimension=*/0);
    XnnTensor weights({.type = Type::kI4,
                       .shape = {2, 32},
                       .buffer = std::move(weights_data),
                       .quantization = std::move(quantization)});
    XnnTensor output = FullyConnected(input, weights);
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, BuildXnnpackGraph({output}));
    subgraph = graph->GetSubgraph();

    struct xnn_node* fully_connected = nullptr;
    for (size_t i = 0; i < subgraph->num_nodes; ++i) {
      if (subgraph->nodes[i].type == xnn_node_type_fully_connected) {
        fully_connected = &subgraph->nodes[i];
        break;
      }
    }
    ASSERT_NE(fully_connected, nullptr);
    output_id = fully_connected->outputs[0];

    // The Tensor API always emits FP16 blockwise scales; patch to BF16 when
    // requested. Only the scale_type enum matters for the fallback decision
    // under test.
    xnn_value& filter = subgraph->values[fully_connected->inputs[1]];
    ASSERT_EQ(filter.datatype, xnn_datatype_qbint4);
    if (scale_datatype == xnn_datatype_bf16) {
      filter.quantization.scale_type = xnn_datatype_bf16;
      filter.quantization.blockwise_scale.bf16_scale =
          reinterpret_cast<const xnn_bfloat16*>(bf16_scales.data());
    } else {
      filter.quantization.scale_type = scale_datatype;
    }

    // FullyConnected infers the output type from the FP32 input; override when
    // the test needs a different output datatype.
    if (output_datatype != xnn_datatype_fp32) {
      xnn_value& out = subgraph->values[output_id];
      out.datatype = output_datatype;
      out.size = xnn_tensor_get_size(&out);
    }

    // Match the graph state produced by packed-LHS fusion while retaining the
    // original input datatype used by the inline packer. The Tensor API
    // backend inserted a convert to qdint8; rewire the FC to consume the
    // original input like fusion does.
    bool rewired = false;
    for (size_t i = 0; i < subgraph->num_nodes; ++i) {
      struct xnn_node& node = subgraph->nodes[i];
      if (&node != fully_connected && node.num_outputs > 0 &&
          node.outputs[0] == fully_connected->inputs[0]) {
        fully_connected->inputs[0] = node.inputs[0];
        rewired = true;
        break;
      }
    }
    ASSERT_TRUE(rewired);
    if (input_datatype != xnn_datatype_fp32) {
      xnn_value& in = subgraph->values[fully_connected->inputs[0]];
      in.datatype = input_datatype;
      in.size = xnn_tensor_get_size(&in);
    }
    fully_connected->flags |= XNN_FLAG_INLINE_LHS_PACKING;
    fully_connected->packed_input_datatype = xnn_datatype_qdint8;
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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

  ASSERT_THAT(xnn_subgraph_fallback_from_bf16_to_fp32(graph->GetSubgraph(),
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
