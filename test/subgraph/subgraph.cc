// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/subgraph.h"
#include "src/xnnpack/params.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "test/subgraph/runtime-tester.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

TEST(SUBGRAPH, hanging_nodes) {
  SubgraphTester tester(6);
  tester.AddDynamicTensorF32({1, 256, 256, 3}, 0)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, 1)
      .AddStaticTensorF32({32}, TensorType::kDense, 2)
      .AddDynamicTensorF32({1, 128, 128, 32}, 3)
      .AddOutputTensorF32({32}, 4)
      .AddDynamicTensorF32({32}, 5)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{1, 1, 1, 1},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          0, 1, 2, 3)
      .AddGlobalAveragePooling(3, 4)
      // Add hanging node
      .AddGlobalAveragePooling(3, 5)
      .Optimize();

  // The hanging node is no longer there.
  ASSERT_EQ(tester.NumNodes(), 2);
}

TEST(SUBGRAPH, multiple_outputs_with_hanging_nodes) {
  SubgraphTester tester(4);
  tester.AddDynamicTensorF32({96}, 0)
      .AddDynamicTensorF32({32}, 1)
      .AddDynamicTensorF32({32}, 2)
      .AddOutputTensorF32({32}, 3)
      // Add split3 with 1 consumed output and two unconsumed outputs.
      .AddEvenSplit(0, 0, {1, 2, 3})
      .Optimize();

  // The node is still there.
  ASSERT_EQ(tester.NumNodes(), 1);
  // And all four values also.
  ASSERT_EQ(tester.NumValues(), 4);
  // The first two outputs are optimized away.
  ASSERT_EQ(tester.Value(1)->type, xnn_value_type_invalid);
  ASSERT_EQ(tester.Value(2)->type, xnn_value_type_invalid);
  // The last output is consumed.
  ASSERT_EQ(tester.Value(3)->type, xnn_value_type_dense_tensor);
}

TEST(SUBGRAPH, even_split3_first_two_outputs_optimized_away) {
  RuntimeTester tester(5);
  constexpr size_t size = 9;
  float inputs[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  tester.AddStaticTensorF32({size}, TensorType::kDense, 0, 0, inputs)
      .AddDynamicTensorF32({3}, 1)
      .AddDynamicTensorF32({3}, 2)
      .AddOutputTensorF32({3}, 3)
      // Add split3 with 1 consumed output and two unconsumed outputs.
      .AddEvenSplit(0, 0, {1, 2, 3});
  // Regression test for a crash where we could not deal with a split where the
  // 0th output is not used (and optimized away).
  auto output = tester.RunWithFusion<float>();
  xnnpack::Buffer<float> expected = {6, 7, 8};
  ASSERT_EQ(expected, output);
}

TEST(SUBGRAPH, create_subgraph_invalid_external_value_ids) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  EXPECT_EQ(xnn_create_subgraph(XNN_INVALID_VALUE_ID, /*flags=*/0, &subgraph),
            xnn_status_invalid_parameter);
  EXPECT_EQ(subgraph, nullptr);
}

TEST(SUBGRAPH, reserve_values_overflow) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0,
                                &subgraph),
            xnn_status_success);
  EXPECT_EQ(xnn_subgraph_reserve_values(subgraph, SIZE_MAX),
            xnn_status_out_of_memory);
  EXPECT_EQ(xnn_subgraph_reserve_values(
                subgraph, static_cast<size_t>(XNN_INVALID_VALUE_ID)),
            xnn_status_out_of_memory);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

TEST(SUBGRAPH, reserve_nodes_overflow) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0,
                                &subgraph),
            xnn_status_success);
  EXPECT_EQ(xnn_subgraph_reserve_nodes(subgraph, SIZE_MAX),
            xnn_status_out_of_memory);
  EXPECT_EQ(xnn_subgraph_reserve_nodes(
                subgraph, static_cast<size_t>(XNN_INVALID_NODE_ID)),
            xnn_status_out_of_memory);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

namespace {

class FailingValueTableReallocGuard {
 public:
  FailingValueTableReallocGuard() : saved_allocator_(xnn_params.allocator) {
    xnn_params.allocator.context = this;
    xnn_params.allocator.reallocate = FailingReallocate;
  }
  ~FailingValueTableReallocGuard() { xnn_params.allocator = saved_allocator_; }

  FailingValueTableReallocGuard(const FailingValueTableReallocGuard&) = delete;
  FailingValueTableReallocGuard& operator=(
      const FailingValueTableReallocGuard&) = delete;

 private:
  static void* FailingReallocate(void* context, void* pointer, size_t size) {
    auto* self = static_cast<FailingValueTableReallocGuard*>(context);
    if (self->fail_active_ && size % sizeof(struct xnn_value) == 0 &&
        size != sizeof(struct xnn_value)) {
      self->fail_active_ = false;
      return nullptr;
    }
    return self->saved_allocator_.reallocate(
        self->saved_allocator_.context, pointer, size);
  }

  const struct xnn_allocator saved_allocator_;
  bool fail_active_ = true;
};

struct DuplicateOutputGraph {
  xnn_subgraph_t subgraph = nullptr;
  uint32_t input_id = 0;
  uint32_t a_id = 0;
  uint32_t b_id = 0;
  uint32_t output_id = 0;
  uint32_t persistent_id = 0;
};

DuplicateOutputGraph CreateDuplicateOutputGraph() {
  DuplicateOutputGraph graph;
  if (xnn_create_subgraph(5, 0, &graph.subgraph) != xnn_status_success) {
    graph.subgraph = nullptr;
    return graph;
  }
  const size_t dims[1] = {8};
  const size_t scalar[1] = {1};
  if (xnn_define_tensor_value(
          graph.subgraph, xnn_datatype_fp32, 1, dims, nullptr, 0,
          XNN_VALUE_FLAG_EXTERNAL_INPUT, &graph.input_id) !=
          xnn_status_success ||
      xnn_define_tensor_value(
          graph.subgraph, xnn_datatype_fp32, 1, scalar, nullptr, 1,
          XNN_VALUE_FLAG_EXTERNAL_INPUT, &graph.a_id) !=
          xnn_status_success ||
      xnn_define_tensor_value(
          graph.subgraph, xnn_datatype_fp32, 1, scalar, nullptr, 2,
          XNN_VALUE_FLAG_EXTERNAL_INPUT, &graph.b_id) !=
          xnn_status_success ||
      xnn_define_tensor_value(
          graph.subgraph, xnn_datatype_fp32, 1, dims, nullptr, 3,
          XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &graph.output_id) !=
          xnn_status_success ||
      xnn_define_tensor_value(
          graph.subgraph, xnn_datatype_fp32, 1, dims, nullptr, 4,
          XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
          &graph.persistent_id) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  if (xnn_define_binary(graph.subgraph, xnn_binary_multiply, nullptr,
                        graph.persistent_id, graph.a_id, graph.output_id,
                        0) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  if (xnn_define_binary(graph.subgraph, xnn_binary_multiply, nullptr,
                        graph.persistent_id, graph.b_id, graph.output_id,
                        0) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  if (xnn_define_binary(graph.subgraph, xnn_binary_add, nullptr,
                        graph.output_id, graph.input_id,
                        graph.persistent_id, 0) != xnn_status_success) {
    xnn_delete_subgraph(graph.subgraph);
    graph.subgraph = nullptr;
    return graph;
  }
  return graph;
}

}  // namespace

TEST(SUBGRAPH, ssa_rewrite_allocation_failure_does_not_crash) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  DuplicateOutputGraph graph = CreateDuplicateOutputGraph();
  ASSERT_NE(graph.subgraph, nullptr);

  float input_data[8] = {0.0f};
  float a_data[1] = {2.0f};
  float b_data[1] = {3.0f};
  float output_data[8] = {0.0f};
  float persistent_data[8] = {1.0f};
  struct xnn_external_value values[5] = {
      {graph.input_id, input_data},
      {graph.a_id, a_data},
      {graph.b_id, b_data},
      {graph.output_id, output_data},
      {graph.persistent_id, persistent_data},
  };

  xnn_runtime_t runtime = nullptr;
  xnn_status status;
  {
    FailingValueTableReallocGuard guard;
    status = xnn_create_runtime_v4(
        graph.subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  }

  if (status == xnn_status_success) {
    ASSERT_NE(runtime, nullptr);
    EXPECT_EQ(xnn_reshape_runtime(runtime), xnn_status_success);
    EXPECT_EQ(xnn_setup_runtime(runtime, 5, values), xnn_status_success);
    EXPECT_EQ(xnn_invoke_runtime(runtime), xnn_status_success);
    xnn_delete_runtime(runtime);
  }
  xnn_delete_subgraph(graph.subgraph);
}

}  // namespace xnnpack
