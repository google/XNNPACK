// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/math.h>
#include <xnnpack/memory-planner.h>
#include <xnnpack/subgraph.h>

#include "runtime-tester.h"
#include <gtest/gtest.h>

namespace xnnpack {

TEST(MemoryPlanner, ValueLiveInfo) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  // Create simple subgraph where it has 2 nodes and 4 tensors as illustrated below:
  // T0 ----> N0 ----> T2  and T2 ----> N1 ----> T3
  // T1 ----/                  T1 ----/
  struct xnn_subgraph subgraph;
  subgraph.num_values = 4;
  subgraph.num_nodes = 2;
  struct xnn_node nodes[2];
  nodes[0].num_inputs = 2;
  nodes[0].inputs[0] = 0;
  nodes[0].inputs[1] = 1;
  nodes[0].num_outputs = 1;
  nodes[0].outputs[0] = 2;

  nodes[1].num_inputs = 2;
  nodes[1].inputs[0] = 1;
  nodes[1].inputs[1] = 2;
  nodes[1].num_outputs = 1;
  nodes[1].outputs[0] = 3;
  subgraph.nodes = nodes;

  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &subgraph);

  EXPECT_EQ(0, tracker.usage[0].first_node);
  EXPECT_EQ(0, tracker.usage[0].last_node);

  EXPECT_EQ(0, tracker.usage[1].first_node);
  EXPECT_EQ(1, tracker.usage[1].last_node);

  EXPECT_EQ(0, tracker.usage[2].first_node);
  EXPECT_EQ(1, tracker.usage[2].last_node);

  EXPECT_EQ(1, tracker.usage[3].first_node);
  EXPECT_EQ(1, tracker.usage[3].last_node);

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, MemoryBlocksCoalescing) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_subgraph subgraph;
  subgraph.num_nodes = 0;
  subgraph.num_values = 5;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &subgraph);
  // As this is an empty subgraph, we create the following xnn_value_usage stub.
  tracker.usage[0].first_node = 1,
  tracker.usage[0].last_node = 1,
  xnn_add_value_allocation_tracker(&tracker, 0, 56);

  tracker.usage[1].first_node = 0,
  tracker.usage[1].last_node = 1,
  xnn_add_value_allocation_tracker(&tracker, 1, 40);

  tracker.usage[2].first_node = 1,
  tracker.usage[2].last_node = 1,
  xnn_add_value_allocation_tracker(&tracker, 2, 64);

  tracker.usage[3].first_node = 0,
  tracker.usage[3].last_node = 0,
  xnn_add_value_allocation_tracker(&tracker, 3, 152);

  tracker.usage[4].first_node = 1,
  tracker.usage[4].last_node = 1,
  xnn_add_value_allocation_tracker(&tracker, 4, 20);

  for (size_t i = 0; i < subgraph.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

#if XNN_ENABLE_MEMOPT
  EXPECT_EQ(192, tracker.mem_arena_size);
  EXPECT_EQ(64, tracker.usage[0].alloc_offset);
  EXPECT_EQ(152, tracker.usage[1].alloc_offset);
  EXPECT_EQ(0, tracker.usage[2].alloc_offset);
  EXPECT_EQ(0, tracker.usage[3].alloc_offset);
  EXPECT_EQ(120, tracker.usage[4].alloc_offset);
#else
  EXPECT_EQ(332, tracker.mem_arena_size);
  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(57, tracker.usage[1].alloc_offset);
  EXPECT_EQ(96, tracker.usage[2].alloc_offset);
  EXPECT_EQ(160, tracker.usage[3].alloc_offset);
  EXPECT_EQ(312, tracker.usage[4].alloc_offset);
#endif

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, GeneralPlanning) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_subgraph subgraph;
  subgraph.num_nodes = 0;
  subgraph.num_values = 8;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &subgraph);
  // As this is an empty subgraph, we create the following xnn_value_usage stub.
  tracker.usage[0].first_node = 0,
  tracker.usage[0].last_node = 1,
  xnn_add_value_allocation_tracker(&tracker, 0, 32);

  tracker.usage[1].first_node = 1,
  tracker.usage[1].last_node = 4,
  xnn_add_value_allocation_tracker(&tracker, 1, 28);

  tracker.usage[2].first_node = 2,
  tracker.usage[2].last_node = 5,
  xnn_add_value_allocation_tracker(&tracker, 2, 36);

  tracker.usage[3].first_node = 3,
  tracker.usage[3].last_node = 5,
  xnn_add_value_allocation_tracker(&tracker, 3, 16);

  tracker.usage[4].first_node = 4,
  tracker.usage[4].last_node = 5,
  xnn_add_value_allocation_tracker(&tracker, 4, 8);

  tracker.usage[5].first_node = 5,
  tracker.usage[5].last_node = 7,
  xnn_add_value_allocation_tracker(&tracker, 5, 64);

  tracker.usage[6].first_node = 6,
  tracker.usage[6].last_node = 8,
  xnn_add_value_allocation_tracker(&tracker, 6, 10);

  tracker.usage[7].first_node = 7,
  tracker.usage[7].last_node = 8,
  xnn_add_value_allocation_tracker(&tracker, 7, 40);

  for (size_t i = 0; i < subgraph.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

#if XNN_ENABLE_MEMOPT
  EXPECT_EQ(124, tracker.mem_arena_size);
  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(32, tracker.usage[1].alloc_offset);
  EXPECT_EQ(64, tracker.usage[2].alloc_offset);
  EXPECT_EQ(100, tracker.usage[3].alloc_offset);
  EXPECT_EQ(116, tracker.usage[4].alloc_offset);
  EXPECT_EQ(0, tracker.usage[5].alloc_offset);
  EXPECT_EQ(104, tracker.usage[6].alloc_offset);
  EXPECT_EQ(64, tracker.usage[7].alloc_offset);
#else
  EXPECT_EQ(234, tracker.mem_arena_size);
  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(32, tracker.usage[1].alloc_offset);
  EXPECT_EQ(60, tracker.usage[2].alloc_offset);
  EXPECT_EQ(96, tracker.usage[3].alloc_offset);
  EXPECT_EQ(112, tracker.usage[4].alloc_offset);
  EXPECT_EQ(120, tracker.usage[5].alloc_offset);
  EXPECT_EQ(184, tracker.usage[6].alloc_offset);
  EXPECT_EQ(194, tracker.usage[7].alloc_offset);
#endif

  xnn_release_value_allocation_tracker(&tracker);
}

// Extra space for memory arena due to sparse microkernels reading extra. Should be in sync with runtime.c
namespace {
constexpr size_t MEMORY_ARENA_EXTRA_BYTES = 2 * XNN_EXTRA_BYTES;
}

TEST(MemoryPlanner, LeakyReluInPlaceAfterConv) {
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = XNN_INVALID_VALUE_ID;  // No bias tensor.
  uint32_t conv_out = 2;
  uint32_t leaky_relu_out = 3;
  uint32_t output_id = 4;

  // Conv -> Leaky Relu -> Clamp
  RuntimeTester tester(5);
  tester
    .AddInputTensorF32({1, 5, 5, 3}, input_id)
    .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddDynamicTensorF32({1, 3, 3, 3}, conv_out, /*flags=*/0)
    .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out, /*flags=*/0)
    .AddOutputTensorF32({1, 3, 3, 3}, output_id)
    .AddConvolution2D(
        ConvolutionParams{
            Padding{0, 0, 0, 0},
            Kernel{3, 3},
            Subsampling{1, 1},
            Dilation{1, 1},
            /*groups=*/1,
            /*group_input_channels=*/3,
            /*group_output_channels=*/1,
        },
        input_id, filter_id, bias_id, conv_out)
    .AddLeakyRelu(1.0f, conv_out, leaky_relu_out)
    .AddClamp(0.0f, 1.0f, leaky_relu_out, output_id);
  tester.CreateRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for conv_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            round_up_po2(1 * 3 * 3 * 3 * sizeof(float), XNN_EXTRA_BYTES) * 1 + MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->blobs[conv_out].data, runtime->blobs[leaky_relu_out].data);
}

TEST(MemoryPlanner, LeakyReluWithTwoConsumersCannotBeInPlace) {
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = XNN_INVALID_VALUE_ID;  // No bias tensor.
  uint32_t conv_out = 2;
  uint32_t leaky_relu_out = 3;
  uint32_t output_id = 4;
  uint32_t output_id2 = 5;

  // Conv -> Leaky Relu -> Clamp
  //                   \-> Clamp
  RuntimeTester tester(6);
  tester
    .AddInputTensorF32({1, 5, 5, 3}, input_id)
    .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddDynamicTensorF32({1, 3, 3, 3}, conv_out, /*flags=*/0)  // 108 bytes.
    .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out, /*flags=*/0) // 108 bytes.
    .AddOutputTensorF32({1, 3, 3, 3}, output_id)
    .AddOutputTensorF32({1, 3, 3, 3}, output_id2)
    .AddConvolution2D(
        ConvolutionParams{
            Padding{0, 0, 0, 0},
            Kernel{3, 3},
            Subsampling{1, 1},
            Dilation{1, 1},
            /*groups=*/1,
            /*group_input_channels=*/3,
            /*group_output_channels=*/1,
        },
        input_id, filter_id, bias_id, conv_out)
    .AddLeakyRelu(1.0f, conv_out, leaky_relu_out)
    .AddClamp(0.0f, 1.0f, leaky_relu_out, output_id)
    .AddClamp(1.0f, 2.0f, leaky_relu_out, output_id2);
  tester.CreateRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Since leaky relu has 2 consumers, we cannot yet do it in place since we cannot easily find all consumers of the
  // value without traversing the graph. This limitation can be lifted in the future.
  ASSERT_EQ(runtime->workspace->size,
            round_up_po2(1 * 3 * 3 * 3 * sizeof(float), XNN_EXTRA_BYTES) * 2 + MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_NE(runtime->blobs[conv_out].data, runtime->blobs[leaky_relu_out].data);
}

TEST(MemoryPlanner, HardSwishAndLeakyReluInPlaceAfterConv) {
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = XNN_INVALID_VALUE_ID;  // No bias tensor.
  uint32_t conv_out = 2;
  uint32_t leaky_relu_out = 3;
  uint32_t hard_swish_out = 4;
  uint32_t output_id = 5;

  // Conv -> Leaky Relu -> Hard Swish -> Clamp
  RuntimeTester tester(6);
  tester
    .AddInputTensorF32({1, 5, 5, 3}, input_id)
    .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddDynamicTensorF32({1, 3, 3, 3}, conv_out, /*flags=*/0)  // 108 bytes.
    .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out, /*flags=*/0) // 108 bytes.
    .AddDynamicTensorF32({1, 3, 3, 3}, hard_swish_out, /*flags=*/0) // 108 bytes.
    .AddOutputTensorF32({1, 3, 3, 3}, output_id)
    .AddConvolution2D(
        ConvolutionParams{
            Padding{0, 0, 0, 0},
            Kernel{3, 3},
            Subsampling{1, 1},
            Dilation{1, 1},
            /*groups=*/1,
            /*group_input_channels=*/3,
            /*group_output_channels=*/1,
        },
        input_id, filter_id, bias_id, conv_out)
    .AddLeakyRelu(1.0f, conv_out, leaky_relu_out)
    .AddHardSwish(leaky_relu_out, hard_swish_out)
    .AddClamp(0.0f, 1.0f, hard_swish_out, output_id);
  tester.CreateRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for conv_out tensor, leaky relu and hard swish can be in place.
  ASSERT_EQ(runtime->workspace->size,
            round_up_po2(1 * 3 * 3 * 3 * sizeof(float), XNN_EXTRA_BYTES) * 1 + MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->blobs[conv_out].data, runtime->blobs[leaky_relu_out].data);
  ASSERT_EQ(runtime->blobs[leaky_relu_out].data, runtime->blobs[hard_swish_out].data);
}

TEST(MemoryPlanner, ExternalInputsCannotBeInPlace) {
  uint32_t input_id = 0;
  uint32_t leaky_relu_out = 1;
  uint32_t output_id = 3;

  // Leaky Relu -> Clamp
  RuntimeTester tester(4);
  tester
      .AddInputTensorF32({1, 3, 3, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out, /*flags=*/0) // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddLeakyRelu(1.0f, input_id, leaky_relu_out)
      .AddClamp(0.0f, 1.0f, leaky_relu_out, output_id);
  tester.CreateRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Need space allocated for leaky relu output tensor because we cannot modify the external input tensor.
  ASSERT_EQ(runtime->workspace->size,
            round_up_po2(1 * 3 * 3 * 3 * sizeof(float), XNN_EXTRA_BYTES) * 1 + MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, PersistentValuesCannotReuseInternalValues) {
  uint32_t input_id = 0;
  uint32_t clamp_out_id = 1;
  uint32_t leaky_relu_out_id = 2;
  uint32_t output_id = 3;

  // Clamp -> Leaky Relu -> Clamp
  RuntimeTester tester(4);
  tester
      .AddInputTensorF32({1, 3, 3, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, clamp_out_id, /*flags=*/0) // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out_id, XNN_VALUE_FLAG_PERSISTENT) // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddClamp(0.0f, 1.0f, input_id, clamp_out_id)
      .AddLeakyRelu(1.0f, clamp_out_id, leaky_relu_out_id)
      .AddClamp(0.0f, 1.0f, leaky_relu_out_id, output_id);
  tester.CreateRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Persistent values need to be allocated their own space.
  ASSERT_EQ(runtime->workspace->size,
            round_up_po2(1 * 3 * 3 * 3 * sizeof(float), XNN_EXTRA_BYTES) * 2 + MEMORY_ARENA_EXTRA_BYTES);

}

TEST(MemoryPlanner, CannotReuseStaticValues) {
  uint32_t static_id = 0;
  uint32_t clamp_out_id = 1;
  uint32_t output_id = 2;

  // --- static_id --> Clamp --- clamp_out_id --> Leaky Relu --- output_id -->
  RuntimeTester tester(3);
  tester
      .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense, static_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, clamp_out_id) // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddClamp(0.0f, 1.0f, static_id, clamp_out_id)
      .AddLeakyRelu(1.0f, clamp_out_id, output_id);
  tester.CreateRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // clamp_out_id cannot reuse static_id (because it is static).
  ASSERT_EQ(runtime->workspace->size,
            round_up_po2(1 * 3 * 3 * 3 * sizeof(float), XNN_EXTRA_BYTES) * 1 + MEMORY_ARENA_EXTRA_BYTES);
}

} // namespace xnnpack
