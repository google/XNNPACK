// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/memory-planner.h"

#include <cstddef>
#include <cstdint>

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "test/subgraph/mock-allocator.h"
#include "include/xnnpack.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"
#include "test/subgraph/runtime-flags.h"
#include "test/subgraph/runtime-tester.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

TEST(MemoryPlanner, ValueLiveInfo) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  // Create simple runtime where it has 2 nodes and 4 tensors as illustrated
  // below: T0 ----> N0 ----> T2  and T2 ----> N1 ----> T3 T1 ----/ T1 ----/
  struct xnn_runtime runtime;
  runtime.num_values = 4;
  runtime.num_ops = 2;
  struct xnn_operator_data nodes[2];
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
  runtime.opdata = nodes;

  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);

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


TEST(MemoryPlanner, FailedWorkspaceReplanRetriesBeforeInvoke) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));

  MockAllocator mock_allocator;
  std::unique_ptr<MockAllocator, decltype(&RestoreDefaultAllocator)>
      auto_mock_allocator(&mock_allocator, &RestoreDefaultAllocator);
  SetUpMockAllocator(&mock_allocator);

  size_t alloc_calls = 0;
  EXPECT_CALL(mock_allocator, allocate(testing::_, testing::_))
      .WillRepeatedly([&](void* context, size_t size) -> void* {
        ++alloc_calls;
        return xnn_default_allocator.allocate(context, size);
      });
  EXPECT_CALL(mock_allocator, reallocate(testing::_, testing::_, testing::_))
      .Times(testing::AnyNumber());
  EXPECT_CALL(mock_allocator, deallocate(testing::_, testing::_))
      .Times(testing::AnyNumber());
  EXPECT_CALL(mock_allocator, aligned_deallocate(testing::_, testing::_))
      .Times(testing::AnyNumber());

  size_t aligned_alloc_calls = 0;
  bool fail_aligned_alloc = false;
  EXPECT_CALL(mock_allocator,
              aligned_allocate(testing::_, testing::_, testing::_))
      .WillRepeatedly([&](void* context, size_t alignment, size_t size) -> void* {
        ++aligned_alloc_calls;
        if (fail_aligned_alloc) {
          return nullptr;
        }
        return xnn_default_allocator.aligned_allocate(context, alignment, size);
      });

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  const size_t initial_dim = 1;
  uint32_t in_id = XNN_INVALID_VALUE_ID;
  uint32_t mid_id = XNN_INVALID_VALUE_ID;
  uint32_t out_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1,
                                    &initial_dim, nullptr, 0,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, &in_id));
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1,
                                    &initial_dim, nullptr, XNN_INVALID_VALUE_ID,
                                    0, &mid_id));
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1,
                                    &initial_dim, nullptr, 1,
                                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id));

  const int64_t begin = 0;
  const int64_t end = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_define_static_slice_v3(subgraph, 1, &begin, &end, nullptr,
                                       in_id, mid_id, 0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_static_slice_v3(subgraph, 1, &begin, &end, nullptr,
                                       mid_id, out_id, 0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0,
                                  &runtime));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  std::vector<float> input(8192, 3.5f);
  std::vector<float> output(8192, 0.0f);
  const xnn_external_value external[2] = {
      {in_id, input.data()},
      {out_id, output.data()},
  };

  // 1. Positive baseline control: small arena plans and invokes cleanly.
  size_t dim = 1;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, in_id, 1, &dim));
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime_v2(runtime, 2, external));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  EXPECT_FLOAT_EQ(output[0], 3.5f);

  // 2. Grow internal workspace tensor to 8192 floats while aligned_allocate
  // fails; reshape must report out_of_memory without marking the new size planned.
  dim = 8192;
  fail_aligned_alloc = true;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, in_id, 1, &dim));
  EXPECT_EQ(xnn_status_out_of_memory, xnn_reshape_runtime(runtime));
  fail_aligned_alloc = false;

  // 3. Intermediate reshape (1 < 1024 < 8192) must retry memory planning rather
  // than invoking against the stale 1-element workspace.
  dim = 1024;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, in_id, 1, &dim));
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime_v2(runtime, 2, external));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  EXPECT_FLOAT_EQ(output[1023], 3.5f);

  // 4. Steady-state same-shape reshape must not re-run memory planning once planned.
  const size_t alloc_calls_after_recovery = alloc_calls;
  const size_t calls_after_recovery = aligned_alloc_calls;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, in_id, 1, &dim));
  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
  EXPECT_EQ(alloc_calls, alloc_calls_after_recovery);
  EXPECT_EQ(aligned_alloc_calls, calls_after_recovery);
}

TEST(MemoryPlanner, MemoryBlocksCoalescing) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 5;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);
  // As this is an empty runtime, we create the following xnn_value_usage stub.
  tracker.usage[0].first_node = 1;
  tracker.usage[0].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 0, 56);

  tracker.usage[1].first_node = 0;
  tracker.usage[1].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 1, 40);

  tracker.usage[2].first_node = 1;
  tracker.usage[2].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 2, 64);

  tracker.usage[3].first_node = 0;
  tracker.usage[3].last_node = 0;
  xnn_add_value_allocation_tracker(&tracker, 3, 152);

  tracker.usage[4].first_node = 1;
  tracker.usage[4].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 4, 20);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

  EXPECT_EQ(192, tracker.mem_arena_size);
  EXPECT_EQ(64, tracker.usage[0].alloc_offset);
  EXPECT_EQ(152, tracker.usage[1].alloc_offset);
  EXPECT_EQ(0, tracker.usage[2].alloc_offset);
  EXPECT_EQ(0, tracker.usage[3].alloc_offset);
  EXPECT_EQ(120, tracker.usage[4].alloc_offset);

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, ReusesLeadingGap) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 3;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);

  tracker.usage[0].first_node = 0;
  tracker.usage[0].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 0, 100);

  tracker.usage[1].first_node = 1;
  tracker.usage[1].last_node = 2;
  xnn_add_value_allocation_tracker(&tracker, 1, 80);

  tracker.usage[2].first_node = 2;
  tracker.usage[2].last_node = 3;
  xnn_add_value_allocation_tracker(&tracker, 2, 60);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(100, tracker.usage[1].alloc_offset);
  EXPECT_EQ(0, tracker.usage[2].alloc_offset);
  EXPECT_EQ(180, tracker.mem_arena_size);

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, PrefersSmallerInternalGapOverLeadingGap) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 5;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);

  tracker.usage[0].first_node = 0;
  tracker.usage[0].last_node = 0;
  xnn_add_value_allocation_tracker(&tracker, 0, 100);

  tracker.usage[1].first_node = 0;
  tracker.usage[1].last_node = 2;
  xnn_add_value_allocation_tracker(&tracker, 1, 90);

  tracker.usage[2].first_node = 0;
  tracker.usage[2].last_node = 0;
  xnn_add_value_allocation_tracker(&tracker, 2, 80);

  tracker.usage[3].first_node = 0;
  tracker.usage[3].last_node = 2;
  xnn_add_value_allocation_tracker(&tracker, 3, 70);

  tracker.usage[4].first_node = 2;
  tracker.usage[4].last_node = 3;
  xnn_add_value_allocation_tracker(&tracker, 4, 60);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(100, tracker.usage[1].alloc_offset);
  EXPECT_EQ(190, tracker.usage[2].alloc_offset);
  EXPECT_EQ(270, tracker.usage[3].alloc_offset);
  EXPECT_EQ(190, tracker.usage[4].alloc_offset);
  EXPECT_EQ(340, tracker.mem_arena_size);

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, PrefersLeadingGapOnEqualFit) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 6;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);

  tracker.usage[0].first_node = 0;
  tracker.usage[0].last_node = 0;
  xnn_add_value_allocation_tracker(&tracker, 0, 110);

  tracker.usage[1].first_node = 0;
  tracker.usage[1].last_node = 2;
  xnn_add_value_allocation_tracker(&tracker, 1, 100);

  tracker.usage[2].first_node = 0;
  tracker.usage[2].last_node = 0;
  xnn_add_value_allocation_tracker(&tracker, 2, 60);

  tracker.usage[3].first_node = 0;
  tracker.usage[3].last_node = 0;
  xnn_add_value_allocation_tracker(&tracker, 3, 50);

  tracker.usage[4].first_node = 0;
  tracker.usage[4].last_node = 2;
  xnn_add_value_allocation_tracker(&tracker, 4, 45);

  tracker.usage[5].first_node = 2;
  tracker.usage[5].last_node = 3;
  xnn_add_value_allocation_tracker(&tracker, 5, 40);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(110, tracker.usage[1].alloc_offset);
  EXPECT_EQ(210, tracker.usage[2].alloc_offset);
  EXPECT_EQ(270, tracker.usage[3].alloc_offset);
  EXPECT_EQ(320, tracker.usage[4].alloc_offset);
  EXPECT_EQ(0, tracker.usage[5].alloc_offset);
  EXPECT_EQ(365, tracker.mem_arena_size);

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, GeneralPlanning) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 8;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);
  // As this is an empty runtime, we create the following xnn_value_usage stub.
  tracker.usage[0].first_node = 0;
  tracker.usage[0].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 0, 32);

  tracker.usage[1].first_node = 1;
  tracker.usage[1].last_node = 4;
  xnn_add_value_allocation_tracker(&tracker, 1, 28);

  tracker.usage[2].first_node = 2;
  tracker.usage[2].last_node = 5;
  xnn_add_value_allocation_tracker(&tracker, 2, 36);

  tracker.usage[3].first_node = 3;
  tracker.usage[3].last_node = 5;
  xnn_add_value_allocation_tracker(&tracker, 3, 16);

  tracker.usage[4].first_node = 4;
  tracker.usage[4].last_node = 5;
  xnn_add_value_allocation_tracker(&tracker, 4, 8);

  tracker.usage[5].first_node = 5;
  tracker.usage[5].last_node = 7;
  xnn_add_value_allocation_tracker(&tracker, 5, 64);

  tracker.usage[6].first_node = 6;
  tracker.usage[6].last_node = 8;
  xnn_add_value_allocation_tracker(&tracker, 6, 10);

  tracker.usage[7].first_node = 7;
  tracker.usage[7].last_node = 8;
  xnn_add_value_allocation_tracker(&tracker, 7, 40);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  xnn_plan_value_allocation_tracker(&tracker);

  EXPECT_EQ(124, tracker.mem_arena_size);
  EXPECT_EQ(0, tracker.usage[0].alloc_offset);
  EXPECT_EQ(32, tracker.usage[1].alloc_offset);
  EXPECT_EQ(64, tracker.usage[2].alloc_offset);
  EXPECT_EQ(100, tracker.usage[3].alloc_offset);
  EXPECT_EQ(116, tracker.usage[4].alloc_offset);
  EXPECT_EQ(0, tracker.usage[5].alloc_offset);
  EXPECT_EQ(104, tracker.usage[6].alloc_offset);
  EXPECT_EQ(64, tracker.usage[7].alloc_offset);

  xnn_release_value_allocation_tracker(&tracker);
}

// Extra space for memory arena due to sparse microkernels reading extra. Should
// be in sync with runtime.c
namespace {
constexpr size_t MEMORY_ARENA_EXTRA_BYTES = 2 * XNN_EXTRA_BYTES;
}

TEST(MemoryPlanner, LeakyReluInPlaceAfterStencil) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t leaky_relu_out = 3;
  uint32_t output_id = 4;

  // Stencil -> Leaky Relu -> Clamp
  RuntimeTester tester(5);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out, /*flags=*/0)
      .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out, /*flags=*/0)
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddLeakyRelu(1.0f, stencil_out, leaky_relu_out)
      .AddClamp(0.0f, 1.0f, leaky_relu_out, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for stencil_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->values[stencil_out].data,
            runtime->values[leaky_relu_out].data);
}

TEST(MemoryPlanner, LeakyReluWithTwoConsumersCannotBeInPlace) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t leaky_relu_out = 3;
  uint32_t output_id = 4;
  uint32_t output_id2 = 5;

  // Stencil -> Leaky Relu -> Clamp
  //                   \-> Clamp
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out,
                           /*flags=*/0)  // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out,
                           /*flags=*/0)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddOutputTensorF32({1, 3, 3, 3}, output_id2)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddLeakyRelu(1.0f, stencil_out, leaky_relu_out)
      .AddClamp(0.0f, 1.0f, leaky_relu_out, output_id)
      .AddClamp(1.0f, 2.0f, leaky_relu_out, output_id2);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Since leaky relu has 2 consumers, we cannot yet do it in place since we
  // cannot easily find all consumers of the value without traversing the graph.
  // This limitation can be lifted in the future.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[leaky_relu_out]) +
                xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_NE(runtime->values[stencil_out].data,
            runtime->values[leaky_relu_out].data);
}

TEST(MemoryPlanner, HardSwishAndLeakyReluInPlaceAfterStencil) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t leaky_relu_out = 3;
  uint32_t hard_swish_out = 4;
  uint32_t output_id = 5;

  // Stencil -> Leaky Relu -> Hard Swish -> Clamp
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out,
                           /*flags=*/0)  // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out,
                           /*flags=*/0)  // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, hard_swish_out,
                           /*flags=*/0)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddLeakyRelu(1.0f, stencil_out, leaky_relu_out)
      .AddHardSwish(leaky_relu_out, hard_swish_out)
      .AddClamp(0.0f, 1.0f, hard_swish_out, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for stencil_out tensor, leaky relu and hard swish
  // can be in place.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->values[stencil_out].data,
            runtime->values[leaky_relu_out].data);
  ASSERT_EQ(runtime->values[leaky_relu_out].data,
            runtime->values[hard_swish_out].data);
}

TEST(MemoryPlanner, ExternalInputsCannotBeInPlace) {
  uint32_t input_id = 0;
  uint32_t leaky_relu_out = 1;
  uint32_t output_id = 3;

  // Leaky Relu -> Clamp
  RuntimeTester tester(4);
  tester.AddInputTensorF32({1, 3, 3, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, leaky_relu_out,
                           /*flags=*/0)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddLeakyRelu(1.0f, input_id, leaky_relu_out)
      .AddClamp(0.0f, 1.0f, leaky_relu_out, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // Need space allocated for leaky relu output tensor because we cannot modify
  // the external input tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[leaky_relu_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, CannotReuseStaticValues) {
  uint32_t static_id = 0;
  uint32_t clamp_out_id = 1;
  uint32_t output_id = 2;

  // --- static_id --> Clamp --- clamp_out_id --> Leaky Relu --- output_id -->
  RuntimeTester tester(3);
  tester.AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense, static_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, clamp_out_id)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddClamp(0.0f, 1.0f, static_id, clamp_out_id)
      .AddLeakyRelu(1.0f, clamp_out_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();

  xnn_runtime_t runtime = tester.Runtime();

  // clamp_out_id cannot reuse static_id (because it is static).
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[clamp_out_id]) +
                MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, Add2WithLHSConstantInPlace) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t add_constant_input_id = 3;
  uint32_t add_out_id = 4;
  uint32_t output_id = 5;

  // Stencil -> Add -> LeakyRelu
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out)  // 108 bytes.
      .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense,
                          add_constant_input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, add_out_id)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddAddition(add_constant_input_id, stencil_out, add_out_id)
      .AddLeakyRelu(1.0f, add_out_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for stencil_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->values[stencil_out].data,
            runtime->values[add_out_id].data);
}

TEST(MemoryPlanner, Add2WithLHSConstantRHSBroadcast) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t add_constant_input_id = 3;
  uint32_t add_out_id = 4;
  uint32_t output_id = 5;

  // Stencil -> Add -> LeakyRelu
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 1}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 1}, stencil_out)  // 108 bytes.
      .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense,
                          add_constant_input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, add_out_id)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddAddition(add_constant_input_id, stencil_out, add_out_id)
      .AddLeakyRelu(1.0f, add_out_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Need space for conv_out tensor and add out as add cannot be done in-place.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                xnn_tensor_get_rounded_size(&runtime->values[add_out_id]) +
                MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, Add2WithRHSConstantInPlace) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t add_constant_input_id = 3;
  uint32_t add_out_id = 4;
  uint32_t output_id = 5;

  // Stencil -> Add -> LeakyRelu
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out)  // 108 bytes.
      .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense,
                          add_constant_input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, add_out_id)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddAddition(stencil_out, add_constant_input_id, add_out_id)
      .AddLeakyRelu(1.0f, add_out_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for stencil_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->values[stencil_out].data,
            runtime->values[add_out_id].data);
}

TEST(MemoryPlanner, Mul2WithLHSConstant) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t mul_constant_input_id = 3;
  uint32_t mul_out_id = 4;
  uint32_t output_id = 5;

  // Stencil -> Mul -> LeakyRelu
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out)  // 108 bytes.
      .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense,
                          mul_constant_input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, mul_out_id)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddMultiply(mul_constant_input_id, stencil_out, mul_out_id)
      .AddLeakyRelu(1.0f, mul_out_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for stencil_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->values[stencil_out].data,
            runtime->values[mul_out_id].data);
}

TEST(MemoryPlanner, Mul2WithRHSConstant) {
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t mul_constant_input_id = 3;
  uint32_t mul_out_id = 4;
  uint32_t output_id = 5;

  // Stencil -> Mul -> LeakyRelu
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out)  // 108 bytes.
      .AddStaticTensorF32({1, 3, 3, 3}, TensorType::kDense,
                          mul_constant_input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, mul_out_id)  // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddMultiply(stencil_out, mul_constant_input_id, mul_out_id)
      .AddLeakyRelu(1.0f, mul_out_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Should only need space for stencil_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  ASSERT_EQ(runtime->values[stencil_out].data,
            runtime->values[mul_out_id].data);
}

// check a case where input is reused, different size.
TEST(MemoryPlanner, Add2WithImplicitBroadcast) {
  // input1     input2
  //   |          |
  // HardSwish   Stencil
  //       \      /
  //         Add
  //          |
  //        LeakyRelu
  //          |
  //        output
  uint32_t input1_id = 0;
  uint32_t input2_id = 1;
  uint32_t hard_swish_out = 3;
  uint32_t stencil_out = 4;
  uint32_t add_out = 5;
  uint32_t output_id = 6;

  RuntimeTester tester(7);
  tester
      .AddInputTensorF32({1, 1, 1, 3}, input1_id)  // intentionally smaller
      .AddInputTensorF32({1, 5, 5, 3}, input2_id)
      .AddDynamicTensorF32({1, 1, 1, 3}, hard_swish_out)  // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out)     // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, add_out)         // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddHardSwish(input1_id, hard_swish_out)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input2_id,
          /*output_id=*/stencil_out)
      .AddAddition(hard_swish_out, stencil_out, add_out)
      .AddLeakyRelu(1.0f, add_out, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Need space for hard_swish_out stencil_out tensor.
  ASSERT_EQ(runtime->workspace->size,
            xnn_tensor_get_rounded_size(&runtime->values[hard_swish_out]) +
                xnn_tensor_get_rounded_size(&runtime->values[stencil_out]) +
                MEMORY_ARENA_EXTRA_BYTES);
  // add_out should reuse stencil_out, hard_swish_out is too small.
  ASSERT_EQ(runtime->values[stencil_out].data, runtime->values[add_out].data);
}

TEST(MemoryPlanner, Add2WithInputMultipleConsumers) {
  //   input1
  //     |
  //   Stencil (Average pooling)
  //   /    \
  //   |   MaxPooling
  //   \    /
  //     Add
  //      |
  //    LeakyRelu
  //      |
  //    output
  //
  uint32_t input_id = 0;
  uint32_t stencil_out = 2;
  uint32_t max_pooling_2d_out = 3;
  uint32_t add_out = 4;
  uint32_t output_id = 5;

  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddDynamicTensorF32({1, 3, 3, 3}, stencil_out)         // 108 bytes.
      .AddDynamicTensorF32({1, 1, 1, 3}, max_pooling_2d_out)  // 108 bytes.
      .AddDynamicTensorF32({1, 3, 3, 3}, add_out)             // 108 bytes.
      .AddOutputTensorF32({1, 3, 3, 3}, output_id)
      .AddAveragePooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*input_id=*/input_id,
          /*output_id=*/stencil_out)
      .AddMaxPooling2D(
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          /*pooling_height=*/3,
          /*pooling_width=*/3,
          /*stride_height=*/1,
          /*stride_width=*/1,
          /*dilation_height=*/1,
          /*dilation_width=*/1,
          /*input_id=*/stencil_out,
          /*output_id=*/max_pooling_2d_out)
      .AddAddition(stencil_out, max_pooling_2d_out, add_out)
      .AddLeakyRelu(1.0f, add_out, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();

  // Need space for stencil_out, add cannot reuse stencil_out,
  // max_pooling_2d_out is also too small, so it needs allocation.
  ASSERT_EQ(
      runtime->workspace->size,
      xnn_tensor_get_rounded_size(
          &runtime->values[stencil_out])  // for stencil_out
          + xnn_tensor_get_rounded_size(
                &runtime->values[max_pooling_2d_out])  // for max_pooling_2d_out
          +
          xnn_tensor_get_rounded_size(&runtime->values[add_out])  // for add_out
          + MEMORY_ARENA_EXTRA_BYTES);
  // add_out should reuse stencil_out, hard_swish_out is too small.
  ASSERT_NE(runtime->values[stencil_out].data,
            runtime->values[max_pooling_2d_out].data);
  ASSERT_NE(runtime->values[max_pooling_2d_out].data,
            runtime->values[add_out].data);
}

TEST(MemoryPlanner, FullyConnectedDynamicFilterDynamicBias) {
  uint32_t input1_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t output_id = 3;
  uint32_t input2_id = 4;
  uint32_t input3_id = 5;

  // input1 input2 input3
  //  |      |      |
  //  |    [pad]  [pad]
  //  |      |      |
  //  \  filter_id bias_id
  //   \     |      |
  //   [fully connected]
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input1_id)
      .AddInputTensorF32({2, 3, 3, 2}, input2_id)
      .AddInputTensorF32({1}, input3_id)
      .AddDynamicTensorF32({2, 3, 3, 3}, filter_id)
      .AddDynamicTensorF32({2}, bias_id)
      .AddOutputTensorF32({2, 3, 3, 2}, output_id)
      .AddConstantPad({0, 0, 0, 1}, {0, 0, 0, 0}, 0.0f, input2_id, filter_id)
      .AddConstantPad({1}, {0}, 0.0f, input3_id, bias_id)
      .AddFullyConnected(input1_id, filter_id, bias_id, output_id);

  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();
  xnn_operator_data* fc_opdata = nullptr;
  size_t packed_fc_input_bytes = 0;
  switch (runtime->opdata[2].type) {
    case xnn_node_type_fully_connected:
      fc_opdata = &runtime->opdata[2];
      break;
    case xnn_node_type_pack_lh:
      fc_opdata = &runtime->opdata[3];
      ASSERT_EQ(fc_opdata->type, xnn_node_type_fully_connected);
      packed_fc_input_bytes =
          xnn_tensor_get_rounded_size(&runtime->values[fc_opdata->inputs[0]]);
      break;
    default:
      GTEST_FAIL() << "Expected either a fully-connected or pack-lh node.";
  }

  ASSERT_EQ(
      runtime->workspace->size,
      xnn_tensor_get_rounded_size(&runtime->values[filter_id])  // for filter_id
          + xnn_get_rounded_size(
                fc_opdata->workspace_size)  // for weights packing
          +
          xnn_tensor_get_rounded_size(&runtime->values[bias_id])  // for bias_id
          + packed_fc_input_bytes  // for packed lhs.
          + MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, FullyConnectedDynamicFilterStaticBias) {
  uint32_t input1_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t output_id = 3;
  uint32_t input2_id = 4;

  // input1 input2
  //  |      |
  //  |    [pad]
  //  |      |
  //  \  filter_id bias_id
  //   \     |      |
  //   [fully connected]
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input1_id)
      .AddInputTensorF32({2, 3, 3, 2}, input2_id)
      .AddDynamicTensorF32({2, 3, 3, 3}, filter_id)
      .AddStaticTensorF32({2}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({2, 3, 3, 2}, output_id)
      .AddConstantPad({0, 0, 0, 1}, {0, 0, 0, 0}, 0.0f, input2_id, filter_id)
      .AddFullyConnected(input1_id, filter_id, bias_id, output_id);

  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();
  xnn_operator_data* fc_opdata = nullptr;
  size_t packed_fc_input_bytes = 0;
  switch (runtime->opdata[1].type) {
    case xnn_node_type_fully_connected:
      fc_opdata = &runtime->opdata[1];
      break;
    case xnn_node_type_pack_lh:
      fc_opdata = &runtime->opdata[2];
      ASSERT_EQ(fc_opdata->type, xnn_node_type_fully_connected);
      packed_fc_input_bytes =
          xnn_tensor_get_rounded_size(&runtime->values[fc_opdata->inputs[0]]);
      break;
    default:
      GTEST_FAIL() << "Expected either a fully-connected or pack-lh node.";
  }

  ASSERT_EQ(
      runtime->workspace->size,
      xnn_tensor_get_rounded_size(&runtime->values[filter_id])  // for filter_id
          + xnn_get_rounded_size(
                fc_opdata->workspace_size)  // for weights packing
          + packed_fc_input_bytes           // for packed lhs.
          + MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, FullyConnectedDynamicFilterNoBias) {
  uint32_t input1_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  uint32_t output_id = 3;
  uint32_t input2_id = 4;

  // input1 input2
  //  |      |
  //  |    [pad]
  //  |      |
  //  \  filter_id
  //   \     |
  //   [fully connected]
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input1_id)
      .AddInputTensorF32({2, 3, 3, 2}, input2_id)
      .AddDynamicTensorF32({2, 3, 3, 3}, filter_id)
      .AddOutputTensorF32({2, 3, 3, 2}, output_id)
      .AddConstantPad({0, 0, 0, 1}, {0, 0, 0, 0}, 0.0f, input2_id, filter_id)
      .AddFullyConnected(input1_id, filter_id, bias_id, output_id);

  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();
  xnn_operator_data* fc_opdata = nullptr;
  size_t packed_fc_input_bytes = 0;
  switch (runtime->opdata[1].type) {
    case xnn_node_type_fully_connected:
      fc_opdata = &runtime->opdata[1];
      break;
    case xnn_node_type_pack_lh:
      fc_opdata = &runtime->opdata[2];
      ASSERT_EQ(fc_opdata->type, xnn_node_type_fully_connected);
      packed_fc_input_bytes =
          xnn_tensor_get_rounded_size(&runtime->values[fc_opdata->inputs[0]]);
      break;
    default:
      GTEST_FAIL() << "Expected either a fully-connected or pack-lh node.";
  }

  ASSERT_EQ(
      runtime->workspace->size,
      xnn_tensor_get_rounded_size(&runtime->values[filter_id])  // for filter_id
          + xnn_get_rounded_size(
                fc_opdata->workspace_size)  // for weights packing
          + packed_fc_input_bytes           // for packed lhs.
          + MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, FullyConnectedStaticFilterDynamicBias) {
  uint32_t input1_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t output_id = 3;
  uint32_t input3_id = 5;

  // input1        input3
  //  |             |
  //  |           [pad]
  //  |             |
  //  \  filter_id bias_id
  //   \     |      |
  //   [fully connected]
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input1_id)
      .AddInputTensorF32({1}, input3_id)
      .AddStaticTensorF32({2, 3, 3, 3}, TensorType::kDense, filter_id)
      .AddDynamicTensorF32({2}, bias_id)
      .AddOutputTensorF32({2, 3, 3, 2}, output_id)
      .AddConstantPad({1}, {0}, 0.0f, input3_id, bias_id)
      .AddFullyConnected(input1_id, filter_id, bias_id, output_id);

  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();
  xnn_operator_data* fc_opdata = nullptr;
  size_t packed_fc_input_bytes = 0;
  switch (runtime->opdata[1].type) {
    case xnn_node_type_fully_connected:
      fc_opdata = &runtime->opdata[1];
      break;
    case xnn_node_type_pack_lh:
      fc_opdata = &runtime->opdata[2];
      ASSERT_EQ(fc_opdata->type, xnn_node_type_fully_connected);
      packed_fc_input_bytes =
          xnn_tensor_get_rounded_size(&runtime->values[fc_opdata->inputs[0]]);
      break;
    default:
      GTEST_FAIL() << "Expected either a fully-connected or pack-lh node.";
  }

  ASSERT_EQ(
      runtime->workspace->size,
      xnn_get_rounded_size(fc_opdata->workspace_size)  // for weights packing
          +
          xnn_tensor_get_rounded_size(&runtime->values[bias_id])  // for bias_id
          + packed_fc_input_bytes  // for packed lhs.
          + MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, FullyConnectedExternalFilterExternalBias) {
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t output_id = 3;

  // input1 filter_id bias_id
  //  |      |         |
  //   \     |         /
  //   [fully connected]
  RuntimeTester tester(6);
  tester.AddInputTensorF32({1, 5, 5, 3}, input_id)
      .AddInputTensorF32({2, 3, 3, 3}, filter_id)
      .AddInputTensorF32({2}, bias_id)
      .AddOutputTensorF32({2, 3, 3, 2}, output_id)
      .AddFullyConnected(input_id, filter_id, bias_id, output_id);

  tester.CreateRuntime(xnn_test_runtime_flags());
  tester.SetupRuntime();
  xnn_runtime_t runtime = tester.Runtime();
  xnn_operator_data* fc_opdata = nullptr;
  size_t packed_fc_input_bytes = 0;
  switch (runtime->opdata[0].type) {
    case xnn_node_type_fully_connected:
      fc_opdata = &runtime->opdata[0];
      break;
    case xnn_node_type_pack_lh:
      fc_opdata = &runtime->opdata[1];
      ASSERT_EQ(fc_opdata->type, xnn_node_type_fully_connected);
      packed_fc_input_bytes =
          xnn_tensor_get_rounded_size(&runtime->values[fc_opdata->inputs[0]]);
      break;
    default:
      GTEST_FAIL() << "Expected either a fully-connected or pack-lh node.";
  }

  ASSERT_EQ(
      runtime->workspace->size,
      +xnn_get_rounded_size(fc_opdata->workspace_size)  // for weights packing
          + packed_fc_input_bytes                       // for packed lhs.
          + MEMORY_ARENA_EXTRA_BYTES);
}

TEST(MemoryPlanner, RoundedSizeOverflow) {
  EXPECT_EQ(SIZE_MAX, xnn_get_rounded_size(SIZE_MAX));
  EXPECT_EQ(SIZE_MAX, xnn_get_rounded_size(SIZE_MAX - 1));
}

TEST(MemoryPlanner, ValueAllocationTrackerOverflow) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 2;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);

  tracker.usage[0].first_node = 0;
  tracker.usage[0].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 0, SIZE_MAX);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  EXPECT_EQ(xnn_status_out_of_memory,
            xnn_plan_value_allocation_tracker(&tracker));

  xnn_release_value_allocation_tracker(&tracker);
}

TEST(MemoryPlanner, LiveMemoryBlocksOverflow) {
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime;
  runtime.num_ops = 0;
  runtime.num_values = 2;
  struct xnn_value_allocation_tracker tracker;
  xnn_init_value_allocation_tracker(&tracker, &runtime);

  tracker.usage[0].first_node = 0;
  tracker.usage[0].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 0, SIZE_MAX - 16);

  tracker.usage[1].first_node = 0;
  tracker.usage[1].last_node = 1;
  xnn_add_value_allocation_tracker(&tracker, 1, SIZE_MAX - 16);

  for (size_t i = 0; i < runtime.num_values; i++) {
    tracker.usage[i].reuse_value_id = XNN_INVALID_VALUE_ID;
  }
  EXPECT_EQ(xnn_status_out_of_memory,
            xnn_plan_value_allocation_tracker(&tracker));

  xnn_release_value_allocation_tracker(&tracker);
}

}  // namespace xnnpack
