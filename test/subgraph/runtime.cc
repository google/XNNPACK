#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "test/subgraph/runtime-tester.h"

TEST(RUNTIME, reshape_runtime) {
  xnnpack::RuntimeTester tester(4);
  uint32_t input0_id = 0;
  uint32_t input1_id = 1;
  uint32_t input2_id = 2;
  uint32_t output_id = 3;
  uint32_t add1_out, add2_out;
  size_t dim0 = 3;
  size_t new_dim0 = 400;
  size_t dummy_internal_dim = 1;

  // Set up input and output tensors.
  tester.AddInputTensorF32({dim0}, input0_id)
      .AddInputTensorF32({dim0}, input1_id)
      .AddInputTensorF32({dim0}, input2_id)
      .AddOutputTensorF32({dim0}, output_id)
      .AddInternalDynamicTensorF32({dummy_internal_dim}, &add1_out)
      .AddInternalDynamicTensorF32({dummy_internal_dim}, &add2_out);

  // Add ops. Note that we do this in two steps to avoid problems with the
  // `cmake-windows-x86` (using Visual C) build which doesn't propagate the
  // values for `add1_out` and `add2_out` properly.
  tester.AddAddition(input0_id, input1_id, add1_out)
      .AddAddition(input0_id, input2_id, add2_out)
      .AddMultiply(add1_out, add2_out, output_id);

  xnnpack::Buffer<float> expected(dim0);
  const float* input0_data = tester.GetExternalTensorDataF32(input0_id);
  const float* input1_data = tester.GetExternalTensorDataF32(input1_id);
  const float* input2_data = tester.GetExternalTensorDataF32(input2_id);
  for (size_t i = 0; i < dim0; ++i) {
    expected[i] =
        (input0_data[i] + input1_data[i]) * (input0_data[i] + input2_data[i]);
  }
  auto output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(expected, output);

  tester.ReshapeInput({new_dim0}, input0_id);
  tester.ReshapeInput({new_dim0}, input1_id);
  tester.ReshapeInput({new_dim0}, input2_id);

  tester.ReshapeRuntime();
  tester.SetupRuntimeV2();

  output = tester.RepeatRun<float>();
  expected = xnnpack::Buffer<float>(new_dim0);
  input0_data = tester.GetExternalTensorDataF32(input0_id);
  input1_data = tester.GetExternalTensorDataF32(input1_id);
  input2_data = tester.GetExternalTensorDataF32(input2_id);
  for (size_t i = 0; i < new_dim0; ++i) {
    expected[i] =
        (input0_data[i] + input1_data[i]) * (input0_data[i] + input2_data[i]);
  }
  ASSERT_EQ(expected, output);
}

TEST(RUNTIME, null_runtime) {
  size_t required_size = 0;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_get_runtime_profiling_info(nullptr,
                                           xnn_profile_info_num_operators, 0,
                                           nullptr, &required_size));
  EXPECT_EQ(xnn_status_invalid_parameter, xnn_invoke_runtime(nullptr));
}

#if SIZE_MAX <= UINT32_MAX
TEST(RUNTIME, create_runtime_overflow_num_nodes) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(1, 0, &subgraph));
  subgraph->num_nodes = static_cast<uint32_t>(
      SIZE_MAX / sizeof(struct xnn_operator_data) + 1);
  xnn_runtime_t runtime = nullptr;
  EXPECT_EQ(xnn_status_out_of_memory,
            xnn_create_runtime_v2(subgraph, nullptr, 0, &runtime));
  EXPECT_EQ(nullptr, runtime);
  subgraph->num_nodes = 0;
  xnn_delete_subgraph(subgraph);
}

TEST(RUNTIME, create_runtime_overflow_num_values) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(1, 0, &subgraph));
  subgraph->num_values = static_cast<uint32_t>(
      SIZE_MAX / sizeof(struct xnn_runtime_value) + 1);
  xnn_runtime_t runtime = nullptr;
  EXPECT_EQ(xnn_status_out_of_memory,
            xnn_create_runtime_v2(subgraph, nullptr, 0, &runtime));
  EXPECT_EQ(nullptr, runtime);
  subgraph->num_values = 1;
  xnn_delete_subgraph(subgraph);
}
#endif  // SIZE_MAX <= UINT32_MAX

extern "C" enum xnn_status xnn_plan_memory(xnn_runtime_t runtime);

TEST(RUNTIME, plan_memory_total_values_and_ops_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_runtime runtime = {};
  runtime.num_values = XNN_INVALID_VALUE_ID;
  runtime.num_ops = 1;
  EXPECT_EQ(xnn_status_out_of_memory, xnn_plan_memory(&runtime));
}
