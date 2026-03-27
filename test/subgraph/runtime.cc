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

TEST(RUNTIME, reshape_external_value_rejects_rank_above_max_tensor_dims) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));

  xnnpack::RuntimeTester tester(2);
  constexpr uint32_t input_id = 0;
  constexpr uint32_t output_id = 1;
  tester.AddInputTensorF32({1}, input_id)
      .AddOutputTensorF32({1}, output_id)
      .AddCopy(input_id, output_id);
  tester.CreateRuntime(xnn_test_runtime_flags());

  const size_t dims[XNN_MAX_TENSOR_DIMS + 1] = {1, 1, 1, 1, 1, 1, 1};
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_reshape_external_value(tester.Runtime(), input_id,
                                       XNN_MAX_TENSOR_DIMS + 1, dims));
}
