#include <vector>

#include "runtime-tester.h"

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
  tester
    .AddInputTensorF32({dim0}, input0_id)
    .AddInputTensorF32({dim0}, input1_id)
    .AddInputTensorF32({dim0}, input2_id)
    .AddOutputTensorF32({dim0}, output_id)
    .AddInternalDynamicTensorF32({dummy_internal_dim}, &add1_out)
    .AddInternalDynamicTensorF32({dummy_internal_dim}, &add2_out)
    .AddAddition(input0_id, input1_id, add1_out)
    .AddAddition(input0_id, input2_id, add2_out)
    .AddMultiply(add1_out, add2_out, output_id);

  std::vector<float> expected(dim0);
  const float* input0_data = tester.GetExternalTensorDataF32(input0_id);
  const float* input1_data = tester.GetExternalTensorDataF32(input1_id);
  const float* input2_data = tester.GetExternalTensorDataF32(input2_id);
  for (size_t i = 0; i < dim0; ++i) {
    expected[i] = (input0_data[i] + input1_data[i]) * (input0_data[i] + input2_data[i]);
  }
  auto output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(expected, output);

  tester.ReshapeInput({new_dim0}, input0_id);
  tester.ReshapeInput({new_dim0}, input1_id);
  tester.ReshapeInput({new_dim0}, input2_id);

  tester.ReshapeRuntime();
  tester.SetupRuntimeV2();

  output = tester.RepeatRun<float>();
  expected.resize(new_dim0);
  input0_data = tester.GetExternalTensorDataF32(input0_id);
  input1_data = tester.GetExternalTensorDataF32(input1_id);
  input2_data = tester.GetExternalTensorDataF32(input2_id);
  for (size_t i = 0; i < new_dim0; ++i) {
    expected[i] = (input0_data[i] + input1_data[i]) * (input0_data[i] + input2_data[i]);
  }
  ASSERT_EQ(expected, output);
}
