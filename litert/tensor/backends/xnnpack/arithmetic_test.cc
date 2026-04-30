/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "litert/tensor/backends/xnnpack/arithmetic.h"

#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "litert/tensor/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/runners/xnnpack/runner.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/matchers.h"

namespace litert::tensor {
namespace {

using XnnTensor = Tensor<XnnpackMixinTag>;

// Check XNNPACK specific flag behavior during lowering.
TEST(ArithmeticXnnpackTest, AddBuildsExternalFlags) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {3}});
  XnnTensor bias({.name = "bias",
                  .type = Type::kFP32,
                  .shape = {3},
                  .buffer = std::vector<float>{1.f, 2.f, 3.f}});
  XnnTensor output = Add(input, bias);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto graph, BuildXnnpackGraph({output}));
  EXPECT_EQ(graph->values().size(), 3);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t input_index,
                                  graph->Lookup(input.GetRaw()));
  const XnnpackValue& input_value = graph->values()[input_index];
  EXPECT_NE(input_value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT, 0);
  EXPECT_EQ(input_value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT, 0);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t bias_index,
                                  graph->Lookup(bias.GetRaw()));
  const XnnpackValue& bias_value = graph->values()[bias_index];
  EXPECT_EQ(bias_value.flags, 0);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(size_t output_index,
                                  graph->Lookup(output.GetRaw()));
  const XnnpackValue& output_value = graph->values()[output_index];
  EXPECT_NE(output_value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT, 0);
}

TEST(ArithmeticXnnpackTest, SoftmaxRejectsNonUnitBeta) {
  XnnTensor input({.name = "input",
                   .type = Type::kFP32,
                   .shape = {2},
                   .buffer = std::vector<float>{0.f, 1.f}});

  XnnTensor output = Softmax(input, /*beta=*/2.0f);

  absl::Status status = BuildXnnpackGraph({output}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

TEST(ArithmeticXnnpackTest, GatherUnimplemented) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor indices({.name = "indices", .type = Type::kI32, .shape = {2}});
  XnnTensor output = Gather(input, indices, 0);

  absl::Status status = BuildXnnpackGraph({output}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

TEST(ArithmeticXnnpackTest, SpaceToDepthWorks) {
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 4, 4, 1}});
  XnnTensor output = SpaceToDepth(input, 2);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                   7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> expected_output = {
      1.0f, 2.0f,  5.0f,  6.0f,  3.0f,  4.0f,  7.0f,  8.0f,
      9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, DepthToSpaceWorks) {
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 4}});
  XnnTensor output = DepthToSpace(input, 2);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f,  2.0f,  5.0f,  6.0f,  3.0f,  4.0f,
                                   7.0f,  8.0f,  9.0f,  10.0f, 13.0f, 14.0f,
                                   11.0f, 12.0f, 15.0f, 16.0f};
  std::vector<float> expected_output = {
      1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, SplitWorks) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 4, 6}});
  std::vector<XnnTensor> outputs = Split(input, 1, 2);

  std::vector<TensorHandle> outputs_handles;
  for (auto& output : outputs) {
    outputs_handles.push_back(output);
  }
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                  XnnpackRunner::Create(outputs_handles));

  std::vector<float> input_data;
  for (int i = 0; i < 2 * 4 * 6; ++i) {
    input_data.push_back(i + 1);
  }

  std::vector<float> expected_output_0 = {1,  2,  3,  4,  5,  6,  7,  8,
                                          9,  10, 11, 12, 25, 26, 27, 28,
                                          29, 30, 31, 32, 33, 34, 35, 36};

  std::vector<float> expected_output_1 = {13, 14, 15, 16, 17, 18, 19, 20,
                                          21, 22, 23, 24, 37, 38, 39, 40,
                                          41, 42, 43, 44, 45, 46, 47, 48};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes_0,
                                  runner.ReadOutput(outputs[0]));
  auto output_data_0 =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes_0.data()),
                     output_bytes_0.size() / sizeof(float));
  EXPECT_THAT(output_data_0, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                  expected_output_0));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes_1,
                                  runner.ReadOutput(outputs[1]));
  auto output_data_1 =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes_1.data()),
                     output_bytes_1.size() / sizeof(float));
  EXPECT_THAT(output_data_1, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                  expected_output_1));
}

struct UnaryOpTestParams {
  std::string op_name;
  std::vector<float> input_data;
  std::vector<float> expected_output;
};

std::ostream& operator<<(std::ostream& os, const UnaryOpTestParams& params) {
  return os << "Op: " << params.op_name;
}

class UnaryOpNumericalTest
    : public ::testing::TestWithParam<UnaryOpTestParams> {
 protected:
  void RunTest(const XnnTensor& input, const XnnTensor& output,
               const std::vector<float>& input_data,
               const std::vector<float>& expected_output) {
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner,
                                    XnnpackRunner::Create({output}));
    absl::Span<const std::byte> input_bytes(
        reinterpret_cast<const std::byte*>(input_data.data()),
        input_data.size() * sizeof(float));
    ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
    ASSERT_THAT(runner.Run(), IsOk());
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes,
                                    runner.ReadOutput(output));
    auto output_data =
        absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                       output_bytes.size() / sizeof(float));

    EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                  expected_output));
  }
};

TEST_P(UnaryOpNumericalTest, UnaryOps) {
  const UnaryOpTestParams& params = GetParam();
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor output;

  if (params.op_name == "Abs") {
    output = Abs(input);
  } else if (params.op_name == "Relu") {
    output = Relu(input);
  } else if (params.op_name == "Relu6") {
    output = Relu6(input);
  } else if (params.op_name == "LeakyRelu") {
    output = LeakyRelu(input, 0.2f);
  } else if (params.op_name == "Elu") {
    output = Elu(input);
  } else if (params.op_name == "Gelu") {
    output = Gelu(input);
  } else if (params.op_name == "HardSwish") {
    output = HardSwish(input);
  } else if (params.op_name == "Ceil") {
    output = Ceil(input);
  } else if (params.op_name == "Exp") {
    output = Exp(input);
  } else if (params.op_name == "Log") {
    output = Log(input);
  } else if (params.op_name == "Floor") {
    output = Floor(input);
  } else if (params.op_name == "Neg") {
    output = Neg(input);
  } else if (params.op_name == "Round") {
    output = Round(input);
  } else if (params.op_name == "Rsqrt") {
    output = Rsqrt(input);
  } else if (params.op_name == "Sin") {
    output = Sin(input);
  } else if (params.op_name == "Cos") {
    output = Cos(input);
  } else if (params.op_name == "Sqrt") {
    output = Sqrt(input);
  } else if (params.op_name == "Square") {
    output = Square(input);
  } else {
    FAIL() << "Unknown op: " << params.op_name;
  }
  RunTest(input, output, params.input_data, params.expected_output);
}

INSTANTIATE_TEST_SUITE_P(AbsOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Abs",
                             .input_data = {-1.0f, -2.0f, 3.0f, 4.0f},
                             .expected_output = {1.0f, 2.0f, 3.0f, 4.0f}}));

INSTANTIATE_TEST_SUITE_P(ReluOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Relu",
                             .input_data = {-1.0f, -2.0f, 3.0f, 4.0f},
                             .expected_output = {0.0f, 0.0f, 3.0f, 4.0f}}));

INSTANTIATE_TEST_SUITE_P(Relu6OpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Relu6",
                             .input_data = {-1.0f, -2.0f, 7.0f, 4.0f},
                             .expected_output = {0.0f, 0.0f, 6.0f, 4.0f}}));

INSTANTIATE_TEST_SUITE_P(LeakyReluOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "LeakyRelu",
                             .input_data = {-1.0f, -2.0f, 7.0f, 4.0f},
                             .expected_output = {-0.2f, -0.4f, 7.0f, 4.0f}}));

INSTANTIATE_TEST_SUITE_P(EluOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Elu",
                             .input_data = {-1.0f, -2.0f, 7.0f, 4.0f},
                             .expected_output = {std::expm1(-1.0f),
                                                 std::expm1(-2.0f), 7.0f,
                                                 4.0f}}));

INSTANTIATE_TEST_SUITE_P(GeluOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Gelu",
                             .input_data = {-1.0f, -2.0f, 7.0f, 4.0f},
                             .expected_output = {-0.158655f, -0.0455003f, 7.0f,
                                                 3.99987316f}}));

INSTANTIATE_TEST_SUITE_P(HardSwishOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "HardSwish",
                             .input_data = {-4.0f, -2.0f, 1.0f, 4.0f},
                             .expected_output = {0.0f, -0.33333334f, 0.6666667f,
                                                 4.0f}}));

INSTANTIATE_TEST_SUITE_P(CeilOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Ceil",
                             .input_data = {-1.5f, -2.1f, 3.4f, 4.9f},
                             .expected_output = {-1.0f, -2.0f, 4.0f, 5.0f}}));

INSTANTIATE_TEST_SUITE_P(ExpOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Exp",
                             .input_data = {0.0f, 1.0f, -1.0f, 0.5f},
                             .expected_output = {1.0f, 2.7182817f, 0.36787945f,
                                                 1.6487213f}}));

INSTANTIATE_TEST_SUITE_P(LogOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Log",
                             .input_data = {1.0f, 2.0f, 0.5f, 0.1f},
                             .expected_output = {0.0f, 0.6931472f, -0.6931472f,
                                                 -2.3025851f}}));

INSTANTIATE_TEST_SUITE_P(FloorOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Floor",
                             .input_data = {-1.5f, -2.1f, 3.4f, 4.9f},
                             .expected_output = {-2.0f, -3.0f, 3.0f, 4.0f}}));

INSTANTIATE_TEST_SUITE_P(NegOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Neg",
                             .input_data = {-1.0f, -2.0f, 3.0f, 4.0f},
                             .expected_output = {1.0f, 2.0f, -3.0f, -4.0f}}));

INSTANTIATE_TEST_SUITE_P(RoundOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Round",
                             .input_data = {-1.5f, -2.1f, 3.4f, 4.9f},
                             .expected_output = {-2.0f, -2.0f, 3.0f, 5.0f}}));

INSTANTIATE_TEST_SUITE_P(RsqrtOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Rsqrt",
                             .input_data = {1.0f, 4.0f, 0.25f, 0.01f},
                             .expected_output = {1.0f, 0.5f, 2.0f, 10.0f}}));

INSTANTIATE_TEST_SUITE_P(SinOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Sin",
                             .input_data = {0.0f, 1.5707964f, 3.1415927f,
                                            4.712389f},
                             .expected_output = {0.0f, 1.0f, 0.0f, -1.0f}}));

INSTANTIATE_TEST_SUITE_P(CosOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Cos",
                             .input_data = {0.0f, 1.5707964f, 3.1415927f,
                                            4.712389f},
                             .expected_output = {1.0f, 0.0f, -1.0f, 0.0f}}));

INSTANTIATE_TEST_SUITE_P(SqrtOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Sqrt",
                             .input_data = {1.0f, 4.0f, 0.25f, 0.01f},
                             .expected_output = {1.0f, 2.0f, 0.5f, 0.1f}}));

INSTANTIATE_TEST_SUITE_P(SquareOpTest, UnaryOpNumericalTest,
                         ::testing::Values(UnaryOpTestParams{
                             .op_name = "Square",
                             .input_data = {-1.0f, -2.0f, 3.0f, 4.0f},
                             .expected_output = {1.0f, 4.0f, 9.0f, 16.0f}}));

TEST(ArithmeticXnnpackTest, AveragePool2DNumericalTest) {
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  XnnTensor output =
      AveragePool2D(input, /*filter_height=*/2, /*filter_width=*/2,
                    /*stride_h=*/1, /*stride_w=*/1, kPaddingValid);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {2.5f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, ExpandDimsNumericalTest) {
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  XnnTensor output = ExpandDims(input, 1);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, SqueezeWorks) {
  XnnTensor input({.type = Type::kFP32, .shape = {1, 2, 1, 2}});
  XnnTensor output = Squeeze(input);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, SqueezeWithDimsWorks) {
  XnnTensor input({.type = Type::kFP32, .shape = {1, 2, 1, 2}});
  XnnTensor output = Squeeze(input, {0, 2});

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, MaxPool2DNumericalTest) {
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  XnnTensor output = MaxPool2D(input, /*filter_height=*/2, /*filter_width=*/2,
                               /*stride_h=*/1, /*stride_w=*/1, kPaddingValid);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected_output = {4.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, PReluNumericalTest) {
  XnnTensor input({.name = "input", .type = Type::kFP32, .shape = {2, 2}});
  XnnTensor alpha({.name = "alpha", .type = Type::kFP32, .shape = {2}});
  XnnTensor output = PRelu(input, alpha);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));
  std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f};
  std::vector<float> alpha_data = {0.1f, 0.2f};
  std::vector<float> expected_output = {-0.1f, 2.0f, -0.3f, 4.0f};

  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));
  absl::Span<const std::byte> alpha_bytes(
      reinterpret_cast<const std::byte*>(alpha_data.data()),
      alpha_data.size() * sizeof(float));

  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());
  ASSERT_THAT(runner.SetInput(alpha, alpha_bytes), IsOk());
  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  EXPECT_THAT(output_data, ::testing::Pointwise(::testing::FloatNear(1e-6),
                                                expected_output));
}

TEST(ArithmeticXnnpackTest, TransposeConvOp) {
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  XnnTensor filter({.name = "filter",
                    .type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(filter_data)});
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> bias_data = {0.0f};
  XnnTensor bias({.name = "bias",
                  .type = Type::kFP32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  std::vector<int> output_shape_vec = {1, 3, 3, 1};
  XnnTensor output =
      TransposeConv(filter, input, bias, output_shape_vec, kPaddingValid, 1, 1);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));
  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  std::vector<float> expected = {1.0f,  4.0f, 4.0f,  6.0f, 20.0f,
                                 16.0f, 9.0f, 24.0f, 16.0f};

  EXPECT_THAT(output_data,
              ::testing::Pointwise(::testing::FloatNear(1e-6), expected));
}

TEST(ArithmeticXnnpackTest, TransposeConv2DOp) {
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  XnnTensor filter({.name = "filter",
                    .type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(filter_data)});
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> bias_data = {0.0f};
  XnnTensor bias({.name = "bias",
                  .type = Type::kFP32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  std::vector<int> output_shape_vec = {1, 3, 3, 1};
  XnnTensor output = TransposeConv2D(filter, input, bias, output_shape_vec,
                                     kPaddingValid, 1, 1);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));
  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  std::vector<float> expected = {1.0f,  4.0f, 4.0f,  6.0f, 20.0f,
                                 16.0f, 9.0f, 24.0f, 16.0f};

  EXPECT_THAT(output_data,
              ::testing::Pointwise(::testing::FloatNear(1e-6), expected));
}

TEST(ArithmeticXnnpackTest, Conv2DOp) {
  XnnTensor input(
      {.name = "input", .type = Type::kFP32, .shape = {1, 2, 2, 1}});
  std::vector<float> filter_data = {1.0f, 2.0f, 3.0f, 4.0f};
  XnnTensor filter({.name = "filter",
                    .type = Type::kFP32,
                    .shape = {1, 2, 2, 1},
                    .buffer = OwningCpuBuffer::Copy<Type::kFP32>(filter_data)});
  std::vector<float> bias_data = {1.0f};
  XnnTensor bias({.name = "bias",
                  .type = Type::kFP32,
                  .shape = {1},
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(bias_data)});

  XnnTensor output = Conv2D(input, filter, bias, /*stride_h=*/1, /*stride_w=*/1,
                            kPaddingValid);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto runner, XnnpackRunner::Create({output}));

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  absl::Span<const std::byte> input_bytes(
      reinterpret_cast<const std::byte*>(input_data.data()),
      input_data.size() * sizeof(float));
  ASSERT_THAT(runner.SetInput(input, input_bytes), IsOk());

  ASSERT_THAT(runner.Run(), IsOk());

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(auto output_bytes, runner.ReadOutput(output));
  auto output_data =
      absl::MakeSpan(reinterpret_cast<const float*>(output_bytes.data()),
                     output_bytes.size() / sizeof(float));

  // input * filter + bias
  // 1*1 + 2*2 + 3*3 + 4*4 + 1 = 1 + 4 + 9 + 16 + 1 = 31
  EXPECT_THAT(output_data, ::testing::ElementsAreArray({31.0f}));
}

}  // namespace
}  // namespace litert::tensor
