// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include "binary-elementwise-operator-tester.h"

constexpr size_t kDim1 = 2;
constexpr size_t kDim2 = 3;
constexpr size_t kDim3 = 4;
constexpr size_t kDim4 = 5;
constexpr size_t kDim5 = 6;
constexpr size_t kDim6 = 7;
const size_t kDims[] = {kDim1, kDim2, kDim3, kDim4, kDim5, kDim6};

const size_t kBroadcastRanks[] = {0, 1, 2, 3, 4, 5, 6};
const size_t kTestRank = 4;

template <typename T, typename Params>
void BroadcastNDTestImpl(const Params& params) {
  RunMode mode = std::get<0>(params);
  BinaryElementwiseOperatorTester::OperationType op = std::get<1>(params);
  const size_t rank_a = std::get<2>(params);
  const size_t rank_b = std::get<3>(params);
  BinaryElementwiseOperatorTester tester;
  tester.operation_type(op);
  if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    // TODO(dsharlet): This is a lame way to do this. The tester needs to be
    // refactored to not require this.
    tester.qmin(std::numeric_limits<T>::min());
    tester.qmax(std::numeric_limits<T>::max());
  }
  RunBinaryOpTester<T>(rank_a, rank_b, kDims, mode, tester);
}

template <typename T>
class BroadcastNDTest
    : public testing::TestWithParam<
          std::tuple<RunMode, BinaryElementwiseOperatorTester::OperationType,
                     size_t, size_t>> {};

using BroadcastNDTestQS8 = BroadcastNDTest<int8_t>;
using BroadcastNDTestQU8 = BroadcastNDTest<uint8_t>;
#ifndef XNN_EXCLUDE_F16_TESTS
using BroadcastNDTestF16 = BroadcastNDTest<xnn_float16>;
#endif  // XNN_EXCLUDE_F16_TESTS
using BroadcastNDTestF32 = BroadcastNDTest<float>;
using BroadcastNDTestS32 = BroadcastNDTest<int32_t>;

TEST_P(BroadcastNDTestQS8, op) { BroadcastNDTestImpl<int8_t>(GetParam()); }
TEST_P(BroadcastNDTestQU8, op) { BroadcastNDTestImpl<uint8_t>(GetParam()); }
#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BroadcastNDTestF16, op) { BroadcastNDTestImpl<xnn_float16>(GetParam()); }
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BroadcastNDTestF32, op) { BroadcastNDTestImpl<float>(GetParam()); }
TEST_P(BroadcastNDTestS32, op) { BroadcastNDTestImpl<int32_t>(GetParam()); }

std::string ToString(
    const std::tuple<RunMode, BinaryElementwiseOperatorTester::OperationType,
                     size_t, size_t>& param) {
  return BinaryElementwiseOperatorTester::ToString(std::get<1>(param)) + "_" +
         std::to_string(std::get<2>(param)) + "d_x_" +
         std::to_string(std::get<3>(param)) + "d";
}

std::string ToString(
    const std::tuple<RunMode, BinaryElementwiseOperatorTester::OperationType>&
        param) {
  return BinaryElementwiseOperatorTester::ToString(std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BroadcastNDTestQS8,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BroadcastNDTestQS8,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BroadcastNDTestQU8,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BroadcastNDTestQU8,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
#ifndef XNN_EXCLUDE_F16_TESTS
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BroadcastNDTestF16,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Maximum,
            BinaryElementwiseOperatorTester::OperationType::Minimum,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Prelu,
            BinaryElementwiseOperatorTester::OperationType::SquaredDifference,
            BinaryElementwiseOperatorTester::OperationType::Subtract),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BroadcastNDTestF16,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Maximum,
            BinaryElementwiseOperatorTester::OperationType::Minimum,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Prelu,
            BinaryElementwiseOperatorTester::OperationType::SquaredDifference,
            BinaryElementwiseOperatorTester::OperationType::Subtract),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
#endif
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BroadcastNDTestF32,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::CopySign,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Maximum,
            BinaryElementwiseOperatorTester::OperationType::Minimum,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Prelu,
            BinaryElementwiseOperatorTester::OperationType::SquaredDifference,
            BinaryElementwiseOperatorTester::OperationType::Subtract),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BroadcastNDTestF32,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Maximum,
            BinaryElementwiseOperatorTester::OperationType::Minimum,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Prelu,
            BinaryElementwiseOperatorTester::OperationType::SquaredDifference,
            BinaryElementwiseOperatorTester::OperationType::Subtract),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BroadcastNDTestS32,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Multiply),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BroadcastNDTestS32,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Multiply),
        testing::ValuesIn(kBroadcastRanks), testing::ValuesIn(kBroadcastRanks)),
    [](const auto& info) { return ToString(info.param); });

template <typename T, typename Params>
void FloatMinTestImpl(Params params) {
  for (int32_t qmin = std::numeric_limits<int16_t>::max() - 1000;
       qmin > std::numeric_limits<int16_t>::min(); qmin -= 5000) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .qmin(qmin));
  }
}

template <typename T, typename Params>
void FloatMaxTestImpl(Params params) {
  for (int32_t qmax = std::numeric_limits<int16_t>::min() + 1000;
       qmax < std::numeric_limits<int16_t>::max(); qmax += 5000) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .qmax(qmax));
  }
}

template <typename T>
class FloatMinMaxTest
    : public testing::TestWithParam<
          std::tuple<RunMode, BinaryElementwiseOperatorTester::OperationType>> {
};

#ifndef XNN_EXCLUDE_F16_TESTS
using FloatMinMaxTestNDF16 = FloatMinMaxTest<xnn_float16>;
TEST_P(FloatMinMaxTestNDF16, qmin) { FloatMinTestImpl<xnn_float16>(GetParam()); }
TEST_P(FloatMinMaxTestNDF16, qmax) { FloatMaxTestImpl<xnn_float16>(GetParam()); }
#endif  // XNN_EXCLUDE_F16_TESTS

using FloatMinMaxTestNDF32 = FloatMinMaxTest<float>;
TEST_P(FloatMinMaxTestNDF32, qmin) { FloatMinTestImpl<float>(GetParam()); }
TEST_P(FloatMinMaxTestNDF32, qmax) { FloatMaxTestImpl<float>(GetParam()); }

#ifndef XNN_EXCLUDE_F16_TESTS
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, FloatMinMaxTestNDF16,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Subtract)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, FloatMinMaxTestNDF16,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Subtract)),
    [](const auto& info) { return ToString(info.param); });
#endif
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, FloatMinMaxTestNDF32,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Subtract)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, FloatMinMaxTestNDF32,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Divide,
            BinaryElementwiseOperatorTester::OperationType::Multiply,
            BinaryElementwiseOperatorTester::OperationType::Subtract)),
    [](const auto& info) { return ToString(info.param); });

template <typename T, typename Params>
void QuantizedTest_Input1Scale(Params params) {
  for (float input1_scale = 0.1f; input1_scale <= 10.0f;
       input1_scale *= 3.14f) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input1_scale(input1_scale)
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(std::numeric_limits<T>::max()));
  }
}

template <typename T, typename Params>
void QuantizedTest_Input1ZeroPoint(Params params) {
  for (int16_t input1_zero_point = std::numeric_limits<T>::min();
       input1_zero_point <= std::numeric_limits<T>::max();
       input1_zero_point += 51) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input1_zero_point(input1_zero_point)
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(std::numeric_limits<T>::max()));
  }
}

template <typename T, typename Params>
void QuantizedTest_Input2Scale(Params params) {
  for (float input2_scale = 0.1f; input2_scale <= 10.0f;
       input2_scale *= 3.14f) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input2_scale(input2_scale)
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(std::numeric_limits<T>::max()));
  }
}

template <typename T, typename Params>
void QuantizedTest_Input2ZeroPoint(Params params) {
  for (int16_t input2_zero_point = std::numeric_limits<T>::min();
       input2_zero_point <= std::numeric_limits<T>::max();
       input2_zero_point += 51) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input2_zero_point(input2_zero_point)
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(std::numeric_limits<T>::max()));
  }
}

template <typename T, typename Params>
void QuantizedTest_OutputScale(Params params) {
  for (float output_scale = 0.1f; output_scale <= 10.0f;
       output_scale *= 3.14f) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .output_scale(output_scale)
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(std::numeric_limits<T>::max()));
  }
}

template <typename T, typename Params>
void QuantizedTest_OutputZeroPoint(Params params) {
  for (int16_t output_zero_point = std::numeric_limits<T>::min();
       output_zero_point <= std::numeric_limits<T>::max();
       output_zero_point += 51) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .output_zero_point(output_zero_point)
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(std::numeric_limits<T>::max()));
  }
}

template <typename T, typename Params>
void QuantizedTest_Qmin(Params params) {
  for (int16_t qmin = std::numeric_limits<T>::max() - 1;
       qmin > std::numeric_limits<T>::min(); qmin -= 50) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .qmin(qmin)
                             .qmax(std::numeric_limits<T>::max()));
  }
}
template <typename T, typename Params>
void QuantizedTest_Qmax(Params params) {
  for (int16_t qmax = std::numeric_limits<T>::min() + 1;
       qmax < std::numeric_limits<T>::max(); qmax += 50) {
    RunBinaryOpTester<T>(kTestRank, kTestRank, kDims, std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .qmin(std::numeric_limits<T>::min())
                             .qmax(qmax));
  }
}

template <typename T>
class QuantizedTest
    : public testing::TestWithParam<
          std::tuple<RunMode, BinaryElementwiseOperatorTester::OperationType>> {
};

using QuantizedTestQS8 = QuantizedTest<int8_t>;

TEST_P(QuantizedTestQS8, input1_scale) {
  QuantizedTest_Input1Scale<int8_t>(GetParam());
}
TEST_P(QuantizedTestQS8, input1_zero_point) {
  QuantizedTest_Input1ZeroPoint<int8_t>(GetParam());
}
TEST_P(QuantizedTestQS8, input2_scale) {
  QuantizedTest_Input2Scale<int8_t>(GetParam());
}
TEST_P(QuantizedTestQS8, input2_zero_point) {
  QuantizedTest_Input2ZeroPoint<int8_t>(GetParam());
}

TEST_P(QuantizedTestQS8, output_scale) {
  QuantizedTest_OutputScale<int8_t>(GetParam());
}
TEST_P(QuantizedTestQS8, output_zero_point) {
  QuantizedTest_OutputZeroPoint<int8_t>(GetParam());
}

TEST_P(QuantizedTestQS8, qmin) { QuantizedTest_Qmin<int8_t>(GetParam()); }
TEST_P(QuantizedTestQS8, qmax) { QuantizedTest_Qmax<int8_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, QuantizedTestQS8,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, QuantizedTestQS8,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply)),
    [](const auto& info) { return ToString(info.param); });

using QuantizedTestQU8 = QuantizedTest<uint8_t>;

TEST_P(QuantizedTestQU8, input1_scale) {
  QuantizedTest_Input1Scale<uint8_t>(GetParam());
}
TEST_P(QuantizedTestQU8, input1_zero_point) {
  QuantizedTest_Input1ZeroPoint<uint8_t>(GetParam());
}
TEST_P(QuantizedTestQU8, input2_scale) {
  QuantizedTest_Input2Scale<uint8_t>(GetParam());
}
TEST_P(QuantizedTestQU8, input2_zero_point) {
  QuantizedTest_Input2ZeroPoint<uint8_t>(GetParam());
}

TEST_P(QuantizedTestQU8, output_scale) {
  QuantizedTest_OutputScale<uint8_t>(GetParam());
}
TEST_P(QuantizedTestQU8, output_zero_point) {
  QuantizedTest_OutputZeroPoint<uint8_t>(GetParam());
}

TEST_P(QuantizedTestQU8, qmin) { QuantizedTest_Qmin<uint8_t>(GetParam()); }
TEST_P(QuantizedTestQU8, qmax) { QuantizedTest_Qmax<uint8_t>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, QuantizedTestQU8,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, QuantizedTestQU8,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(
            BinaryElementwiseOperatorTester::OperationType::Add,
            BinaryElementwiseOperatorTester::OperationType::Subtract,
            BinaryElementwiseOperatorTester::OperationType::Multiply)),
    [](const auto& info) { return ToString(info.param); });
