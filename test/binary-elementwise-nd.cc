// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "replicable_random_device.h"

enum class RunMode {
  kCreateReshapeRun,
  kEager,
};

template <typename T>
xnn_datatype datatype_of() {
  if (std::is_same<T, uint8_t>::value) {
    return xnn_datatype_quint8;
  } else if (std::is_same<T, int8_t>::value) {
    return xnn_datatype_qint8;
  } else if (std::is_same<T, xnn_float16>::value) {
    return xnn_datatype_fp16;
  } else if (std::is_same<T, float>::value) {
    return xnn_datatype_fp32;
  } else if (std::is_same<T, int32_t>::value) {
    return xnn_datatype_int32;
  } else {
    XNN_UNREACHABLE;
  }
}

template <typename T>
double compute_tolerance(double output_ref) {
  if (std::is_integral<T>::value) {
    return 0.6;
  } else if (std::is_same<T, xnn_float16>::value) {
    return 1.0e-3 * std::abs(output_ref);
  } else {
    return 1.0e-6 * std::abs(output_ref);
  }
}

class BinaryElementwiseOperatorTester {
 public:
  static std::string ToString(xnn_binary_operator operation_type) {
    switch (operation_type) {
      case xnn_binary_invalid:
        return "Unknown";
      case xnn_binary_add:
        return "Add";
      case xnn_binary_copysign:
        return "CopySign";
      case xnn_binary_divide:
        return "Divide";
      case xnn_binary_maximum:
        return "Maximum";
      case xnn_binary_minimum:
        return "Minimum";
      case xnn_binary_multiply:
        return "Multiply";
      case xnn_binary_subtract:
        return "Subtract";
      case xnn_binary_squared_difference:
        return "SquaredDifference";
      default:
        return "Unknown";
    }
  }

  double Compute(double a, double b) const {
    switch (operation_type()) {
      case xnn_binary_add:
        return a + b;
      case xnn_binary_copysign:
        return std::copysign(a, b);
      case xnn_binary_divide:
        return a / b;
      case xnn_binary_maximum:
        return std::max(a, b);
      case xnn_binary_minimum:
        return std::min(a, b);
      case xnn_binary_multiply:
        return a * b;
      case xnn_binary_subtract:
        return a - b;
      case xnn_binary_squared_difference:
        return (a - b) * (a - b);
      default:
        return std::nanf("");
    }
  }

  BinaryElementwiseOperatorTester& input1_shape(
      std::vector<size_t> input1_shape) {
    assert(input1_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input1_shape_ = std::move(input1_shape);
    return *this;
  }

  const std::vector<size_t>& input1_shape() const {
    return this->input1_shape_;
  }

  size_t input1_dim(size_t i) const {
    return i < num_input1_dims() ? this->input1_shape_[i] : 1;
  }

  size_t num_input1_dims() const { return this->input1_shape_.size(); }

  size_t num_input1_elements() const {
    return std::accumulate(this->input1_shape_.begin(),
                           this->input1_shape_.end(), size_t(1),
                           std::multiplies<size_t>());
  }

  BinaryElementwiseOperatorTester& input1_zero_point(
      int32_t input1_zero_point) {
    this->input1_zero_point_ = input1_zero_point;
    return *this;
  }

  int32_t input1_zero_point() const { return this->input1_zero_point_; }

  BinaryElementwiseOperatorTester& input1_scale(float input1_scale) {
    assert(std::isfinite(input1_scale));
    this->input1_scale_ = input1_scale;
    return *this;
  }

  float input1_scale() const { return this->input1_scale_; }

  BinaryElementwiseOperatorTester& input2_shape(
      std::vector<size_t> input2_shape) {
    assert(input2_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input2_shape_ = std::move(input2_shape);
    return *this;
  }

  const std::vector<size_t>& input2_shape() const {
    return this->input2_shape_;
  }

  size_t input2_dim(size_t i) const {
    return i < num_input2_dims() ? this->input2_shape_[i] : 1;
  }

  size_t num_input2_dims() const { return this->input2_shape_.size(); }

  size_t num_input2_elements() const {
    return std::accumulate(this->input2_shape_.begin(),
                           this->input2_shape_.end(), size_t(1),
                           std::multiplies<size_t>());
  }

  BinaryElementwiseOperatorTester& input2_zero_point(
      int32_t input2_zero_point) {
    this->input2_zero_point_ = input2_zero_point;
    return *this;
  }

  int32_t input2_zero_point() const { return this->input2_zero_point_; }

  BinaryElementwiseOperatorTester& input2_scale(float input2_scale) {
    assert(std::isfinite(input2_scale));
    this->input2_scale_ = input2_scale;
    return *this;
  }

  float input2_scale() const { return this->input2_scale_; }

  BinaryElementwiseOperatorTester& output_zero_point(
      int32_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  int32_t output_zero_point() const { return this->output_zero_point_; }

  BinaryElementwiseOperatorTester& output_scale(float output_scale) {
    assert(std::isfinite(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const { return this->output_scale_; }

  BinaryElementwiseOperatorTester& operation_type(
      xnn_binary_operator operation_type) {
    this->operation_type_ = operation_type;
    return *this;
  }

  xnn_binary_operator operation_type() const { return this->operation_type_; }

  BinaryElementwiseOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  template <typename T>
  void Test(RunMode mode) {
    ASSERT_NE(operation_type(), xnn_binary_invalid);

    xnnpack::ReplicableRandomDevice rng;
    double input_min = std::is_integral<T>::value
                         ? static_cast<double>(std::numeric_limits<T>::min())
                         : 0.01;
    double input_max = std::is_integral<T>::value
                         ? static_cast<double>(std::numeric_limits<T>::max())
                         : 1.0;
    std::uniform_real_distribution<double> dist(input_min, input_max);

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input1_dims.begin(), input1_dims.end(), 1);
    std::fill(input2_dims.begin(), input2_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input1_shape().cbegin(), input1_shape().cend(),
              input1_dims.end() - num_input1_dims());
    std::copy(input2_shape().cbegin(), input2_shape().cend(),
              input2_dims.end() - num_input2_dims());
    for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
      if (input1_dims[i] != 1 && input2_dims[i] != 1) {
        ASSERT_EQ(input1_dims[i], input2_dims[i]);
      }
      output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
    }
    const size_t num_output_elements =
        std::accumulate(output_dims.begin(), output_dims.end(), size_t(1),
                        std::multiplies<size_t>());

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input1_stride = 1, input2_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input1_strides[i - 1] = input1_dims[i - 1] == 1 ? 0 : input1_stride;
      input2_strides[i - 1] = input2_dims[i - 1] == 1 ? 0 : input2_stride;
      output_strides[i - 1] = output_stride;
      input1_stride *= input1_dims[i - 1];
      input2_stride *= input2_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    xnn_datatype datatype = datatype_of<T>();
    xnn_quantization_params input1_quantization = {input1_zero_point(),
                                                   input1_scale()};
    xnn_quantization_params input2_quantization = {input2_zero_point(),
                                                   input2_scale()};
    xnn_quantization_params output_quantization = {output_zero_point(),
                                                   output_scale()};
    std::vector<T> input1(XNN_EXTRA_BYTES / sizeof(T) + num_input1_elements());
    std::vector<T> input2(XNN_EXTRA_BYTES / sizeof(T) + num_input2_elements());
    std::vector<T> output(num_output_elements);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input1.begin(), input1.end(), [&]() { return dist(rng); });
      std::generate(input2.begin(), input2.end(), [&]() { return dist(rng); });
      std::fill(output.begin(), output.end(), 0xAA);

      if (mode == RunMode::kCreateReshapeRun) {
        // Create, setup, run, and destroy a binary elementwise operator.
        ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
        xnn_operator_t binary_elementwise_op = nullptr;
        xnn_status status = xnn_create_binary_elementwise_nd(
            operation_type(), datatype, &input1_quantization,
            &input2_quantization, &output_quantization, 0,
            &binary_elementwise_op);
        if (status == xnn_status_unsupported_hardware) {
          GTEST_SKIP();
        }
        ASSERT_EQ(xnn_status_success, status);
        ASSERT_NE(nullptr, binary_elementwise_op);

        // Smart pointer to automatically delete binary_elementwise_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_binary_elementwise_op(binary_elementwise_op,
                                       xnn_delete_operator);

        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_binary_elementwise_nd(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_binary_elementwise_nd(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));

        ASSERT_EQ(xnn_status_success, xnn_run_operator(binary_elementwise_op,
                                                       /*threadpool=*/nullptr));
      } else if (mode == RunMode::kEager) {
        // Run a binary elementwise operator without creating it.
        xnn_status status = xnn_run_binary_elementwise_nd(
            operation_type(), datatype, &input1_quantization,
            &input2_quantization, &output_quantization, 0, input1_dims.size(),
            input1_dims.data(), input2_dims.size(), input2_dims.data(),
            input1.data(), input2.data(), output.data(),
            /*threadpool=*/nullptr);
        if (status == xnn_status_unsupported_hardware) {
          GTEST_SKIP();
        }
        ASSERT_EQ(xnn_status_success, status);
      } else {
        XNN_UNREACHABLE;
      }

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const double input1_value =
                      input1_scale() *
                      (input1[i * input1_strides[0] + j * input1_strides[1] +
                              k * input1_strides[2] + l * input1_strides[3] +
                              m * input1_strides[4] + n * input1_strides[5]] -
                       input1_zero_point());
                  const double input2_value =
                      input2_scale() *
                      (input2[i * input2_strides[0] + j * input2_strides[1] +
                              k * input2_strides[2] + l * input2_strides[3] +
                              m * input2_strides[4] + n * input2_strides[5]] -
                       input2_zero_point());
                  double output_ref =
                      Compute(input1_value, input2_value) / output_scale() +
                      output_zero_point();
                  const size_t index =
                      i * output_strides[0] + j * output_strides[1] +
                      k * output_strides[2] + l * output_strides[3] +
                      m * output_strides[4] + n * output_strides[5];
                  if (output_ref < std::numeric_limits<T>::lowest() ||
                      output_ref > std::numeric_limits<T>::max()) {
                    // This is expected to overflow.
                  } else {
                    const double tolerance = compute_tolerance<T>(output_ref);
                    ASSERT_NEAR(output[index], output_ref, tolerance)
                        << "input1_value = " << input1_value << ", "
                        << "input2_value = " << input2_value << ", "
                        << "(i, j, k, l, m, n) = (" << i << ", " << j << ", "
                        << k << ", " << l << ", " << m << ", " << n << ")"
                        << ", input1 zero point = " << input1_zero_point()
                        << ", input1 scale = " << input1_scale()
                        << ", input2 zero point = " << input2_zero_point()
                        << ", input2 scale = " << input2_scale()
                        << ", output zero point = " << output_zero_point()
                        << ", output scale = " << output_scale();
                  }
                }
              }
            }
          }
        }
      }
    }
  }

 private:
  std::vector<size_t> input1_shape_;
  std::vector<size_t> input2_shape_;
  int32_t input1_zero_point_{0};
  float input1_scale_{1.0f};
  int32_t input2_zero_point_{0};
  float input2_scale_{1.0f};
  int32_t output_zero_point_{0};
  float output_scale_{1.0f};
  xnn_binary_operator operation_type_{xnn_binary_invalid};
  size_t iterations_{3};
};

template <typename Rng>
std::vector<size_t> RandomShape(Rng& rng) {
  const size_t rank = rng() % XNN_MAX_TENSOR_DIMS;
  std::vector<size_t> dims(rank);
  for (size_t i = 0; i < rank; i++) {
    dims[i] = rng() % 10 + 1;
  }
  return dims;
}

template <typename Rng>
std::vector<size_t> RandomBroadcast(Rng& rng, std::vector<size_t> dims) {
  // Randomly assign some dimensions to 1.
  for (size_t i = 0; i < dims.size(); i++) {
    if (rng() % 8 == 0) {
      dims[i] = 1;
    }
  }
  // Possibly remove leading 1s.
  if (rng() % 2 == 0) {
    while (!dims.empty() && dims.front() == 1) {
      dims.erase(dims.begin());
    }
  }
  return dims;
}

template <typename T>
void RunBinaryOpTester(RunMode run_mode,
                       BinaryElementwiseOperatorTester& tester) {
  xnnpack::ReplicableRandomDevice rng;
  for (int iterations = 0; iterations < 100; iterations++) {
    std::vector<size_t> output_shape = RandomShape(rng);
    tester.input1_shape(RandomBroadcast(rng, output_shape))
        .input2_shape(RandomBroadcast(rng, output_shape));
    tester.Test<T>(run_mode);
  }
}

template <typename T, typename Params>
void BinaryNDTestImpl(const Params& params) {
  RunMode mode = std::get<0>(params);
  xnn_binary_operator op = std::get<1>(params);
  BinaryElementwiseOperatorTester tester;
  tester.operation_type(op);
  RunBinaryOpTester<T>(mode, tester);
}

template <typename T>
class BinaryNDTest
    : public testing::TestWithParam<
          std::tuple<RunMode, xnn_binary_operator>> {};

using BinaryNDTestQS8 = BinaryNDTest<int8_t>;
using BinaryNDTestQU8 = BinaryNDTest<uint8_t>;
#ifndef XNN_EXCLUDE_F16_TESTS
using BinaryNDTestF16 = BinaryNDTest<xnn_float16>;
#endif  // XNN_EXCLUDE_F16_TESTS
using BinaryNDTestF32 = BinaryNDTest<float>;
using BinaryNDTestS32 = BinaryNDTest<int32_t>;

TEST_P(BinaryNDTestQS8, op) { BinaryNDTestImpl<int8_t>(GetParam()); }
TEST_P(BinaryNDTestQU8, op) { BinaryNDTestImpl<uint8_t>(GetParam()); }
#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryNDTestF16, op) { BinaryNDTestImpl<xnn_float16>(GetParam()); }
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryNDTestF32, op) { BinaryNDTestImpl<float>(GetParam()); }
TEST_P(BinaryNDTestS32, op) { BinaryNDTestImpl<int32_t>(GetParam()); }

std::string ToString(const std::tuple<RunMode, xnn_binary_operator>& param) {
  return BinaryElementwiseOperatorTester::ToString(std::get<1>(param));
}

INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BinaryNDTestQS8,
    testing::Combine(testing::Values(RunMode::kCreateReshapeRun),
                     testing::Values(xnn_binary_add, xnn_binary_subtract,
                                     xnn_binary_multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(Eager, BinaryNDTestQS8,
                         testing::Combine(testing::Values(RunMode::kEager),
                                          testing::Values(xnn_binary_add,
                                                          xnn_binary_subtract,
                                                          xnn_binary_multiply)),
                         [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BinaryNDTestQU8,
    testing::Combine(testing::Values(RunMode::kCreateReshapeRun),
                     testing::Values(xnn_binary_add, xnn_binary_subtract,
                                     xnn_binary_multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(Eager, BinaryNDTestQU8,
                         testing::Combine(testing::Values(RunMode::kEager),
                                          testing::Values(xnn_binary_add,
                                                          xnn_binary_subtract,
                                                          xnn_binary_multiply)),
                         [](const auto& info) { return ToString(info.param); });
#ifndef XNN_EXCLUDE_F16_TESTS
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BinaryNDTestF16,
    testing::Combine(
        testing::Values(RunMode::kCreateReshapeRun),
        testing::Values(xnn_binary_add, xnn_binary_divide, xnn_binary_maximum,
                        xnn_binary_minimum, xnn_binary_multiply,
                        xnn_binary_squared_difference, xnn_binary_subtract)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BinaryNDTestF16,
    testing::Combine(
        testing::Values(RunMode::kEager),
        testing::Values(xnn_binary_add, xnn_binary_divide, xnn_binary_maximum,
                        xnn_binary_minimum, xnn_binary_multiply,
                        xnn_binary_squared_difference, xnn_binary_subtract)),
    [](const auto& info) { return ToString(info.param); });
#endif
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BinaryNDTestF32,
    testing::Combine(testing::Values(RunMode::kCreateReshapeRun),
                     testing::Values(xnn_binary_add, xnn_binary_copysign,
                                     xnn_binary_divide, xnn_binary_maximum,
                                     xnn_binary_minimum, xnn_binary_multiply,
                                     xnn_binary_subtract,
                                     xnn_binary_squared_difference)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    Eager, BinaryNDTestF32,
    testing::Combine(testing::Values(RunMode::kEager),
                     testing::Values(xnn_binary_add, xnn_binary_divide,
                                     xnn_binary_maximum, xnn_binary_minimum,
                                     xnn_binary_multiply, xnn_binary_subtract,
                                     xnn_binary_squared_difference)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, BinaryNDTestS32,
    testing::Combine(testing::Values(RunMode::kCreateReshapeRun),
                     testing::Values(xnn_binary_multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(Eager, BinaryNDTestS32,
                         testing::Combine(testing::Values(RunMode::kEager),
                                          testing::Values(xnn_binary_multiply)),
                         [](const auto& info) { return ToString(info.param); });

template <typename T, typename Params>
void QuantizedTest_Input1Scale(Params params) {
  for (float input1_scale = 0.1f; input1_scale <= 10.0f;
       input1_scale *= 3.14f) {
    RunBinaryOpTester<T>(std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input1_scale(input1_scale));
  }
}

template <typename T, typename Params>
void QuantizedTest_Input1ZeroPoint(Params params) {
  for (int32_t input1_zero_point = std::numeric_limits<T>::min();
       input1_zero_point <= std::numeric_limits<T>::max();
       input1_zero_point += 51) {
    RunBinaryOpTester<T>(std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input1_zero_point(input1_zero_point));
  }
}

template <typename T, typename Params>
void QuantizedTest_Input2Scale(Params params) {
  for (float input2_scale = 0.1f; input2_scale <= 10.0f;
       input2_scale *= 3.14f) {
    RunBinaryOpTester<T>(std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input2_scale(input2_scale));
  }
}

template <typename T, typename Params>
void QuantizedTest_Input2ZeroPoint(Params params) {
  for (int32_t input2_zero_point = std::numeric_limits<T>::min();
       input2_zero_point <= std::numeric_limits<T>::max();
       input2_zero_point += 51) {
    RunBinaryOpTester<T>(std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .input2_zero_point(input2_zero_point));
  }
}

template <typename T, typename Params>
void QuantizedTest_OutputScale(Params params) {
  for (float output_scale = 0.1f; output_scale <= 10.0f;
       output_scale *= 3.14f) {
    RunBinaryOpTester<T>(std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .output_scale(output_scale));
  }
}

template <typename T, typename Params>
void QuantizedTest_OutputZeroPoint(Params params) {
  for (int32_t output_zero_point = std::numeric_limits<T>::min();
       output_zero_point <= std::numeric_limits<T>::max();
       output_zero_point += 51) {
    RunBinaryOpTester<T>(std::get<0>(params),
                         BinaryElementwiseOperatorTester()
                             .operation_type(std::get<1>(params))
                             .output_zero_point(output_zero_point));
  }
}

template <typename T>
class QuantizedTest
    : public testing::TestWithParam<std::tuple<RunMode, xnn_binary_operator>> {
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

INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, QuantizedTestQS8,
    testing::Combine(testing::Values(RunMode::kCreateReshapeRun),
                     testing::Values(xnn_binary_add, xnn_binary_subtract,
                                     xnn_binary_multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(Eager, QuantizedTestQS8,
                         testing::Combine(testing::Values(RunMode::kEager),
                                          testing::Values(xnn_binary_add,
                                                          xnn_binary_subtract,
                                                          xnn_binary_multiply)),
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

INSTANTIATE_TEST_SUITE_P(
    CreateReshapeRun, QuantizedTestQU8,
    testing::Combine(testing::Values(RunMode::kCreateReshapeRun),
                     testing::Values(xnn_binary_add, xnn_binary_subtract,
                                     xnn_binary_multiply)),
    [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(Eager, QuantizedTestQU8,
                         testing::Combine(testing::Values(RunMode::kEager),
                                          testing::Values(xnn_binary_add,
                                                          xnn_binary_subtract,
                                                          xnn_binary_multiply)),
                         [](const auto& info) { return ToString(info.param); });
