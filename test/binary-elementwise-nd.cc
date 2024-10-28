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
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/datatype.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/operator-utils.h"
#include "operator-test-utils.h"
#include "replicable_random_device.h"

using ::testing::Combine;
using ::testing::Values;
using ::testing::ValuesIn;

enum class RunMode {
  kCreateReshapeRun,
  kEager,
};

double ComputeTolerance(xnn_datatype datatype, double output_ref) {
  switch (datatype) {
    case xnn_datatype_fp16:
      return std::abs(output_ref);
    case xnn_datatype_fp32:
      return 1.0e-6 * std::abs(output_ref);
    default:
      return 0.6;
  }
}

struct MinMaxLow {
  double min, max, low;
};

MinMaxLow DatatypeMinMaxLow(xnn_datatype datatype) {
  MinMaxLow result;
  switch (datatype) {
    case xnn_datatype_quint8:
      result.low = std::numeric_limits<uint8_t>::lowest();
      result.min = std::numeric_limits<uint8_t>::min();
      result.max = std::numeric_limits<uint8_t>::max();
      break;
    case xnn_datatype_qint8:
      result.low = std::numeric_limits<int8_t>::lowest();
      result.min = std::numeric_limits<int8_t>::min();
      result.max = std::numeric_limits<int8_t>::max();
      break;
    case xnn_datatype_int32:
      result.low = std::numeric_limits<int32_t>::lowest();
      result.min = std::numeric_limits<int32_t>::min();
      result.max = std::numeric_limits<int32_t>::max();
      break;
    case xnn_datatype_fp16:
      result.low = 0.0;  // don't use std::numeric_limits here
      result.min = 0.01;
      result.max = 1.0;
      break;
    case xnn_datatype_fp32:
      result.low = std::numeric_limits<float>::lowest();
      result.min = 0.01;
      result.max = 1.0;
      break;
    default:
      result.low = 0;
      result.min = 0;
      result.max = 0;
      assert(false);
      break;
  }
  return result;
}

class BinaryElementwiseOperatorTester {
 public:
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

  BinaryElementwiseOperatorTester& datatype(xnn_datatype datatype) {
    this->datatype_ = datatype;
    return *this;
  }

  xnn_datatype datatype() const { return this->datatype_; }

  BinaryElementwiseOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  // Some combinations aren't implemented.
  bool SupportedBinaryNDTest() const {
    switch (datatype()) {
      case xnn_datatype_quint8:
      case xnn_datatype_qint8:
        switch (operation_type()) {
          case xnn_binary_add:
          case xnn_binary_multiply:
          case xnn_binary_subtract:
            return true;
          default:
            return false;
        }
      case xnn_datatype_int32:
        switch (operation_type()) {
          case xnn_binary_multiply:
            return true;
          default:
            return false;
        }
      case xnn_datatype_fp16:
        switch (operation_type()) {
          case xnn_binary_add:
          case xnn_binary_divide:
          case xnn_binary_maximum:
          case xnn_binary_minimum:
          case xnn_binary_multiply:
          case xnn_binary_subtract:
          case xnn_binary_squared_difference:
            return true;
          default:
            return false;
        }
      case xnn_datatype_fp32:
        switch (operation_type()) {
          case xnn_binary_add:
          case xnn_binary_copysign:
          case xnn_binary_divide:
          case xnn_binary_maximum:
          case xnn_binary_minimum:
          case xnn_binary_multiply:
          case xnn_binary_subtract:
          case xnn_binary_squared_difference:
            return true;
          default:
            return false;
        }
      default:
        return false;
    }
  }

  void Test(RunMode mode) {
    ASSERT_NE(operation_type(), xnn_binary_invalid);
    ASSERT_NE(datatype(), xnn_datatype_invalid);

    MinMaxLow limits = DatatypeMinMaxLow(datatype());
    size_t datatype_size_bytes = xnn_datatype_size_bytes(datatype());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<double> dist(limits.min, limits.max);

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input1_dims.begin(), input1_dims.end(), 1);
    std::fill(input2_dims.begin(), input2_dims.end(), 1);
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

    xnn_quantization_params input1_quantization = {input1_zero_point(),
                                                   input1_scale()};
    xnn_quantization_params input2_quantization = {input2_zero_point(),
                                                   input2_scale()};
    xnn_quantization_params output_quantization = {output_zero_point(),
                                                   output_scale()};
    xnnpack::Buffer<char> input1(XNN_EXTRA_BYTES +
                                 num_input1_elements() * datatype_size_bytes);
    xnnpack::Buffer<char> input2(XNN_EXTRA_BYTES +
                                 num_input2_elements() * datatype_size_bytes);
    xnnpack::Buffer<char> output(num_output_elements * datatype_size_bytes);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      randomize_buffer(datatype(), rng, dist, input1);
      randomize_buffer(datatype(), rng, dist, input2);

      if (mode == RunMode::kCreateReshapeRun) {
        // Create, setup, run, and destroy a binary elementwise operator.
        ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
        xnn_operator_t binary_elementwise_op = nullptr;
        xnn_status status = xnn_create_binary_elementwise_nd(
            operation_type(), datatype(), &input1_quantization,
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
            operation_type(), datatype(), &input1_quantization,
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

      switch (datatype()) {
        case xnn_datatype_quint8:
          ValidateResults<uint8_t>(input1, input1_strides, input2,
                                   input2_strides, output, output_strides,
                                   output_dims);
          break;
        case xnn_datatype_qint8:
          ValidateResults<int8_t>(input1, input1_strides, input2,
                                  input2_strides, output, output_strides,
                                  output_dims);
          break;
        case xnn_datatype_int32:
          ValidateResults<int32_t>(input1, input1_strides, input2,
                                   input2_strides, output, output_strides,
                                   output_dims);
          break;
        case xnn_datatype_fp16:
          ValidateResults<xnn_float16>(input1, input1_strides, input2,
                                       input2_strides, output, output_strides,
                                       output_dims);
          break;
        case xnn_datatype_fp32:
          ValidateResults<float>(input1, input1_strides, input2, input2_strides,
                                 output, output_strides, output_dims);
          break;
        default:
          assert(false);
      }
    }
  }

  template <typename T>
  void ValidateResults(
      const xnnpack::Buffer<char>& input1,
      const std::array<size_t, XNN_MAX_TENSOR_DIMS>& input1_strides,
      const xnnpack::Buffer<char>& input2,
      const std::array<size_t, XNN_MAX_TENSOR_DIMS>& input2_strides,
      const xnnpack::Buffer<char>& output,
      const std::array<size_t, XNN_MAX_TENSOR_DIMS>& output_strides,
      const std::array<size_t, XNN_MAX_TENSOR_DIMS>& output_dims) {
    // Verify results.
    MinMaxLow limits = DatatypeMinMaxLow(datatype());
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t input1_index =
                    i * input1_strides[0] + j * input1_strides[1] +
                    k * input1_strides[2] + l * input1_strides[3] +
                    m * input1_strides[4] + n * input1_strides[5];
                const double input1_value =
                    input1_scale() *
                    (reinterpret_cast<const T*>(&input1[0])[input1_index] -
                     input1_zero_point());
                const size_t input2_index =
                    i * input2_strides[0] + j * input2_strides[1] +
                    k * input2_strides[2] + l * input2_strides[3] +
                    m * input2_strides[4] + n * input2_strides[5];
                const double input2_value =
                    input2_scale() *
                    (reinterpret_cast<const T*>(&input2[0])[input2_index] -
                     input2_zero_point());
                double output_ref =
                    Compute(input1_value, input2_value) / output_scale() +
                    output_zero_point();
                if (output_ref < limits.low || output_ref > limits.max) {
                  // This is expected to overflow.
                } else {
                  const double tolerance =
                      ComputeTolerance(datatype(), output_ref);
                  const size_t output_index =
                      i * output_strides[0] + j * output_strides[1] +
                      k * output_strides[2] + l * output_strides[3] +
                      m * output_strides[4] + n * output_strides[5];
                  const double output_value =
                      reinterpret_cast<const T*>(&output[0])[output_index];
                  ASSERT_NEAR(output_value, output_ref, tolerance)
                      << "input1_value = " << input1_value << ", "
                      << "input2_value = " << input2_value << ", "
                      << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k
                      << ", " << l << ", " << m << ", " << n << ")"
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
  xnn_datatype datatype_{xnn_datatype_invalid};
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

void RunBinaryOpTester(RunMode run_mode,
                       BinaryElementwiseOperatorTester& tester) {
  xnnpack::ReplicableRandomDevice rng;
  for (int iterations = 0; iterations < 100; iterations++) {
    std::vector<size_t> output_shape = RandomShape(rng);
    tester.input1_shape(RandomBroadcast(rng, output_shape))
        .input2_shape(RandomBroadcast(rng, output_shape));
    tester.Test(run_mode);
  }
}

struct Param {
  using TupleT = std::tuple<xnn_datatype, RunMode, xnn_binary_operator>;
  explicit Param(TupleT p)
      : datatype(std::get<0>(p)),
        run_mode(std::get<1>(p)),
        binary_operator(std::get<2>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    if (run_mode == RunMode::kEager) {
      sstr << "Eager_";
    } else {
      sstr << "CreateReshapeRun_";
    }
    sstr << xnn_datatype_to_string(datatype) << "_"
         << xnn_binary_operator_to_string(binary_operator);
    return sstr.str();
  }

  xnn_datatype datatype;
  RunMode run_mode;
  xnn_binary_operator binary_operator;
};

class BinaryNDTest : public testing::TestWithParam<Param> {};

TEST_P(BinaryNDTest, op) {
  BinaryElementwiseOperatorTester tester;
  tester.operation_type(GetParam().binary_operator);
  tester.datatype(GetParam().datatype);
#ifdef XNN_EXCLUDE_F16_TESTS
  if (GetParam().datatype == xnn_datatype_fp16) {
    GTEST_SKIP();
  }
#endif
  if (!tester.SupportedBinaryNDTest()) {
    GTEST_SKIP();
  }
  RunBinaryOpTester(GetParam().run_mode, tester);
}

const xnn_binary_operator all_binary_ops[] = {
    xnn_binary_add,     xnn_binary_copysign, xnn_binary_divide,
    xnn_binary_maximum, xnn_binary_minimum,  xnn_binary_multiply,
    xnn_binary_prelu,   xnn_binary_subtract, xnn_binary_squared_difference};

// We do the full Cartesian combination here, but some are inappropriate
// and will be skipped for certain combinations -- see SupportedBinaryNDTest
// to see the logic for what is actually supported.
INSTANTIATE_TEST_SUITE_P(
    BinaryNDTest, BinaryNDTest,
    testing::ConvertGenerator<Param::TupleT>(Combine(
        Values(xnn_datatype_quint8, xnn_datatype_qint8, xnn_datatype_fp16,
               xnn_datatype_fp32, xnn_datatype_int32),
        Values(RunMode::kCreateReshapeRun, RunMode::kEager),
        ValuesIn(all_binary_ops))),
    [](const auto& info) { return info.param.Name(); });

class QuantizedTest : public testing::TestWithParam<Param> {
 public:
  // Some combinations aren't implemented.
  bool SupportedQuantizedTest(xnn_binary_operator op) const {
    switch (op) {
      case xnn_binary_add:
      case xnn_binary_multiply:
      case xnn_binary_subtract:
        return true;
      default:
        return false;
    }
  }
};

TEST_P(QuantizedTest, input1_scale) {
  if (!SupportedQuantizedTest(GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  for (float input1_scale = 0.1f; input1_scale <= 10.0f;
       input1_scale *= 3.14f) {
    RunBinaryOpTester(GetParam().run_mode,
                      BinaryElementwiseOperatorTester()
                          .operation_type(GetParam().binary_operator)
                          .datatype(GetParam().datatype)
                          .input1_scale(input1_scale));
  }
}

TEST_P(QuantizedTest, input1_zero_point) {
  if (!SupportedQuantizedTest(GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  MinMaxLow limits = DatatypeMinMaxLow(GetParam().datatype);
  for (int32_t input1_zero_point = limits.min; input1_zero_point <= limits.max;
       input1_zero_point += 51) {
    RunBinaryOpTester(GetParam().run_mode,
                      BinaryElementwiseOperatorTester()
                          .operation_type(GetParam().binary_operator)
                          .datatype(GetParam().datatype)
                          .input1_zero_point(input1_zero_point));
  }
}

TEST_P(QuantizedTest, input2_scale) {
  if (!SupportedQuantizedTest(GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  for (float input2_scale = 0.1f; input2_scale <= 10.0f;
       input2_scale *= 3.14f) {
    RunBinaryOpTester(GetParam().run_mode,
                      BinaryElementwiseOperatorTester()
                          .operation_type(GetParam().binary_operator)
                          .datatype(GetParam().datatype)
                          .input2_scale(input2_scale));
  }
}

TEST_P(QuantizedTest, input2_zero_point) {
  if (!SupportedQuantizedTest(GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  MinMaxLow limits = DatatypeMinMaxLow(GetParam().datatype);
  for (int32_t input2_zero_point = limits.min; input2_zero_point <= limits.max;
       input2_zero_point += 51) {
    RunBinaryOpTester(GetParam().run_mode,
                      BinaryElementwiseOperatorTester()
                          .operation_type(GetParam().binary_operator)
                          .datatype(GetParam().datatype)
                          .input2_zero_point(input2_zero_point));
  }
}

TEST_P(QuantizedTest, output_scale) {
  if (!SupportedQuantizedTest(GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  for (float output_scale = 0.1f; output_scale <= 10.0f;
       output_scale *= 3.14f) {
    RunBinaryOpTester(GetParam().run_mode,
                      BinaryElementwiseOperatorTester()
                          .operation_type(GetParam().binary_operator)
                          .datatype(GetParam().datatype)
                          .output_scale(output_scale));
  }
}

TEST_P(QuantizedTest, output_zero_point) {
  if (!SupportedQuantizedTest(GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  MinMaxLow limits = DatatypeMinMaxLow(GetParam().datatype);
  for (int32_t output_zero_point = limits.min; output_zero_point <= limits.max;
       output_zero_point += 51) {
    RunBinaryOpTester(GetParam().run_mode,
                      BinaryElementwiseOperatorTester()
                          .operation_type(GetParam().binary_operator)
                          .datatype(GetParam().datatype)
                          .output_zero_point(output_zero_point));
  }
}

INSTANTIATE_TEST_SUITE_P(QuantizedTest, QuantizedTest,
                         testing::ConvertGenerator<Param::TupleT>(Combine(
                             Values(xnn_datatype_quint8, xnn_datatype_qint8),
                             Values(RunMode::kCreateReshapeRun,
                                    RunMode::kEager),
                             ValuesIn(all_binary_ops))),
                         [](const auto& info) { return info.param.Name(); });
