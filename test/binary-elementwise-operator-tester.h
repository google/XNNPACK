// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"

class BinaryElementwiseOperatorTester {
 public:
  enum class OperationType {
    Unknown,
    Add,
    CopySign,
    Divide,
    Maximum,
    Minimum,
    Multiply,
    Prelu,
    SquaredDifference,
    Subtract,
  };

  static std::string ToString(OperationType operation_type) {
    switch (operation_type) {
      case OperationType::Unknown:
        return "Unknown";
      case OperationType::Add:
        return "Add";
      case OperationType::CopySign:
        return "CopySign";
      case OperationType::Divide:
        return "Divide";
      case OperationType::Maximum:
        return "Maximum";
      case OperationType::Minimum:
        return "Minimum";
      case OperationType::Multiply:
        return "Multiply";
      case OperationType::Prelu:
        return "Prelu";
      case OperationType::SquaredDifference:
        return "SquaredDifference";
      case OperationType::Subtract:
        return "Subtract";
      default:
        return "Unknown";
    }
  }

  template <typename T>
  void CheckResults(const size_t* output_dims, const T* input1, const T* input2,
                    const T* output, const size_t* input1_strides,
                    const size_t* input2_strides,
                    const size_t* output_strides) const {
    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                float output_ref =
                    Compute(
                        input1_scale() * (static_cast<int32_t>(
                                              input1[i * input1_strides[0] +
                                                     j * input1_strides[1] +
                                                     k * input1_strides[2] +
                                                     l * input1_strides[3] +
                                                     m * input1_strides[4] +
                                                     n * input1_strides[5]]) -
                                          input1_zero_point()),
                        input2_scale() * (static_cast<int32_t>(
                                              input2[i * input2_strides[0] +
                                                     j * input2_strides[1] +
                                                     k * input2_strides[2] +
                                                     l * input2_strides[3] +
                                                     m * input2_strides[4] +
                                                     n * input2_strides[5]]) -
                                          input2_zero_point())) /
                        output_scale() +
                    static_cast<float>(output_zero_point());
                output_ref =
                    std::max<float>(output_ref, static_cast<float>(qmin()));
                output_ref =
                    std::min<float>(output_ref, static_cast<float>(qmax()));
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(static_cast<float>(output[index]), output_ref, 0.6f)
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
      int16_t input1_zero_point) {
    this->input1_zero_point_ = input1_zero_point;
    return *this;
  }

  int16_t input1_zero_point() const { return this->input1_zero_point_; }

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
      int16_t input2_zero_point) {
    this->input2_zero_point_ = input2_zero_point;
    return *this;
  }

  int16_t input2_zero_point() const { return this->input2_zero_point_; }

  BinaryElementwiseOperatorTester& input2_scale(float input2_scale) {
    assert(std::isfinite(input2_scale));
    this->input2_scale_ = input2_scale;
    return *this;
  }

  float input2_scale() const { return this->input2_scale_; }

  BinaryElementwiseOperatorTester& output_zero_point(
      int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  int16_t output_zero_point() const { return this->output_zero_point_; }

  BinaryElementwiseOperatorTester& output_scale(float output_scale) {
    assert(std::isfinite(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const { return this->output_scale_; }

  BinaryElementwiseOperatorTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  int16_t qmin() const { return this->qmin_; }

  BinaryElementwiseOperatorTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  int16_t qmax() const { return this->qmax_; }

  BinaryElementwiseOperatorTester& operation_type(
      OperationType operation_type) {
    this->operation_type_ = operation_type;
    return *this;
  }

  OperationType operation_type() const { return this->operation_type_; }

  BinaryElementwiseOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  float Compute(float a, float b) const {
    switch (operation_type()) {
      case OperationType::Add:
        return a + b;
      case OperationType::CopySign:
        return std::copysign(a, b);
      case OperationType::Divide:
        return a / b;
      case OperationType::Maximum:
        return std::max<float>(a, b);
      case OperationType::Minimum:
        return std::min<float>(a, b);
      case OperationType::Multiply:
        return a * b;
      case OperationType::Prelu:
        return a < 0 ? a * b : a;
      case OperationType::SquaredDifference:
        return (a - b) * (a - b);
      case OperationType::Subtract:
        return a - b;
      default:
        return std::nanf("");
    }
  }

  int32_t Compute(int32_t a, int32_t b) const{
    switch (operation_type()) {
      case OperationType::Add:
        return a + b;
      case OperationType::CopySign:
        return std::copysign(a, b);
      case OperationType::Divide:
        return a / b;
      case OperationType::Maximum:
        return std::max<int32_t>(a, b);
      case OperationType::Minimum:
        return std::min<int32_t>(a, b);
      case OperationType::Multiply:
        return a * b;
      case OperationType::SquaredDifference:
        return (a - b) * (a - b);
      case OperationType::Subtract:
        return a - b;
      default:
        return INT_MAX;

    }
  }
  void TestQS8() const;

  void TestQU8() const;

  void TestF16() const;

  void TestF32() const;

  void TestS32() const;

  void TestRunF32() const;

  void TestRunQS8() const;

  void TestRunQU8() const;

  void Test(int8_t) { TestQS8(); }
  void Test(uint8_t) { TestQU8(); }
  void Test(xnn_float16) { TestF16(); }
  void Test(float) { TestF32(); }
  void Test(int32_t) { TestS32(); }

  void TestRun(int8_t) { TestRunQS8(); }
  void TestRun(uint8_t) { TestRunQU8(); }
  void TestRun(xnn_float16) {}
  void TestRun(float) { TestRunF32(); }
  void TestRun(int32_t) {}

 private:
  std::vector<size_t> input1_shape_;
  std::vector<size_t> input2_shape_;
  int16_t input1_zero_point_{0};
  float input1_scale_{1.0f};
  int16_t input2_zero_point_{0};
  float input2_scale_{1.0f};
  int16_t output_zero_point_{0};
  float output_scale_{1.0f};
  int16_t qmin_{std::numeric_limits<int16_t>::min()};
  int16_t qmax_{std::numeric_limits<int16_t>::max()};
  OperationType operation_type_{OperationType::Unknown};
  size_t iterations_{3};
};

// Make a shape of `rank` dimensions, broadcasting in each dimension according
// `broadcast_mask`.
inline std::vector<size_t> MakeShapeOfRank(size_t rank, uint32_t broadcast_mask,
                                           const size_t* dims) {
  std::vector<size_t> shape;
  for (size_t i = 0; i < rank; i++) {
    const bool broadcast = (broadcast_mask & (uint32_t(1) << i)) != 0;
    shape.push_back(broadcast ? 1 : dims[i]);
  }
  std::reverse(shape.begin(), shape.end());
  return shape;
}

enum class RunMode {
  kCreateReshapeRun,
  kEager,
};

template <typename T>
void RunBinaryOpTester(size_t rank_a, size_t rank_b, const size_t* dims,
                       RunMode run_mode,
                       BinaryElementwiseOperatorTester& tester) {
  for (uint32_t bm1 = 0; bm1 < (uint32_t(1) << rank_a); bm1++) {
    for (uint32_t bm2 = 0; bm2 < (uint32_t(1) << rank_b); bm2++) {
      tester.input1_shape(MakeShapeOfRank(rank_a, bm1, dims))
          .input2_shape(MakeShapeOfRank(rank_b, bm2, dims));
      if (run_mode == RunMode::kCreateReshapeRun) {
        tester.Test(T());
      } else if (run_mode == RunMode::kEager) {
        tester.TestRun(T());
      } else {
        FAIL() << "Unknown run_mode";
      }
    }
  }
}