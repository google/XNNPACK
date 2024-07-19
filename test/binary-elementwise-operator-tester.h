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
#include <initializer_list>
#include <limits>
#include <numeric>
#include <vector>

#include "xnnpack.h"

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
    Subtract,
    SquaredDifference,
  };

  BinaryElementwiseOperatorTester& input1_shape(
      std::initializer_list<size_t> input1_shape) {
    assert(input1_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input1_shape_ = std::vector<size_t>(input1_shape);
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
      std::initializer_list<size_t> input2_shape) {
    assert(input2_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input2_shape_ = std::vector<size_t>(input2_shape);
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
      case OperationType::Subtract:
        return a - b;
      case OperationType::SquaredDifference:
        return (a - b) * (a - b);
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
      case OperationType::Subtract:
        return a - b;
      case OperationType::SquaredDifference:
        return (a - b) * (a - b);
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
