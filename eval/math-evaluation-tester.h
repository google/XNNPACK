// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cmath>

#include <xnnpack/math-stubs.h>


class MathEvaluationTester {
 public:
  inline MathEvaluationTester& input_value(float value) {
    this->input_min_ = value;
    this->input_max_ = value;
    return *this;
  }

  inline MathEvaluationTester& input_range(float lower_bound, float upper_bound) {
    this->input_min_ = lower_bound;
    this->input_max_ = upper_bound;
    return *this;
  }

  inline float input_min() const {
    return this->input_min_;
  }

  inline float input_max() const {
    return this->input_max_;
  }

  void TestOutputMatchReference(xnn_f16_unary_math_fn math_fn, float output_value) const;
  void TestOutputMatchReference(xnn_f32_unary_math_fn math_fn, float output_value) const;

  void TestOutputMatchZero(xnn_f16_unary_math_fn math_fn) const;
  void TestOutputMatchZero(xnn_f32_unary_math_fn math_fn) const;

  void TestNaN(xnn_f16_unary_math_fn math_fn) const;
  void TestNaN(xnn_f32_unary_math_fn math_fn) const;

 private:
  static constexpr int kBlockSize = 1024;

  float input_min_ = std::nanf("");
  float input_max_ = std::nanf("");
};
