// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/normalization.h>


class ReduceNormalizationTester {
 public:
  inline ReduceNormalizationTester& axes(const std::vector<size_t>& axes) {
    assert(axes.size() <= XNN_MAX_TENSOR_DIMS);
    this->axes_ = axes;
    return *this;
  }

  inline const std::vector<size_t>& axes() const {
    return this->axes_;
  }

  inline ReduceNormalizationTester& shape(const std::vector<size_t>& shape) {
    assert(shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->shape_ = shape;
    return *this;
  }

  inline const std::vector<size_t>& shape() const {
    return this->shape_;
  }

  inline ReduceNormalizationTester& expected_axes(const std::vector<size_t>& expected_axes) {
    assert(expected_axes.size() <= XNN_MAX_TENSOR_DIMS);
    this->expected_axes_ = expected_axes;
    return *this;
  }

  inline const std::vector<size_t>& expected_axes() const {
    return this->expected_axes_;
  }

  inline ReduceNormalizationTester& expected_shape(const std::vector<size_t>& expected_shape) {
    assert(expected_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->expected_shape_ = expected_shape;
    return *this;
  }

  inline const std::vector<size_t>& expected_shape() const {
    return this->expected_shape_;
  }

  void Test() const {
    std::vector<size_t> input_dims{shape()};
    size_t num_input_dims = input_dims.size();
    std::vector<size_t> reduction_axes{axes()};
    size_t num_reduction_axes = reduction_axes.size();

    xnn_normalize_reduction(
      &num_reduction_axes,
      reduction_axes.data(),
      &num_input_dims,
      input_dims.data());

    ASSERT_EQ(num_reduction_axes, expected_axes().size());
    for (size_t i = 0; i < num_reduction_axes; i++) {
      ASSERT_EQ(expected_axes()[i], reduction_axes[i]) << " at index " << i;
    }

    ASSERT_EQ(num_input_dims, expected_shape().size());
    for (size_t i = 0; i < num_input_dims; i++) {
      ASSERT_EQ(expected_shape()[i], input_dims[i]) << " at index " << i;
    }
  }

 private:
  std::vector<size_t> axes_;
  std::vector<size_t> shape_;
  std::vector<size_t> expected_axes_;
  std::vector<size_t> expected_shape_;
};
