// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/normalization.h>

#include <gtest/gtest.h>

class SliceNormalizationTester {
 public:
  SliceNormalizationTester()
      : expected_offsets_(XNN_MAX_TENSOR_DIMS, 0),
        expected_input_shape_(XNN_MAX_TENSOR_DIMS, 1),
        expected_output_shape_(XNN_MAX_TENSOR_DIMS, 1) {}

  inline SliceNormalizationTester& input_shape(
      const std::vector<size_t> input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input_shape_ = input_shape;
    return *this;
  }

  inline std::vector<size_t> input_shape() { return input_shape_; }

  inline size_t num_dims() const { return input_shape_.size(); }

  inline SliceNormalizationTester& offsets(const std::vector<size_t> offsets) {
    assert(offsets.size() <= XNN_MAX_TENSOR_DIMS);
    this->offsets_ = offsets;
    return *this;
  }

  inline std::vector<size_t> offsets() { return offsets_; }

  inline SliceNormalizationTester& sizes(const std::vector<size_t> sizes) {
    assert(sizes.size() <= XNN_MAX_TENSOR_DIMS);
    this->sizes_ = sizes;
    return *this;
  }

  inline std::vector<size_t> sizes() { return sizes_; }

  inline SliceNormalizationTester& expected_offsets(
      const std::vector<size_t> expected_offsets) {
    assert(expected_offsets.size() <= XNN_MAX_TENSOR_DIMS);
    std::copy(expected_offsets.begin(), expected_offsets.end(),
              this->expected_offsets_.end() - expected_offsets.size());
    return *this;
  }

  inline std::vector<size_t> expected_offsets() { return expected_offsets_; }

  inline SliceNormalizationTester& expected_input_shape(
      const std::vector<size_t> expected_input_shape) {
    assert(expected_input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    std::copy(expected_input_shape.begin(), expected_input_shape.end(),
              this->expected_input_shape_.end() - expected_input_shape.size());
    return *this;
  }

  inline std::vector<size_t> expected_input_shape() {
    return expected_input_shape_;
  }

  inline SliceNormalizationTester& expected_output_shape(
      const std::vector<size_t> expected_output_shape) {
    assert(expected_output_shape.size() <= XNN_MAX_TENSOR_DIMS);
    std::copy(
        expected_output_shape.begin(), expected_output_shape.end(),
        this->expected_output_shape_.end() - expected_output_shape.size());
    expected_num_normalized_dims_ = expected_output_shape.size();
    return *this;
  }

  inline std::vector<size_t> expected_output_shape() {
    return expected_output_shape_;
  }

  void Test() {
    std::vector<size_t> actual_normalized_offsets(XNN_MAX_TENSOR_DIMS);
    std::vector<size_t> actual_normalized_input_shape(XNN_MAX_TENSOR_DIMS);
    std::vector<size_t> actual_normalized_output_shape(XNN_MAX_TENSOR_DIMS);
    size_t actual_num_normalized_dims;
    xnn_normalize_slice(num_dims(), offsets().data(), sizes().data(),
                        input_shape().data(), actual_normalized_offsets.data(),
                        actual_normalized_input_shape.data(),
                        actual_normalized_output_shape.data(),
                        &actual_num_normalized_dims);

    EXPECT_EQ(expected_num_normalized_dims_, actual_num_normalized_dims);
    EXPECT_EQ(expected_offsets(), actual_normalized_offsets);
    EXPECT_EQ(expected_input_shape(), actual_normalized_input_shape);
    EXPECT_EQ(expected_output_shape(), actual_normalized_output_shape);
  }

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> offsets_;
  std::vector<size_t> sizes_;
  std::vector<size_t> expected_offsets_{XNN_MAX_TENSOR_DIMS, 0};
  std::vector<size_t> expected_input_shape_;
  std::vector<size_t> expected_output_shape_;
  size_t expected_num_normalized_dims_;
};
