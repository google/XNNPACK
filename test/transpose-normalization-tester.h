// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/normalization.h"

class TransposeNormalizationTester {
 public:
  TransposeNormalizationTester& num_dims(size_t num_dims) {
    assert(num_dims != 0);
    this->num_dims_ = num_dims;
    return *this;
  }

  size_t num_dims() const { return this->num_dims_; }

  TransposeNormalizationTester& element_size(size_t element_size) {
    this->element_size_ = element_size;
    return *this;
  }

  size_t element_size() const { return this->element_size_; }

  TransposeNormalizationTester& expected_dims(size_t expected_dims) {
    this->expected_dims_ = expected_dims;
    return *this;
  }

  size_t expected_dims() const { return this->expected_dims_; }

  TransposeNormalizationTester& expected_element_size(size_t expected_element_size) {
    this->expected_element_size_ = expected_element_size;
    return *this;
  }

  size_t expected_element_size() const { return this->expected_element_size_; }

  TransposeNormalizationTester& shape(const std::vector<size_t> shape) {
    assert(shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->shape_ = shape;
    return *this;
  }

  TransposeNormalizationTester& perm(const std::vector<size_t> perm) {
    assert(perm.size() <= XNN_MAX_TENSOR_DIMS);
    this->perm_ = perm;
    return *this;
  }

  TransposeNormalizationTester& input_stride(const std::vector<size_t> input_stride) {
    assert(input_stride.size() <= XNN_MAX_TENSOR_DIMS);
    this->input_stride_ = input_stride;
    return *this;
  }

  TransposeNormalizationTester& output_stride(const std::vector<size_t> output_stride) {
    assert(output_stride.size() <= XNN_MAX_TENSOR_DIMS);
    this->output_stride_ = output_stride;
    return *this;
  }

  TransposeNormalizationTester& expected_shape(const std::vector<size_t> expected_shape) {
    this->expected_shape_ = expected_shape;
    return *this;
  }

  const std::vector<size_t>& expected_shape() const { return this->expected_shape_; }

  TransposeNormalizationTester& expected_perm(const std::vector<size_t> expected_perm) {
    this->expected_perm_ = expected_perm;
    return *this;
  }

  const std::vector<size_t>& expected_perm() const { return this->expected_perm_; }

  TransposeNormalizationTester& expected_input_stride(const std::vector<size_t> expected_input_stride) {
    this->expected_input_stride_ = expected_input_stride;
    return *this;
  }

  TransposeNormalizationTester& expected_output_stride(const std::vector<size_t> expected_output_stride) {
    this->expected_output_stride_ = expected_output_stride;
    return *this;
  }

  const std::vector<size_t>& expected_input_stride() const { return this->expected_input_stride_; }

  const std::vector<size_t>& expected_output_stride() const { return this->expected_output_stride_; }

  TransposeNormalizationTester& calculate_expected_input_stride() {
    expected_input_stride_.resize(expected_dims());
    expected_input_stride_[expected_dims() - 1] = expected_element_size();
    for (size_t i = expected_dims() - 1; i-- != 0;) {
      expected_input_stride_[i] = expected_input_stride_[i + 1] * expected_shape_[i + 1];
    }
    return *this;
  }

  TransposeNormalizationTester& calculate_expected_output_stride() {
    expected_output_stride_.resize(expected_dims());
    expected_output_stride_[expected_dims() - 1] = expected_element_size();
    for (size_t i = expected_dims() - 1; i-- != 0;) {
      expected_output_stride_[i] = expected_output_stride_[i + 1]
          * expected_shape_[expected_perm_[i + 1]];
    }
    return *this;
  }

  void Test() const {
    size_t actual_element_size;
    size_t actual_normalized_dims;
    std::vector<size_t> actual_normalized_shape(num_dims());
    std::vector<size_t> actual_normalized_perm(num_dims());
    std::vector<size_t> actual_normalized_input_stride(num_dims());
    std::vector<size_t> actual_normalized_output_stride(num_dims());

    xnn_normalize_transpose_permutation(num_dims(), element_size(), perm_.data(),
                                        shape_.data(), input_stride_.empty() ? nullptr : input_stride_.data(),
                                        output_stride_.empty() ? nullptr : output_stride_.data(),
                                        &actual_normalized_dims, &actual_element_size, actual_normalized_perm.data(),
                                        actual_normalized_shape.data(), actual_normalized_input_stride.data(),
                                        actual_normalized_output_stride.data());
    EXPECT_EQ(expected_element_size(), actual_element_size);
    EXPECT_EQ(expected_dims(), actual_normalized_dims);

    for (size_t i = 0; i < expected_dims(); ++i) {
      EXPECT_EQ(expected_shape()[i], actual_normalized_shape[i]);
      EXPECT_EQ(expected_perm()[i], actual_normalized_perm[i]);
      EXPECT_EQ(expected_input_stride()[i], actual_normalized_input_stride[i]);
      EXPECT_EQ(expected_output_stride()[i], actual_normalized_output_stride[i]);
    }
  }

 private:
  size_t num_dims_;
  size_t element_size_;
  size_t expected_dims_;
  size_t expected_element_size_;
  std::vector<size_t> shape_;
  std::vector<size_t> perm_;
  std::vector<size_t> input_stride_;
  std::vector<size_t> output_stride_;
  std::vector<size_t> expected_shape_;
  std::vector<size_t> expected_perm_;
  std::vector<size_t> expected_input_stride_;
  std::vector<size_t> expected_output_stride_;
};
