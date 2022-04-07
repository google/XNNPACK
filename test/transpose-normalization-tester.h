// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/normalization.h>

class TransposeNormalizationTester {
 public:
  inline TransposeNormalizationTester& num_dims(size_t num_dims) {
    assert(num_dims != 0);
    this->num_dims_ = num_dims;
    return *this;
  }

  inline size_t num_dims() const { return this->num_dims_; }

  inline TransposeNormalizationTester& element_size(size_t element_size) {
    this->element_size_ = element_size;
    return *this;
  }

  inline size_t element_size() const { return this->element_size_; }

  inline TransposeNormalizationTester& expected_normalized_dims(size_t expected_normalized_dims) {
    this->expected_normalized_dims_ = expected_normalized_dims;
    return *this;
  }

  inline size_t expected_normalized_dims() const { return this->expected_normalized_dims_; }

  inline TransposeNormalizationTester& expected_element_size(size_t expected_element_size) {
    this->expected_element_size_ = expected_element_size;
    return *this;
  }

  inline size_t expected_element_size() const { return this->expected_element_size_; }

  inline TransposeNormalizationTester& shape(std::vector<size_t> shape) {
    assert(shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->shape_ = shape;
    return *this;
  }

  inline TransposeNormalizationTester& perm(std::vector<size_t> perm) {
    assert(perm.size() <= XNN_MAX_TENSOR_DIMS);
    this->perm_ = perm;
    return *this;
  }

  inline TransposeNormalizationTester& expected_normalized_shape(std::vector<size_t> expected_normalized_shape) {
    this->expected_normalized_shape_ = expected_normalized_shape;
    return *this;
  }

  inline const std::vector<size_t>& expected_normalized_shape() const { return this->expected_normalized_shape_; }

  inline TransposeNormalizationTester& expected_normalized_perm(std::vector<size_t> expected_normalized_perm) {
    this->expected_normalized_perm_ = expected_normalized_perm;
    return *this;
  }

  inline const std::vector<size_t>& expected_normalized_perm() const { return this->expected_normalized_perm_; }

  void Test() const {
    size_t actual_element_size;
    size_t actual_normalized_dims;
    std::vector<size_t> actual_normalized_shape(num_dims());
    std::vector<size_t> actual_normalized_perm(num_dims());

    xnn_normalize_transpose_permutation(num_dims(), element_size(), perm_.data(),
                                        shape_.data(), &actual_normalized_dims, &actual_element_size,
                                        actual_normalized_perm.data(), actual_normalized_shape.data());
    EXPECT_EQ(expected_element_size(), actual_element_size);
    EXPECT_EQ(expected_normalized_dims(), actual_normalized_dims);

    for (size_t i = 0; i < expected_normalized_dims(); ++i) {
      EXPECT_EQ(expected_normalized_shape()[i], actual_normalized_shape[i]);
      EXPECT_EQ(expected_normalized_perm()[i], actual_normalized_perm[i]);
    }
  }

 private:
  size_t num_dims_;
  size_t element_size_;
  size_t expected_normalized_dims_;
  size_t expected_element_size_;
  std::vector<size_t> shape_;
  std::vector<size_t> perm_;
  std::vector<size_t> expected_normalized_shape_;
  std::vector<size_t> expected_normalized_perm_;
};
