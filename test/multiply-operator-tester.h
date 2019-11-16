// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>


class MultiplyOperatorTester {
 public:
  inline MultiplyOperatorTester& input1_shape(std::initializer_list<size_t> input1_shape) {
    assert(input1_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input1_shape_ = std::vector<size_t>(input1_shape);
    return *this;
  }

  inline const std::vector<size_t>& input1_shape() const {
    return this->input1_shape_;
  }

  inline size_t input1_dim(size_t i) const {
    return i < num_input1_dims() ? this->input1_shape_[i] : 1;
  }

  inline size_t num_input1_dims() const {
    return this->input1_shape_.size();
  }

  inline size_t num_input1_elements() const {
    return std::accumulate(
      this->input1_shape_.begin(), this->input1_shape_.end(), size_t(1), std::multiplies<size_t>());
  }

  inline MultiplyOperatorTester& input2_shape(std::initializer_list<size_t> input2_shape) {
    assert(input2_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input2_shape_ = std::vector<size_t>(input2_shape);
    return *this;
  }

  inline const std::vector<size_t>& input2_shape() const {
    return this->input2_shape_;
  }

  inline size_t input2_dim(size_t i) const {
    return i < num_input2_dims() ? this->input2_shape_[i] : 1;
  }

  inline size_t num_input2_dims() const {
    return this->input2_shape_.size();
  }

  inline size_t num_input2_elements() const {
    return std::accumulate(
      this->input2_shape_.begin(), this->input2_shape_.end(), size_t(1), std::multiplies<size_t>());
  }

  inline MultiplyOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline MultiplyOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline MultiplyOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input1_dims.begin(), input1_dims.end(), 1);
    std::fill(input2_dims.begin(), input2_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input1_shape().cbegin(), input1_shape().cend(), input1_dims.end() - num_input1_dims());
    std::copy(input2_shape().cbegin(), input2_shape().cend(), input2_dims.end() - num_input2_dims());
    for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
      if (input1_dims[i] != 1 && input2_dims[i] != 1) {
        ASSERT_EQ(input1_dims[i], input2_dims[i]);
      }
      output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
    }
    const size_t num_output_elements =
      std::accumulate(output_dims.begin(), output_dims.end(), size_t(1), std::multiplies<size_t>());

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

    std::vector<float> input1(XNN_EXTRA_BYTES / sizeof(float) + num_input1_elements());
    std::vector<float> input2(XNN_EXTRA_BYTES / sizeof(float) + num_input2_elements());
    std::vector<float> output(num_output_elements);
    std::vector<float> output_ref(num_output_elements);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input1.begin(), input1.end(), std::ref(f32rng));
      std::generate(input2.begin(), input2.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              output_ref[i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3]] =
                input1[i * input1_strides[0] + j * input1_strides[1] + k * input1_strides[2] + l * input1_strides[3]] *
                input2[i * input2_strides[0] + j * input2_strides[1] + k * input2_strides[2] + l * input2_strides[3]];
            }
          }
        }
      }
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = num_output_elements == 1 ?
        -std::numeric_limits<float>::infinity() : accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = num_output_elements == 1 ?
        +std::numeric_limits<float>::infinity() : accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      for (float& output_value : output_ref) {
        output_value = std::min(std::max(output_value, output_min), output_max);
      }

      // Create, setup, run, and destroy Multiply operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize());
      xnn_operator_t multiply_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_multiply_nd_f32(
          output_min, output_max,
          0, &multiply_op));
      ASSERT_NE(nullptr, multiply_op);

      // Smart pointer to automatically delete multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_multiply_op(multiply_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_multiply_nd_f32(
          multiply_op,
          num_input1_dims(),
          input1_shape().data(),
          num_input2_dims(),
          input2_shape().data(),
          input1.data(), input2.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(multiply_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              const size_t index =
                i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3];
              ASSERT_NEAR(output[index], output_ref[index], 1.0e-6f * std::abs(output_ref[index]))
                << "(i, j, k, l) = (" << i << ", " << j << ", " << k << ", " << l << ")";
            }
          }
        }
      }
    }
  }

 private:
  std::vector<size_t> input1_shape_;
  std::vector<size_t> input2_shape_;
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{5};
};
