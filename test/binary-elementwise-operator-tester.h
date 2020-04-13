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
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>


class BinaryElementwiseOperatorTester {
 public:
  enum class OperationType {
    Unknown,
    Add,
    Divide,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
  };

  inline BinaryElementwiseOperatorTester& input1_shape(std::initializer_list<size_t> input1_shape) {
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

  inline BinaryElementwiseOperatorTester& input2_shape(std::initializer_list<size_t> input2_shape) {
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

  inline BinaryElementwiseOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline BinaryElementwiseOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline BinaryElementwiseOperatorTester& operation_type(OperationType operation_type) {
    this->operation_type_ = operation_type;
    return *this;
  }

  inline OperationType operation_type() const {
    return this->operation_type_;
  }

  inline BinaryElementwiseOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  float Compute(float a, float b) const {
    switch (operation_type()) {
      case OperationType::Add:
        return a + b;
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
      default:
        return std::nanf("");
    }
  }

  void TestF32() const {
    ASSERT_NE(operation_type(), OperationType::Unknown);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), rng);

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
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  output_ref[i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5]] = Compute(
                    input1[i * input1_strides[0] + j * input1_strides[1] + k * input1_strides[2] + l * input1_strides[3] + m * input1_strides[4] + n * input1_strides[5]],
                    input2[i * input2_strides[0] + j * input2_strides[1] + k * input2_strides[2] + l * input2_strides[3] + m * input2_strides[4] + n * input2_strides[5]]);
                }
              }
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

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t binary_elementwise_op = nullptr;

      switch (operation_type()) {
        case OperationType::Add:
          ASSERT_EQ(xnn_status_success,
            xnn_create_add_nd_f32(
              output_min, output_max,
              0, &binary_elementwise_op));
          break;
        case OperationType::Divide:
          ASSERT_EQ(xnn_status_success,
            xnn_create_divide_nd_f32(
              output_min, output_max,
              0, &binary_elementwise_op));
          break;
        case OperationType::Maximum:
          ASSERT_EQ(xnn_status_success,
            xnn_create_maximum_nd_f32(
              0, &binary_elementwise_op));
          break;
        case OperationType::Minimum:
          ASSERT_EQ(xnn_status_success,
            xnn_create_minimum_nd_f32(
              0, &binary_elementwise_op));
          break;
        case OperationType::Multiply:
          ASSERT_EQ(xnn_status_success,
            xnn_create_multiply_nd_f32(
              output_min, output_max,
              0, &binary_elementwise_op));
          break;
        case OperationType::Subtract:
          ASSERT_EQ(xnn_status_success,
            xnn_create_subtract_nd_f32(
              output_min, output_max,
              0, &binary_elementwise_op));
          break;
        default:
          FAIL() << "Unsupported operation type";
      }
      ASSERT_NE(nullptr, binary_elementwise_op);

      // Smart pointer to automatically delete binary_elementwise_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_binary_elementwise_op(binary_elementwise_op, xnn_delete_operator);

      switch (operation_type()) {
        case OperationType::Add:
          ASSERT_EQ(xnn_status_success,
            xnn_setup_add_nd_f32(
              binary_elementwise_op,
              num_input1_dims(),
              input1_shape().data(),
              num_input2_dims(),
              input2_shape().data(),
              input1.data(), input2.data(), output.data(),
              nullptr /* thread pool */));
          break;
        case OperationType::Divide:
          ASSERT_EQ(xnn_status_success,
            xnn_setup_divide_nd_f32(
              binary_elementwise_op,
              num_input1_dims(),
              input1_shape().data(),
              num_input2_dims(),
              input2_shape().data(),
              input1.data(), input2.data(), output.data(),
              nullptr /* thread pool */));
          break;
        case OperationType::Maximum:
          ASSERT_EQ(xnn_status_success,
            xnn_setup_maximum_nd_f32(
              binary_elementwise_op,
              num_input1_dims(),
              input1_shape().data(),
              num_input2_dims(),
              input2_shape().data(),
              input1.data(), input2.data(), output.data(),
              nullptr /* thread pool */));
          break;
        case OperationType::Minimum:
          ASSERT_EQ(xnn_status_success,
            xnn_setup_minimum_nd_f32(
              binary_elementwise_op,
              num_input1_dims(),
              input1_shape().data(),
              num_input2_dims(),
              input2_shape().data(),
              input1.data(), input2.data(), output.data(),
              nullptr /* thread pool */));
          break;
        case OperationType::Multiply:
          ASSERT_EQ(xnn_status_success,
            xnn_setup_multiply_nd_f32(
              binary_elementwise_op,
              num_input1_dims(),
              input1_shape().data(),
              num_input2_dims(),
              input2_shape().data(),
              input1.data(), input2.data(), output.data(),
              nullptr /* thread pool */));
          break;
        case OperationType::Subtract:
          ASSERT_EQ(xnn_status_success,
            xnn_setup_subtract_nd_f32(
              binary_elementwise_op,
              num_input1_dims(),
              input1_shape().data(),
              num_input2_dims(),
              input2_shape().data(),
              input1.data(), input2.data(), output.data(),
              nullptr /* thread pool */));
          break;
        default:
          FAIL() << "Unsupported operation type";
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(binary_elementwise_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  ASSERT_NEAR(output[index], output_ref[index], 1.0e-6f * std::abs(output_ref[index]))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")";
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
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  OperationType operation_type_{OperationType::Unknown};
  size_t iterations_{3};
};
