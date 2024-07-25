// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "binary-elementwise-operator-tester.h"

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "replicable_random_device.h"
#include <climits>

void BinaryElementwiseOperatorTester::TestQS8() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_GE(input1_zero_point(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(input1_zero_point(), std::numeric_limits<int8_t>::max());
  ASSERT_GE(input2_zero_point(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(input2_zero_point(), std::numeric_limits<int8_t>::max());
  ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());
  ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<int8_t> input1(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                             num_input1_elements());
  std::vector<int8_t> input2(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                             num_input2_elements());
  std::vector<int8_t> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return i8dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return i8dist(rng); });
    std::fill(output.begin(), output.end(), 0xAA);

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
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
              }
            }
          }
        }
      }
    }

    for (float& output_value : output_ref) {
      output_value = std::max<float>(output_value, static_cast<float>(qmin()));
      output_value = std::min<float>(output_value, static_cast<float>(qmax()));
    }

    // Create, setup, run, and destroy a binary elementwise operator.
    ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
    xnn_operator_t binary_elementwise_op = nullptr;
    xnn_status status = xnn_status_unsupported_parameter;
    switch (operation_type()) {
      case OperationType::Add:
        status = xnn_create_add_nd_qs8(
            input1_zero_point(), input1_scale(), input2_zero_point(),
            input2_scale(), output_zero_point(), output_scale(),
            static_cast<int8_t>(qmin()), static_cast<int8_t>(qmax()), 0,
            &binary_elementwise_op);
        break;
      case OperationType::Multiply:
        status = xnn_create_multiply_nd_qs8(
            input1_zero_point(), input1_scale(), input2_zero_point(),
            input2_scale(), output_zero_point(), output_scale(),
            static_cast<int8_t>(qmin()), static_cast<int8_t>(qmax()), 0,
            &binary_elementwise_op);
        break;
      case OperationType::Subtract:
        status = xnn_create_subtract_nd_qs8(
            input1_zero_point(), input1_scale(), input2_zero_point(),
            input2_scale(), output_zero_point(), output_scale(),
            static_cast<int8_t>(qmin()), static_cast<int8_t>(qmax()), 0,
            &binary_elementwise_op);
        break;
      default:
        FAIL() << "Unsupported operation type";
    }
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
    }
    ASSERT_EQ(xnn_status_success, status);
    ASSERT_NE(nullptr, binary_elementwise_op);

    // Smart pointer to automatically delete binary_elementwise_op.
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
        auto_binary_elementwise_op(binary_elementwise_op, xnn_delete_operator);

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_add_nd_qs8(binary_elementwise_op, num_input1_dims(),
                                   input1_shape().data(), num_input2_dims(),
                                   input2_shape().data(),
                                   /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_add_nd_qs8(binary_elementwise_op, input1.data(),
                                       input2.data(), output.data()));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_multiply_nd_qs8(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_multiply_nd_qs8(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_subtract_nd_qs8(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_subtract_nd_qs8(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(binary_elementwise_op, /*threadpool=*/nullptr));

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(static_cast<float>(output[index]),
                            output_ref[index], 0.6f)
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

void BinaryElementwiseOperatorTester::TestQU8() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_GE(input1_zero_point(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(input1_zero_point(), std::numeric_limits<uint8_t>::max());
  ASSERT_GE(input2_zero_point(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(input2_zero_point(), std::numeric_limits<uint8_t>::max());
  ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());
  ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<uint8_t> input1(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                              num_input1_elements());
  std::vector<uint8_t> input2(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                              num_input2_elements());
  std::vector<uint8_t> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return u8dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return u8dist(rng); });
    std::fill(output.begin(), output.end(), 0xAA);

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
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
              }
            }
          }
        }
      }
    }

    for (float& output_value : output_ref) {
      output_value = std::max<float>(output_value, static_cast<float>(qmin()));
      output_value = std::min<float>(output_value, static_cast<float>(qmax()));
    }

    // Create, setup, run, and destroy a binary elementwise operator.
    ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
    xnn_operator_t binary_elementwise_op = nullptr;
    xnn_status status = xnn_status_unsupported_parameter;
    switch (operation_type()) {
      case OperationType::Add:
        status = xnn_create_add_nd_qu8(
            input1_zero_point(), input1_scale(), input2_zero_point(),
            input2_scale(), output_zero_point(), output_scale(),
            static_cast<uint8_t>(qmin()), static_cast<uint8_t>(qmax()), 0,
            &binary_elementwise_op);
        break;
      case OperationType::Multiply:
        status = xnn_create_multiply_nd_qu8(
            input1_zero_point(), input1_scale(), input2_zero_point(),
            input2_scale(), output_zero_point(), output_scale(),
            static_cast<uint8_t>(qmin()), static_cast<uint8_t>(qmax()), 0,
            &binary_elementwise_op);
        break;
      case OperationType::Subtract:
        status = xnn_create_subtract_nd_qu8(
            input1_zero_point(), input1_scale(), input2_zero_point(),
            input2_scale(), output_zero_point(), output_scale(),
            static_cast<uint8_t>(qmin()), static_cast<uint8_t>(qmax()), 0,
            &binary_elementwise_op);
        break;
      default:
        FAIL() << "Unsupported operation type";
    }
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
    }
    ASSERT_EQ(xnn_status_success, status);
    ASSERT_NE(nullptr, binary_elementwise_op);

    // Smart pointer to automatically delete binary_elementwise_op.
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
        auto_binary_elementwise_op(binary_elementwise_op, xnn_delete_operator);

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_add_nd_qu8(binary_elementwise_op, num_input1_dims(),
                                   input1_shape().data(), num_input2_dims(),
                                   input2_shape().data(),
                                   /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_add_nd_qu8(binary_elementwise_op, input1.data(),
                                       input2.data(), output.data()));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_multiply_nd_qu8(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_multiply_nd_qu8(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_subtract_nd_qu8(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_subtract_nd_qu8(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(binary_elementwise_op, /*threadpool=*/nullptr));

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(
                    static_cast<float>(static_cast<int32_t>(output[index])),
                    output_ref[index], 0.6f)
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

void BinaryElementwiseOperatorTester::TestF16() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<uint16_t> input1(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                               num_input1_elements());
  std::vector<uint16_t> input2(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                               num_input2_elements());
  std::vector<uint16_t> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    std::generate(input2.begin(), input2.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
                    Compute(
                        fp16_ieee_to_fp32_value(input1[i * input1_strides[0] +
                                                       j * input1_strides[1] +
                                                       k * input1_strides[2] +
                                                       l * input1_strides[3] +
                                                       m * input1_strides[4] +
                                                       n * input1_strides[5]]),
                        fp16_ieee_to_fp32_value(input2[i * input2_strides[0] +
                                                       j * input2_strides[1] +
                                                       k * input2_strides[2] +
                                                       l * input2_strides[3] +
                                                       m * input2_strides[4] +
                                                       n * input2_strides[5]]));
              }
            }
          }
        }
      }
    }

    // Compute clamping parameters.
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    float output_min =
        accumulated_min +
        accumulated_range *
            (static_cast<float>(qmin() - std::numeric_limits<int16_t>::min()) /
             static_cast<float>(std::numeric_limits<int16_t>::max() -
                                std::numeric_limits<int16_t>::min()));
    if (qmin() == std::numeric_limits<int16_t>::min()) {
      output_min = -std::numeric_limits<float>::infinity();
    }
    float output_max =
        accumulated_max -
        accumulated_range *
            (static_cast<float>(std::numeric_limits<int16_t>::max() - qmax()) /
             static_cast<float>(std::numeric_limits<int16_t>::max() -
                                std::numeric_limits<int16_t>::min()));
    if (qmax() == std::numeric_limits<int16_t>::max()) {
      output_max = +std::numeric_limits<float>::infinity();
    }
    output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min));
    output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max));

    for (float& output_value : output_ref) {
      output_value = std::max(output_value, output_min);
      output_value = std::min(output_value, output_max);
    }

    // Create, setup, run, and destroy a binary elementwise operator.
    ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
    xnn_operator_t binary_elementwise_op = nullptr;
    xnn_status status = xnn_status_unsupported_parameter;
    switch (operation_type()) {
      case OperationType::Add:
        status = xnn_create_add_nd_f16(output_min, output_max, 0,
                                       &binary_elementwise_op);
        break;
      case OperationType::Divide:
        status = xnn_create_divide_nd_f16(output_min, output_max, 0,
                                          &binary_elementwise_op);
        break;
      case OperationType::Maximum:
        status = xnn_create_maximum_nd_f16(0, &binary_elementwise_op);
        break;
      case OperationType::Minimum:
        status = xnn_create_minimum_nd_f16(0, &binary_elementwise_op);
        break;
      case OperationType::Multiply:
        status = xnn_create_multiply_nd_f16(output_min, output_max, 0,
                                            &binary_elementwise_op);
        break;
      case OperationType::SquaredDifference:
        status =
            xnn_create_squared_difference_nd_f16(0, &binary_elementwise_op);
        break;
      case OperationType::Subtract:
        status = xnn_create_subtract_nd_f16(output_min, output_max, 0,
                                            &binary_elementwise_op);
        break;
      default:
        FAIL() << "Unsupported operation type";
    }
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
    }
    ASSERT_EQ(xnn_status_success, status);
    ASSERT_NE(nullptr, binary_elementwise_op);

    // Smart pointer to automatically delete binary_elementwise_op.
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
        auto_binary_elementwise_op(binary_elementwise_op, xnn_delete_operator);

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_add_nd_f16(binary_elementwise_op, num_input1_dims(),
                                   input1_shape().data(), num_input2_dims(),
                                   input2_shape().data(),
                                   /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_add_nd_f16(binary_elementwise_op, input1.data(),
                                       input2.data(), output.data()));
        break;
      case OperationType::Divide:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_divide_nd_f16(binary_elementwise_op, num_input1_dims(),
                                      input1_shape().data(), num_input2_dims(),
                                      input2_shape().data(),
                                      /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_divide_nd_f16(binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Maximum:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_maximum_nd_f16(binary_elementwise_op, num_input1_dims(),
                                       input1_shape().data(), num_input2_dims(),
                                       input2_shape().data(),
                                       /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_maximum_nd_f16(binary_elementwise_op, input1.data(),
                                           input2.data(), output.data()));
        break;
      case OperationType::Minimum:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_minimum_nd_f16(binary_elementwise_op, num_input1_dims(),
                                       input1_shape().data(), num_input2_dims(),
                                       input2_shape().data(),
                                       /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_minimum_nd_f16(binary_elementwise_op, input1.data(),
                                           input2.data(), output.data()));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_multiply_nd_f16(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_multiply_nd_f16(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::SquaredDifference:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_squared_difference_nd_f16(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_squared_difference_nd_f16(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_subtract_nd_f16(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_subtract_nd_f16(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(binary_elementwise_op, /*threadpool=*/nullptr));

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(
                    fp16_ieee_to_fp32_value(output[index]), output_ref[index],
                    std::max(1.0e-4f, std::abs(output_ref[index]) * 1.0e-2f))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k
                    << ", " << l << ", " << m << ", " << n << ")";
              }
            }
          }
        }
      }
    }
  }
}

void BinaryElementwiseOperatorTester::TestS32() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> s32dist(-10000, 10000);
  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<int32_t> input1(XNN_EXTRA_BYTES / sizeof(int32_t) +
                            num_input1_elements());
  std::vector<int32_t> input2(XNN_EXTRA_BYTES / sizeof(int32_t) +
                            num_input2_elements());
  std::vector<int32_t> output(num_output_elements);
  std::vector<int32_t> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return s32dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return s32dist(rng); });
    std::fill(output.begin(), output.end(), INT_MAX);

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
                    Compute(
                        input1[i * input1_strides[0] + j * input1_strides[1] +
                               k * input1_strides[2] + l * input1_strides[3] +
                               m * input1_strides[4] + n * input1_strides[5]],
                        input2[i * input2_strides[0] + j * input2_strides[1] +
                               k * input2_strides[2] + l * input2_strides[3] +
                               m * input2_strides[4] + n * input2_strides[5]]);
              }
            }
          }
        }
      }
    }

    // Create, setup, run, and destroy a binary elementwise operator.
    ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
    xnn_operator_t binary_elementwise_op = nullptr;

    switch (operation_type()) {
      case OperationType::Multiply:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_multiply_nd_s32(0, &binary_elementwise_op));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }
    ASSERT_NE(nullptr, binary_elementwise_op);

    // Smart pointer to automatically delete binary_elementwise_op.
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
        auto_binary_elementwise_op(binary_elementwise_op, xnn_delete_operator);

    switch (operation_type()) {
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_multiply_nd_s32(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_multiply_nd_s32(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(binary_elementwise_op, /*threadpool=*/nullptr));
    // Verify results.

    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_EQ(output[index], output_ref[index]);
              }
            }
          }
        }
      }
    }
  }
}


void BinaryElementwiseOperatorTester::TestF32() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<float> input1(XNN_EXTRA_BYTES / sizeof(float) +
                            num_input1_elements());
  std::vector<float> input2(XNN_EXTRA_BYTES / sizeof(float) +
                            num_input2_elements());
  std::vector<float> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
    std::fill(output.begin(), output.end(), nanf(""));

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
                    Compute(
                        input1[i * input1_strides[0] + j * input1_strides[1] +
                               k * input1_strides[2] + l * input1_strides[3] +
                               m * input1_strides[4] + n * input1_strides[5]],
                        input2[i * input2_strides[0] + j * input2_strides[1] +
                               k * input2_strides[2] + l * input2_strides[3] +
                               m * input2_strides[4] + n * input2_strides[5]]);
              }
            }
          }
        }
      }
    }
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    float output_min =
        accumulated_min +
        accumulated_range *
            (static_cast<float>(qmin() - std::numeric_limits<int16_t>::min()) /
             static_cast<float>(std::numeric_limits<int16_t>::max() -
                                std::numeric_limits<int16_t>::min()));
    if (qmin() == std::numeric_limits<int16_t>::min()) {
      output_min = -std::numeric_limits<float>::infinity();
    }
    float output_max =
        accumulated_max -
        accumulated_range *
            (static_cast<float>(std::numeric_limits<int16_t>::max() - qmax()) /
             static_cast<float>(std::numeric_limits<int16_t>::max() -
                                std::numeric_limits<int16_t>::min()));
    if (qmax() == std::numeric_limits<int16_t>::max()) {
      output_max = +std::numeric_limits<float>::infinity();
    }
    for (float& output_value : output_ref) {
      output_value = std::max(output_value, output_min);
      output_value = std::min(output_value, output_max);
    }

    // Create, setup, run, and destroy a binary elementwise operator.
    ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
    xnn_operator_t binary_elementwise_op = nullptr;

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_add_nd_f32(output_min, output_max, 0,
                                        &binary_elementwise_op));
        break;
      case OperationType::CopySign:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_copysign_nd_f32(/*flags=*/0, &binary_elementwise_op));
        break;
      case OperationType::Divide:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_divide_nd_f32(output_min, output_max, 0,
                                           &binary_elementwise_op));
        break;
      case OperationType::Maximum:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_maximum_nd_f32(0, &binary_elementwise_op));
        break;
      case OperationType::Minimum:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_minimum_nd_f32(0, &binary_elementwise_op));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_multiply_nd_f32(output_min, output_max, 0,
                                             &binary_elementwise_op));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(xnn_status_success,
                  xnn_create_subtract_nd_f32(output_min, output_max, 0,
                                             &binary_elementwise_op));
        break;
      case OperationType::SquaredDifference:
        ASSERT_EQ(xnn_status_success, xnn_create_squared_difference_nd_f32(
                                          0, &binary_elementwise_op));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }
    ASSERT_NE(nullptr, binary_elementwise_op);

    // Smart pointer to automatically delete binary_elementwise_op.
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
        auto_binary_elementwise_op(binary_elementwise_op, xnn_delete_operator);

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_add_nd_f32(binary_elementwise_op, num_input1_dims(),
                                   input1_shape().data(), num_input2_dims(),
                                   input2_shape().data(),
                                   /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_add_nd_f32(binary_elementwise_op, input1.data(),
                                       input2.data(), output.data()));
        break;
      case OperationType::CopySign:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_copysign_nd_f32(binary_elementwise_op, num_input1_dims(),
                                      input1_shape().data(), num_input2_dims(),
                                      input2_shape().data(),
                                      /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_copysign_nd_f32(binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Divide:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_divide_nd_f32(binary_elementwise_op, num_input1_dims(),
                                      input1_shape().data(), num_input2_dims(),
                                      input2_shape().data(),
                                      /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_divide_nd_f32(binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Maximum:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_maximum_nd_f32(binary_elementwise_op, num_input1_dims(),
                                       input1_shape().data(), num_input2_dims(),
                                       input2_shape().data(),
                                       /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_maximum_nd_f32(binary_elementwise_op, input1.data(),
                                           input2.data(), output.data()));
        break;
      case OperationType::Minimum:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_minimum_nd_f32(binary_elementwise_op, num_input1_dims(),
                                       input1_shape().data(), num_input2_dims(),
                                       input2_shape().data(),
                                       /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_minimum_nd_f32(binary_elementwise_op, input1.data(),
                                           input2.data(), output.data()));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_multiply_nd_f32(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_multiply_nd_f32(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_subtract_nd_f32(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_subtract_nd_f32(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      case OperationType::SquaredDifference:
        ASSERT_EQ(
            xnn_status_success,
            xnn_reshape_squared_difference_nd_f32(
                binary_elementwise_op, num_input1_dims(), input1_shape().data(),
                num_input2_dims(), input2_shape().data(),
                /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_squared_difference_nd_f32(
                                          binary_elementwise_op, input1.data(),
                                          input2.data(), output.data()));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(binary_elementwise_op, /*threadpool=*/nullptr));

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(output[index], output_ref[index],
                            1.0e-6f * std::abs(output_ref[index]))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k
                    << ", " << l << ", " << m << ", " << n << ")";
              }
            }
          }
        }
      }
    }
  }
}

void BinaryElementwiseOperatorTester::TestRunF32() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<float> input1(XNN_EXTRA_BYTES / sizeof(float) +
                            num_input1_elements());
  std::vector<float> input2(XNN_EXTRA_BYTES / sizeof(float) +
                            num_input2_elements());
  std::vector<float> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
    std::fill(output.begin(), output.end(), nanf(""));

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
                    Compute(
                        input1[i * input1_strides[0] + j * input1_strides[1] +
                               k * input1_strides[2] + l * input1_strides[3] +
                               m * input1_strides[4] + n * input1_strides[5]],
                        input2[i * input2_strides[0] + j * input2_strides[1] +
                               k * input2_strides[2] + l * input2_strides[3] +
                               m * input2_strides[4] + n * input2_strides[5]]);
              }
            }
          }
        }
      }
    }
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    float output_min =
        accumulated_min +
        accumulated_range *
            (static_cast<float>(qmin() - std::numeric_limits<int16_t>::min()) /
             static_cast<float>(std::numeric_limits<int16_t>::max() -
                                std::numeric_limits<int16_t>::min()));
    if (qmin() == std::numeric_limits<int16_t>::min()) {
      output_min = -std::numeric_limits<float>::infinity();
    }
    float output_max =
        accumulated_max -
        accumulated_range *
            (static_cast<float>(std::numeric_limits<int16_t>::max() - qmax()) /
             static_cast<float>(std::numeric_limits<int16_t>::max() -
                                std::numeric_limits<int16_t>::min()));
    if (qmax() == std::numeric_limits<int16_t>::max()) {
      output_max = +std::numeric_limits<float>::infinity();
    }
    for (float& output_value : output_ref) {
      output_value = std::max(output_value, output_min);
      output_value = std::min(output_value, output_max);
    }

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_add_nd_f32(num_input1_dims(), input1_shape().data(),
                                     num_input2_dims(), input2_shape().data(),
                                     input1.data(), input2.data(),
                                     output.data(), output_min, output_max, 0,
                                     /*threadpool=*/nullptr));
        break;
      case OperationType::Divide:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_divide_nd_f32(
                      num_input1_dims(), input1_shape().data(),
                      num_input2_dims(), input2_shape().data(), input1.data(),
                      input2.data(), output.data(), output_min, output_max, 0,
                      /*threadpool=*/nullptr));
        break;
      case OperationType::Maximum:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_maximum_nd_f32(
                      num_input1_dims(), input1_shape().data(),
                      num_input2_dims(), input2_shape().data(), input1.data(),
                      input2.data(), output.data(), 0,
                      /*threadpool=*/nullptr));
        break;
      case OperationType::Minimum:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_minimum_nd_f32(
                      num_input1_dims(), input1_shape().data(),
                      num_input2_dims(), input2_shape().data(), input1.data(),
                      input2.data(), output.data(), 0,
                      /*threadpool=*/nullptr));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_multiply_nd_f32(
                      num_input1_dims(), input1_shape().data(),
                      num_input2_dims(), input2_shape().data(), input1.data(),
                      input2.data(), output.data(), output_min, output_max, 0,
                      /*threadpool=*/nullptr));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_subtract_nd_f32(
                      num_input1_dims(), input1_shape().data(),
                      num_input2_dims(), input2_shape().data(), input1.data(),
                      input2.data(), output.data(), output_min, output_max, 0,
                      /*threadpool=*/nullptr));
        break;
      case OperationType::SquaredDifference:
        ASSERT_EQ(xnn_status_success,
                  xnn_run_squared_difference_nd_f32(
                      num_input1_dims(), input1_shape().data(),
                      num_input2_dims(), input2_shape().data(), input1.data(),
                      input2.data(), output.data(), 0,
                      /*threadpool=*/nullptr));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(output[index], output_ref[index],
                            1.0e-6f * std::abs(output_ref[index]))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k
                    << ", " << l << ", " << m << ", " << n << ")";
              }
            }
          }
        }
      }
    }
  }
}

void BinaryElementwiseOperatorTester::TestRunQS8() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_GE(input1_zero_point(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(input1_zero_point(), std::numeric_limits<int8_t>::max());
  ASSERT_GE(input2_zero_point(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(input2_zero_point(), std::numeric_limits<int8_t>::max());
  ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());
  ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
  ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<int8_t> input1(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                             num_input1_elements());
  std::vector<int8_t> input2(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                             num_input2_elements());
  std::vector<int8_t> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return i8dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return i8dist(rng); });
    std::fill(output.begin(), output.end(), 0xAA);

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
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
              }
            }
          }
        }
      }
    }

    for (float& output_value : output_ref) {
      output_value = std::max<float>(output_value, static_cast<float>(qmin()));
      output_value = std::min<float>(output_value, static_cast<float>(qmax()));
    }

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(
            xnn_status_success,
            xnn_run_add_nd_qs8(
                num_input1_dims(), input1_shape().data(), input1_zero_point(),
                input1_scale(), num_input2_dims(), input2_shape().data(),
                input2_zero_point(), input2_scale(), input1.data(),
                input2.data(), output.data(), output_zero_point(),
                output_scale(), static_cast<int8_t>(qmin()),
                static_cast<int8_t>(qmax()), 0,
                /*threadpool=*/nullptr));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_run_multiply_nd_qs8(
                num_input1_dims(), input1_shape().data(), input1_zero_point(),
                input1_scale(), num_input2_dims(), input2_shape().data(),
                input2_zero_point(), input2_scale(), input1.data(),
                input2.data(), output.data(), output_zero_point(),
                output_scale(), static_cast<int8_t>(qmin()),
                static_cast<int8_t>(qmax()), 0,
                /*threadpool=*/nullptr));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(
            xnn_status_success,
            xnn_run_subtract_nd_qs8(
                num_input1_dims(), input1_shape().data(), input1_zero_point(),
                input1_scale(), num_input2_dims(), input2_shape().data(),
                input2_zero_point(), input2_scale(), input1.data(),
                input2.data(), output.data(), output_zero_point(),
                output_scale(), static_cast<int8_t>(qmin()),
                static_cast<int8_t>(qmax()), 0,
                /*threadpool=*/nullptr));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(static_cast<float>(output[index]),
                            output_ref[index], 0.6f)
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

void BinaryElementwiseOperatorTester::TestRunQU8() const {
  ASSERT_NE(operation_type(), OperationType::Unknown);
  ASSERT_GE(input1_zero_point(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(input1_zero_point(), std::numeric_limits<uint8_t>::max());
  ASSERT_GE(input2_zero_point(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(input2_zero_point(), std::numeric_limits<uint8_t>::max());
  ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());
  ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
  ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
  ASSERT_LT(qmin(), qmax());

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

  // Compute generalized shapes.
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::fill(input2_dims.begin(), input2_dims.end(), 1);
  std::fill(output_dims.begin(), output_dims.end(), 1);
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

  std::vector<uint8_t> input1(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                              num_input1_elements());
  std::vector<uint8_t> input2(XNN_EXTRA_BYTES / sizeof(uint16_t) +
                              num_input2_elements());
  std::vector<uint8_t> output(num_output_elements);
  std::vector<float> output_ref(num_output_elements);
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input1.begin(), input1.end(), [&]() { return u8dist(rng); });
    std::generate(input2.begin(), input2.end(), [&]() { return u8dist(rng); });
    std::fill(output.begin(), output.end(), 0xAA);

    // Compute reference results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                output_ref[i * output_strides[0] + j * output_strides[1] +
                           k * output_strides[2] + l * output_strides[3] +
                           m * output_strides[4] + n * output_strides[5]] =
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
              }
            }
          }
        }
      }
    }

    for (float& output_value : output_ref) {
      output_value = std::max<float>(output_value, static_cast<float>(qmin()));
      output_value = std::min<float>(output_value, static_cast<float>(qmax()));
    }

    switch (operation_type()) {
      case OperationType::Add:
        ASSERT_EQ(
            xnn_status_success,
            xnn_run_add_nd_qu8(
                num_input1_dims(), input1_shape().data(), input1_zero_point(),
                input1_scale(), num_input2_dims(), input2_shape().data(),
                input2_zero_point(), input2_scale(), input1.data(),
                input2.data(), output.data(), output_zero_point(),
                output_scale(), static_cast<uint8_t>(qmin()),
                static_cast<uint8_t>(qmax()), 0,
                /*threadpool=*/nullptr));
        break;
      case OperationType::Multiply:
        ASSERT_EQ(
            xnn_status_success,
            xnn_run_multiply_nd_qu8(
                num_input1_dims(), input1_shape().data(), input1_zero_point(),
                input1_scale(), num_input2_dims(), input2_shape().data(),
                input2_zero_point(), input2_scale(), input1.data(),
                input2.data(), output.data(), output_zero_point(),
                output_scale(), static_cast<uint8_t>(qmin()),
                static_cast<uint8_t>(qmax()), 0,
                /*threadpool=*/nullptr));
        break;
      case OperationType::Subtract:
        ASSERT_EQ(
            xnn_status_success,
            xnn_run_subtract_nd_qu8(
                num_input1_dims(), input1_shape().data(), input1_zero_point(),
                input1_scale(), num_input2_dims(), input2_shape().data(),
                input2_zero_point(), input2_scale(), input1.data(),
                input2.data(), output.data(), output_zero_point(),
                output_scale(), static_cast<uint8_t>(qmin()),
                static_cast<uint8_t>(qmax()), 0,
                /*threadpool=*/nullptr));
        break;
      default:
        FAIL() << "Unsupported operation type";
    }

    // Verify results.
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t index =
                    i * output_strides[0] + j * output_strides[1] +
                    k * output_strides[2] + l * output_strides[3] +
                    m * output_strides[4] + n * output_strides[5];
                ASSERT_NEAR(
                    static_cast<float>(static_cast<int32_t>(output[index])),
                    output_ref[index], 0.6f)
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
