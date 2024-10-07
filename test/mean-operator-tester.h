// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/requantization.h"
#include "replicable_random_device.h"
#include "pthreadpool.h"

class MeanOperatorTester {
 public:
  MeanOperatorTester& input_shape(std::initializer_list<size_t> input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input_shape_ = std::vector<size_t>(input_shape);
    return *this;
  }

  MeanOperatorTester& input_shape(const std::vector<size_t>& input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    this->input_shape_ = std::vector<size_t>(input_shape);
    return *this;
  }

  const std::vector<size_t>& input_shape() const {
    return this->input_shape_;
  }

  size_t num_input_dims() const {
    return this->input_shape_.size();
  }

  size_t num_input_elements() const {
    return std::accumulate(
      this->input_shape_.begin(), this->input_shape_.end(), size_t(1), std::multiplies<size_t>());
  }

  MeanOperatorTester& reduction_axes(std::initializer_list<size_t> reduction_axes) {
    assert(reduction_axes.size() <= XNN_MAX_TENSOR_DIMS);
    this->reduction_axes_ = std::vector<size_t>(reduction_axes);
    return *this;
  }

  MeanOperatorTester& reduction_axes(const std::vector<size_t> reduction_axes) {
    assert(reduction_axes.size() <= XNN_MAX_TENSOR_DIMS);
    this->reduction_axes_ = reduction_axes;
    return *this;
  }

  const std::vector<size_t>& reduction_axes() const {
    return this->reduction_axes_;
  }

  size_t num_reduction_axes() const {
    return this->reduction_axes_.size();
  }

  MeanOperatorTester& multithreaded(size_t multithreaded) {
    this->multithreaded_ = multithreaded;
    return *this;
  }

  size_t multithreaded() const {
    return this->multithreaded_;
  }

  size_t num_threads() const {
    // Do not spin up excessive number of threads for tests.
    return multithreaded() ? 5 : 1;
  }

  MeanOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input_shape().cbegin(), input_shape().cend(), input_dims.end() - num_input_dims());
    std::copy(input_dims.cbegin(), input_dims.cend(), output_dims.begin());
    for (size_t axis : reduction_axes()) {
      (output_dims.end() - num_input_dims())[axis] = 1;
    }
    const size_t num_output_elements =
      std::accumulate(output_dims.begin(), output_dims.end(), size_t(1), std::multiplies<size_t>());

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_dims[i - 1] == 1 ? 0 : output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) + num_input_elements());
    std::vector<xnn_float16> output(num_output_elements);
    std::vector<float> output_ref(num_output_elements);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  output_ref[i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5]] +=
                    input[i * input_strides[0] + j * input_strides[1] + k * input_strides[2] + l * input_strides[3] + m * input_strides[4] + n * input_strides[5]];
                }
              }
            }
          }
        }
      }
      const float scale = float(double(num_input_elements() / num_output_elements));
      for (float& value : output_ref) {
        value /= scale;
      }

      // Create, setup, run, and destroy a mean operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t mean_op = nullptr;

      const xnn_status status = xnn_create_mean_nd_f16(
          /*flags=*/0, &mean_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, mean_op);

      // Smart pointer to automatically delete mean_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_mean_op(mean_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_mean_nd_f16(
          mean_op,
          num_reduction_axes(),
          reduction_axes().data(),
          num_input_dims(),
          input_shape().data(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_mean_nd_f16(
          mean_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(mean_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  ASSERT_NEAR(output[index], output_ref[index], 3.0e-2f * std::abs(output_ref[index]))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")";
                }
              }
            }
          }
        }
      }
    }
  }

  void TestF32() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input_shape().cbegin(), input_shape().cend(), input_dims.end() - num_input_dims());
    std::copy(input_dims.cbegin(), input_dims.cend(), output_dims.begin());
    for (size_t axis : reduction_axes()) {
      (output_dims.end() - num_input_dims())[axis] = 1;
    }
    const size_t num_output_elements =
      std::accumulate(output_dims.begin(), output_dims.end(), size_t(1), std::multiplies<size_t>());

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_dims[i - 1] == 1 ? 0 : output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + num_input_elements());
    std::vector<float> output(num_output_elements);
    std::vector<double> output_ref(num_output_elements);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0.0);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  output_ref[i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5]] +=
                    input[i * input_strides[0] + j * input_strides[1] + k * input_strides[2] + l * input_strides[3] + m * input_strides[4] + n * input_strides[5]];
                }
              }
            }
          }
        }
      }
      const double scale = double(num_input_elements() / num_output_elements);
      for (double& value : output_ref) {
        value /= scale;
      }

      // Create, setup, run, and destroy a mean operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t mean_op = nullptr;

      const xnn_status status = xnn_create_mean_nd_f32(
          /*flags=*/0, &mean_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, mean_op);

      // Smart pointer to automatically delete mean_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_mean_op(mean_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_mean_nd_f32(
          mean_op,
          num_reduction_axes(),
          reduction_axes().data(),
          num_input_dims(),
          input_shape().data(),
          auto_threadpool.get()));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_mean_nd_f32(
          mean_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(mean_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  ASSERT_NEAR(output[index], output_ref[index], 3.0e-6f * std::abs(output_ref[index]))
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")";
                }
              }
            }
          }
        }
      }
    }
  }

  void TestQS8() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input_shape().cbegin(), input_shape().cend(), input_dims.end() - num_input_dims());
    std::copy(input_dims.cbegin(), input_dims.cend(), output_dims.begin());
    for (size_t axis : reduction_axes()) {
      (output_dims.end() - num_input_dims())[axis] = 1;
    }
    const size_t num_output_elements =
      std::accumulate(output_dims.begin(), output_dims.end(), size_t(1), std::multiplies<size_t>());

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_dims[i - 1] == 1 ? 0 : output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + num_input_elements());
    std::vector<int8_t> output(num_output_elements);
    std::vector<float> output_ref(num_output_elements);
    std::vector<int8_t> output_ref_qs8(num_output_elements);
    std::vector<int32_t> accumulator(num_output_elements);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::fill(accumulator.begin(), accumulator.end(), 0);

      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      const int32_t num_reduced_elements = num_input_elements() / num_output_elements;
      const float mean_scale = static_cast<double>(1.0f) / num_reduced_elements;
      const float input_scale = 0.5f;
      const float output_scale = 0.75f;
      const int8_t input_zero_point = i8dist(rng);
      const int8_t output_zero_point = i8dist(rng);
      const int8_t quantized_output_min = xnn_qs8_quantize(-INFINITY, output_scale, output_zero_point);
      const int8_t quantized_output_max = xnn_qs8_quantize(INFINITY, output_scale, output_zero_point);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  size_t input_idx = i * input_strides[0] + j * input_strides[1] + k * input_strides[2] + l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  size_t output_idx = i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  accumulator[output_idx] += static_cast<int32_t>(input[input_idx]);
                }
              }
            }
          }
        }
      }

      for (size_t idx = 0; idx < output_ref.size(); ++idx) {
        output_ref[idx] = static_cast<float>(accumulator[idx] - static_cast<int32_t>(input_zero_point) * num_reduced_elements);
        output_ref[idx] *= input_scale * mean_scale / output_scale;
        output_ref[idx] = std::min(output_ref[idx], static_cast<float>(static_cast<int32_t>(quantized_output_max) - static_cast<int32_t>(output_zero_point)));
        output_ref[idx] = std::max(output_ref[idx], static_cast<float>(static_cast<int32_t>(quantized_output_min) - static_cast<int32_t>(output_zero_point)));
        output_ref_qs8[idx] = static_cast<int8_t>(std::lrintf(output_ref[idx]) + static_cast<int32_t>(output_zero_point));
      }

      // Create, setup, run, and destroy a mean operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t mean_op = nullptr;

      const xnn_status status = xnn_create_mean_nd_qs8(
          input_scale / output_scale, input_zero_point, output_zero_point,
          /*flags=*/0, &mean_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, mean_op);

      // Smart pointer to automatically delete mean_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_mean_op(mean_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_mean_nd_qs8(
          mean_op,
          num_reduction_axes(),
          reduction_axes().data(),
          num_input_dims(),
          input_shape().data(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_mean_nd_qs8(
          mean_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(mean_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  ASSERT_EQ(output[index], output_ref_qs8[index])
                    << "(i, j, k, l, m, n) = (" << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")";
                }
              }
            }
          }
        }
      }
    }
  }

  void TestQU8() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
        std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy(input_shape().cbegin(), input_shape().cend(), input_dims.end() - num_input_dims());
    std::copy(input_dims.cbegin(), input_dims.cend(), output_dims.begin());
    for (size_t axis : reduction_axes()) {
      (output_dims.end() - num_input_dims())[axis] = 1;
    }
    const size_t num_output_elements =
      std::accumulate(output_dims.begin(), output_dims.end(), size_t(1), std::multiplies<size_t>());

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_dims[i - 1] == 1 ? 0 : output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + num_input_elements());
    std::vector<uint8_t> output(num_output_elements);
    std::vector<float> output_ref(num_output_elements);
    std::vector<uint8_t> output_ref_qu8(num_output_elements);
    std::vector<uint32_t> accumulator(num_output_elements);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::fill(accumulator.begin(), accumulator.end(), 0);

      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      const int32_t num_reduced_elements = num_input_elements() / num_output_elements;
      const float mean_scale = static_cast<double>(1.0f) / num_reduced_elements;
      const float input_scale = 0.5f;
      const float output_scale = 0.75f;
      const uint8_t input_zero_point = u8dist(rng);
      const uint8_t output_zero_point = u8dist(rng);
      const uint8_t quantized_output_min = xnn_qu8_quantize(-INFINITY, output_scale, output_zero_point);
      const uint8_t quantized_output_max = xnn_qu8_quantize(INFINITY, output_scale, output_zero_point);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  size_t input_idx = i * input_strides[0] + j * input_strides[1] + k * input_strides[2] + l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  size_t output_idx = i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  accumulator[output_idx] += static_cast<uint32_t>(input[input_idx]);
                }
              }
            }
          }
        }
      }

      for (size_t idx = 0; idx < output_ref.size(); ++idx) {
        output_ref[idx] = static_cast<float>(static_cast<int64_t>(accumulator[idx]) - static_cast<int32_t>(input_zero_point) * num_reduced_elements);
        output_ref[idx] *= input_scale * mean_scale / output_scale;
        output_ref[idx] = std::min(output_ref[idx], static_cast<float>(static_cast<int32_t>(quantized_output_max) - static_cast<int32_t>(output_zero_point)));
        output_ref[idx] = std::max(output_ref[idx], static_cast<float>(static_cast<int32_t>(quantized_output_min) - static_cast<int32_t>(output_zero_point)));
        output_ref_qu8[idx] = static_cast<uint8_t>(std::lrintf(output_ref[idx]) + static_cast<int32_t>(output_zero_point));
      }

      // Create, setup, run, and destroy a mean operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t mean_op = nullptr;

      const xnn_status status = xnn_create_mean_nd_qu8(
          input_scale / output_scale, input_zero_point, output_zero_point,
          /*flags=*/0, &mean_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, mean_op);

      // Smart pointer to automatically delete mean_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_mean_op(mean_op, xnn_delete_operator);

      size_t workspace_size = SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_mean_nd_qu8(
          mean_op,
          num_reduction_axes(),
          reduction_axes().data(),
          num_input_dims(),
          input_shape().data(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_NE(workspace_size, SIZE_MAX);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_mean_nd_qu8(
          mean_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(mean_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] + l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  ASSERT_EQ(output[index], output_ref_qu8[index])
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
  std::vector<size_t> input_shape_;
  std::vector<size_t> reduction_axes_;
  bool multithreaded_{false};
  size_t iterations_{3};
};
