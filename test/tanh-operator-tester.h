// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "replicable_random_device.h"

class TanhOperatorTester {
 public:
  TanhOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  TanhOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->input_stride_ >= this->channels_);
      return this->input_stride_;
    }
  }

  TanhOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->output_stride_ >= this->channels_);
      return this->output_stride_;
    }
  }

  TanhOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  TanhOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  TanhOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  float output_scale() const {
    return 1.0f / 128.0f;
  }

  uint8_t output_zero_point() const {
    return 128;
  }

  TanhOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  TanhOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  TanhOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-5.0f, 5.0f);

    std::vector<uint16_t> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = fp16_ieee_to_fp32_value(input[i * input_stride() + c]);
          output_ref[i * channels() + c] = std::tanh(x);
        }
      }

      // Create, setup, run, and destroy Sigmoid operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t tanh_op = nullptr;

      const xnn_status status = xnn_create_tanh_nc_f16(
          0, &tanh_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, tanh_op);

      // Smart pointer to automatically delete tanh_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_tanh_op(tanh_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success, xnn_reshape_tanh_nc_f16(tanh_op, batch_size(),
          channels(), input_stride(), output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, xnn_setup_tanh_nc_f16(tanh_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(tanh_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              fp16_ieee_to_fp32_value(output[i * output_stride() + c]),
              output_ref[i * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[i * channels() + c]) * 5.0e-3f));
        }
      }
    }
  }

  void TestF32() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-10.0f, 10.0f);

    std::vector<float> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<double> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const double x = input[i * input_stride() + c];
          output_ref[i * channels() + c] = std::tanh(x);
        }
      }

      // Create, setup, run, and destroy Tanh operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t tanh_op = nullptr;

      xnn_status status = xnn_create_tanh_nc_f32(
          0, &tanh_op);
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, tanh_op);

      // Smart pointer to automatically delete tanh_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_tanh_op(tanh_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success, xnn_reshape_tanh_nc_f32(tanh_op, batch_size(),
          channels(), input_stride(), output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, xnn_setup_tanh_nc_f32(tanh_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(tanh_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
            output[i * output_stride() + c],
            output_ref[i * channels() + c],
            5.0e-6);
        }
      }
    }
  }

  void TestRunF32() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-10.0f, 10.0f);

    std::vector<float> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<double> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const double x = input[i * input_stride() + c];
          output_ref[i * channels() + c] = std::tanh(x);
        }
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_tanh_nc_f32(
          channels(),
          input_stride(),
          output_stride(),
          batch_size(),
          input.data(), output.data(),
          0, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
            output[i * output_stride() + c],
            output_ref[i * channels() + c],
            5.0e-6);
        }
      }
    }
  }

  void TestQS8() const {
    xnnpack::ReplicableRandomDevice rng;
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

    std::vector<int8_t> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input_scale() *
            (int32_t(input[i * input_stride() + c]) - int32_t(input_zero_point() - 0x80));
          const float tanh_x = std::tanh(x);
          const float scaled_tanh_x = tanh_x / output_scale();
          float y = scaled_tanh_x;
          y = std::min<float>(y, int32_t(qmax() - 0x80) - int32_t(output_zero_point() - 0x80));
          y = std::max<float>(y, int32_t(qmin() - 0x80) - int32_t(output_zero_point() - 0x80));
          output_ref[i * channels() + c] = y + int32_t(output_zero_point() - 0x80);
        }
      }

      // Create, setup, run, and destroy Sigmoid operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t tanh_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_tanh_nc_qs8(
          int8_t(input_zero_point() - 0x80), input_scale(),
          int8_t(output_zero_point() - 0x80), output_scale(),
          int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, &tanh_op));
      ASSERT_NE(nullptr, tanh_op);

      // Smart pointer to automatically delete tanh_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_tanh_op(tanh_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success, xnn_reshape_tanh_nc_qs8(tanh_op, batch_size(),
          channels(), input_stride(), output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, xnn_setup_tanh_nc_qs8(tanh_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(tanh_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.6f);
        }
      }
    }
  }

  void TestQU8() const {
    xnnpack::ReplicableRandomDevice rng;
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input((batch_size() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input_scale() *
            (int32_t(input[i * input_stride() + c]) - int32_t(input_zero_point()));
          const float tanh_x = std::tanh(x);
          const float scaled_tanh_x = tanh_x / output_scale();
          float y = scaled_tanh_x;
          y = std::min<float>(y, int32_t(qmax()) - int32_t(output_zero_point()));
          y = std::max<float>(y, int32_t(qmin()) - int32_t(output_zero_point()));
          output_ref[i * channels() + c] = y + int32_t(output_zero_point());
        }
      }

      // Create, setup, run, and destroy Sigmoid operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t tanh_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_tanh_nc_qu8(
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          qmin(), qmax(),
          0, &tanh_op));
      ASSERT_NE(nullptr, tanh_op);

      // Smart pointer to automatically delete tanh_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_tanh_op(tanh_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success, xnn_reshape_tanh_nc_qu8(tanh_op, batch_size(),
          channels(), input_stride(), output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, xnn_setup_tanh_nc_qu8(tanh_op, input.data(), output.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(tanh_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.6f);
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float input_scale_{0.75f};
  uint8_t input_zero_point_{121};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
