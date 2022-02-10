// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>


class ClampOperatorTester {
 public:
  inline ClampOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline ClampOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->input_stride_ >= this->channels_);
      return this->input_stride_;
    }
  }

  inline ClampOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->output_stride_ >= this->channels_);
      return this->output_stride_;
    }
  }

  inline ClampOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ClampOperatorTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline int16_t qmin() const {
    return this->qmin_;
  }

  inline ClampOperatorTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline int16_t qmax() const {
    return this->qmax_;
  }

  inline ClampOperatorTester& relu_activation(bool relu_activation) {
    this->relu_activation_ = relu_activation;
    return *this;
  }

  inline bool relu_activation() const {
    return this->relu_activation_;
  }

  inline ClampOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    ASSERT_LT(qmin(), qmax());
    ASSERT_FALSE(relu_activation());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(
      std::uniform_real_distribution<float>(std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()),
      std::ref(rng));
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f16rng));
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results.
      const float output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(float(qmin())));
      const float output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(float(qmax())));
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = fp16_ieee_to_fp32_value(input[i * input_stride() + c]);
          const float y = relu_activation() ? std::max(x, 0.f) : std::min(std::max(x, output_min), output_max);
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Clamp operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t clamp_op = nullptr;

      const xnn_status status = xnn_create_clamp_nc_f16(
        channels(), input_stride(), output_stride(),
        output_min, output_max,
        0, &clamp_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, clamp_op);

      // Smart pointer to automatically delete clamp_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_clamp_op(clamp_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_clamp_nc_f16(
          clamp_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(clamp_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_max)
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_GE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_min)
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_EQ(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_ref[i * channels() + c])
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", min " << output_min << ", max " << output_max;
        }
      }
    }
  }

  void TestF32() const {
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(
      std::uniform_real_distribution<float>(std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()),
      std::ref(rng));

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x = input[i * input_stride() + c];
          const float y = relu_activation() ? std::max(x, 0.f) :
            std::min(std::max(x, float(qmin())), float(qmax()));
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Clamp operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t clamp_op = nullptr;

      const float output_min = relu_activation() ? 0.0f : float(qmin());
      const float output_max = relu_activation() ? std::numeric_limits<float>::infinity() : float(qmax());
      ASSERT_EQ(xnn_status_success,
        xnn_create_clamp_nc_f32(
          channels(), input_stride(), output_stride(),
          output_min, output_max,
          0, &clamp_op));
      ASSERT_NE(nullptr, clamp_op);

      // Smart pointer to automatically delete clamp_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_clamp_op(clamp_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_clamp_nc_f32(
          clamp_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(clamp_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(output[i * output_stride() + c], output_max)
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_GE(output[i * output_stride() + c], output_min)
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_EQ(output_ref[i * channels() + c], output[i * output_stride() + c])
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", min " << output_min << ", max " << output_max;
        }
      }
    }
  }

  void TestS8() const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<int8_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i8rng));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const int8_t x = input[i * input_stride() + c];
          const int8_t y = std::min(std::max(x, int8_t(qmin())), int8_t(qmax()));
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Clamp operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t clamp_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_clamp_nc_s8(
          channels(), input_stride(), output_stride(),
          int8_t(qmin()), int8_t(qmax()),
          0, &clamp_op));
      ASSERT_NE(nullptr, clamp_op);

      // Smart pointer to automatically delete clamp_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_clamp_op(clamp_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_clamp_nc_s8(
          clamp_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(clamp_op, nullptr /* thread pool */));

      // Verify results .
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(int16_t(output[i * output_stride() + c]), qmax())
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_GE(int16_t(output[i * output_stride() + c]), qmin())
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_EQ(int16_t(output[i * output_stride() + c]), int16_t(output_ref[i * channels() + c]))
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", min " << qmin() << ", max " << qmax();
        }
      }
    }
  }

  void TestU8() const {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
      std::ref(rng));

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const uint8_t x = input[i * input_stride() + c];
          const uint8_t y = std::min(std::max(x, uint8_t(qmin())), uint8_t(qmax()));
          output_ref[i * channels() + c] = y;
        }
      }

      // Create, setup, run, and destroy Clamp operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t clamp_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_clamp_nc_u8(
          channels(), input_stride(), output_stride(),
          uint8_t(qmin()), uint8_t(qmax()),
          0, &clamp_op));
      ASSERT_NE(nullptr, clamp_op);

      // Smart pointer to automatically delete clamp_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_clamp_op(clamp_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_clamp_nc_u8(
          clamp_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(clamp_op, nullptr /* thread pool */));

      // Verify results .
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(int16_t(output[i * output_stride() + c]), qmax())
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_GE(int16_t(output[i * output_stride() + c]), qmin())
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
          ASSERT_EQ(int16_t(output[i * output_stride() + c]), int16_t(output_ref[i * channels() + c]))
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << channels()
            << ", min " << qmin() << ", max " << qmax();
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  int16_t qmin_{std::numeric_limits<int16_t>::min()};
  int16_t qmax_{std::numeric_limits<int16_t>::max()};
  bool relu_activation_{false};
  size_t iterations_{15};
};
