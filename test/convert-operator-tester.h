// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>


class ConvertOperatorTester {
 public:
  inline ConvertOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline ConvertOperatorTester& input_stride(size_t input_stride) {
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

  inline ConvertOperatorTester& output_stride(size_t output_stride) {
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

  inline ConvertOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ConvertOperatorTester& scale(float scale) {
    assert(scale >= 0.0f);
    assert(std::isnormal(scale));
    this->scale_ = scale;
    return *this;
  }

  inline float scale() const {
    return this->scale_;
  }

  inline ConvertOperatorTester& zero_point(int16_t zero_point) {
    this->zero_point_ = zero_point;
    return *this;
  }

  inline int16_t zero_point() const {
    return this->zero_point_;
  }

  inline ConvertOperatorTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline int16_t qmin() const {
    return this->qmin_;
  }

  inline ConvertOperatorTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline int16_t qmax() const {
    return this->qmax_;
  }

  inline ConvertOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF16toF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f16rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] = fp16_ieee_to_fp32_value(input[i * input_stride() + c]);
        }
      }

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_f16_f32(
          channels(), input_stride(), output_stride(),
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convert_nc_f16_f32(
          convert_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convert_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(output_ref[i * channels() + c], output[i * output_stride() + c])
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestF32toF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<uint16_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), UINT16_C(0x7E));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] = fp16_ieee_from_fp32_value(input[i * input_stride() + c]);
        }
      }

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_f32_f16(
          channels(), input_stride(), output_stride(),
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convert_nc_f32_f16(
          convert_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convert_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(output_ref[i * channels() + c], output[i * output_stride() + c])
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestF32toQS8() const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<int8_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), UINT16_C(0x7E));

      // Compute reference results.
      const float inv_scale = 1.0f / scale();
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          float scaled_input = input[i * input_stride() + c] * inv_scale;
          scaled_input = std::min<float>(scaled_input, float(qmax() - zero_point()));
          scaled_input = std::max<float>(scaled_input, float(qmin() - zero_point()));
          output_ref[i * channels() + c] = int8_t(std::lrintf(scaled_input) + long(zero_point()));
        }
      }

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_f32_qs8(
          channels(), input_stride(), output_stride(),
          scale(), int8_t(zero_point()), int8_t(qmin()), int8_t(qmax()),
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convert_nc_f32_qs8(
          convert_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convert_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(int32_t(output_ref[i * channels() + c]), int32_t(output[i * output_stride() + c]))
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestF32toQU8() const {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), UINT16_C(0x7E));

      // Compute reference results.
      const float inv_scale = 1.0f / scale();
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          float scaled_input = input[i * input_stride() + c] * inv_scale;
          scaled_input = std::min<float>(scaled_input, float(qmax() - zero_point()));
          scaled_input = std::max<float>(scaled_input, float(qmin() - zero_point()));
          output_ref[i * channels() + c] = uint8_t(std::lrintf(scaled_input) + long(zero_point()));
        }
      }

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_f32_qu8(
          channels(), input_stride(), output_stride(),
          scale(), uint8_t(zero_point()), uint8_t(qmin()), uint8_t(qmax()),
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convert_nc_f32_qu8(
          convert_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convert_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(uint32_t(output_ref[i * channels() + c]), uint32_t(output[i * output_stride() + c]))
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestQS8toF32() const {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i8rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] = float(input[i * input_stride() + c] - zero_point()) * scale();
        }
      }

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_qs8_f32(
          channels(), input_stride(), output_stride(),
          scale(), int8_t(zero_point()),
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convert_nc_qs8_f32(
          convert_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convert_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(output_ref[i * channels() + c], output[i * output_stride() + c])
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestQU8toF32() const {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
      std::ref(rng));

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[i * channels() + c] = float(input[i * input_stride() + c] - zero_point()) * scale();
        }
      }

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_qu8_f32(
          channels(), input_stride(), output_stride(),
          scale(), uint8_t(zero_point()),
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convert_nc_qu8_f32(
          convert_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convert_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(output_ref[i * channels() + c], output[i * output_stride() + c])
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float scale_{150.0f};
  int16_t zero_point_{1};
  int16_t qmin_{std::numeric_limits<int16_t>::min()};
  int16_t qmax_{std::numeric_limits<int16_t>::max()};
  size_t iterations_{15};
};
