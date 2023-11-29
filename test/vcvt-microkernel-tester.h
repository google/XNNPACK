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
#include <iomanip>
#include <ios>
#include <limits>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


class VCvtMicrokernelTester {
 public:
  inline VCvtMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VCvtMicrokernelTester& scale(float scale) {
    assert(scale > 0.0f);
    assert(std::isnormal(scale));
    this->scale_ = scale;
    return *this;
  }

  inline float scale() const {
    return this->scale_;
  }

  inline VCvtMicrokernelTester& input_zero_point(int16_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline int16_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline VCvtMicrokernelTester& output_zero_point(int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline int16_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline VCvtMicrokernelTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline int16_t qmin() const {
    return this->qmin_;
  }

  inline VCvtMicrokernelTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline int16_t qmax() const {
    return this->qmax_;
  }

  inline VCvtMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_f32_vcvt_ukernel_fn vcvt, xnn_init_f16_f32_cvt_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-100.0f, 100.0f);

    std::vector<uint16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<float> output(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), nanf(""));

      union xnn_f16_f32_cvt_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint16_t), input.data(), output.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(float_as_uint32(output[i]), float_as_uint32(fp16_ieee_to_fp32_value(input[i])))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(4) << std::setfill('0') << input[i];
      }
    }
  }

  void Test(xnn_f32_f16_vcvt_ukernel_fn vcvt, xnn_init_f32_f16_cvt_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-100.0f, 100.0f);

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<uint16_t> output(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      union xnn_f32_f16_cvt_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(output[i], fp16_ieee_from_fp32_value(input[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

  void Test(xnn_f16_qs8_vcvt_ukernel_fn vcvt, xnn_init_f16_qs8_cvt_params_fn init_params) {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<float> input_float(batch_size());
    std::vector<uint16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    const float scale_fp16 = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(scale()));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input_float.begin(), input_float.end(), [&]() { return f32dist(rng); });
      std::transform(input_float.begin(), input_float.end(), input.begin(), [](float f) { return fp16_ieee_from_fp32_value(f); });

      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      union xnn_f16_qs8_cvt_params params;
      init_params(&params, fp16_ieee_from_fp32_value(scale()), output_zero_point(), qmin(), qmax());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint16_t), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        float scaled_input = fp16_ieee_to_fp32_value(input[i]) * scale_fp16;
        scaled_input = std::min<float>(scaled_input, float(qmax() - output_zero_point()));
        scaled_input = std::max<float>(scaled_input, float(qmin() - output_zero_point()));
        output_ref[i] = int8_t(std::lrintf(scaled_input) + long(output_zero_point()));
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_NEAR(int32_t(output[i]), int32_t(output_ref[i]), 1)
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(input[i])
          << " (" << input[i] << ")" << " INPUT " << fp16_ieee_to_fp32_value(input[i]) << " scale " << scale() << " zp " << (int)output_zero_point();
      }
    }
  }

  void Test(xnn_f32_qs8_vcvt_ukernel_fn vcvt, xnn_init_f32_qs8_cvt_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      union xnn_f32_qs8_cvt_params params;
      init_params(&params, scale(), output_zero_point(), qmin(), qmax());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        float scaled_input = input[i] * scale();
        scaled_input = std::min<float>(scaled_input, float(qmax() - output_zero_point()));
        scaled_input = std::max<float>(scaled_input, float(qmin() - output_zero_point()));
        output_ref[i] = int8_t(std::lrintf(scaled_input) + long(output_zero_point()));
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

  void Test(xnn_f32_qu8_vcvt_ukernel_fn vcvt, xnn_init_f32_qu8_cvt_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<uint8_t> output(batch_size());
    std::vector<uint8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      union xnn_f32_qu8_cvt_params params;
      init_params(&params, scale(), output_zero_point(), qmin(), qmax());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        float scaled_input = input[i] * scale();
        scaled_input = std::min<float>(scaled_input, float(qmax() - output_zero_point()));
        scaled_input = std::max<float>(scaled_input, float(qmin() - output_zero_point()));
        output_ref[i] = uint8_t(std::lrintf(scaled_input) + long(output_zero_point()));
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << float_as_uint32(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

  void Test(xnn_qs8_vcvt_ukernel_fn vcvt, xnn_init_qs8_cvt_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      union xnn_qs8_cvt_params params;
      init_params(&params, scale(), input_zero_point(), output_zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t multiplier = (int32_t) lrintf(-256.0f * scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = (input_zero_point() - input[i]) * 128;
        int32_t output_value = math_asr_s32(input_value * multiplier + INT32_C(0x4000), 15) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<int8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<int8_t>::min());
        output_ref[i] = static_cast<int8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qs16_qs8_vcvt_ukernel_fn vcvt, xnn_init_qs16_qs8_cvt_params_fn init_params) const {
    ASSERT_EQ(input_zero_point(), 0);
    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int16_t> i16dist;

    std::vector<int16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i16dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      union xnn_qs16_qs8_cvt_params params;
      init_params(&params, scale(), output_zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(int16_t), input.data(), output.data(), &params);

      // Compute reference results
      const int64_t multiplier = std::llrintf(65536.0f * scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int64_t input_value = input[i];
        int32_t output_value = static_cast<int32_t>(math_asr_s64(input_value * multiplier + INT64_C(0x8000), 16)) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<int8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<int8_t>::min());
        output_ref[i] = static_cast<int8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << input[i]
          << " * scale " << scale() << " = " << int32_t(output_ref[i]);
      }
    }
  }

  void Test(xnn_qs8_f16_vcvt_ukernel_fn vcvt, xnn_init_qs8_f16_cvt_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<uint16_t> output(batch_size());
    std::vector<float> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00));

      union xnn_qs8_f16_cvt_params params;
      init_params(&params, fp16_ieee_from_fp32_value(scale()), input_zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref[i] = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(float(int16_t(input[i]) - input_zero_point()) * scale()));
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(output_ref[i], fp16_ieee_to_fp32_value(output[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qs8_f32_vcvt_ukernel_fn vcvt, xnn_init_qs8_f32_cvt_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<float> output(batch_size());
    std::vector<float> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      union xnn_qs8_f32_cvt_params params;
      init_params(&params, scale(), input_zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref[i] = float(int16_t(input[i]) - input_zero_point()) * scale();
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(output[i], output_ref[i])
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qu8_vcvt_ukernel_fn vcvt, xnn_init_qu8_cvt_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(batch_size());
    std::vector<uint8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      union xnn_qu8_cvt_params params;
      init_params(&params, scale(), input_zero_point(), output_zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t multiplier = (int32_t) lrintf(-256.0f * scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = (input_zero_point() - input[i]) * 128;
        int32_t output_value = math_asr_s32(input_value * multiplier + INT32_C(0x4000), 15) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<uint8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<uint8_t>::min());
        output_ref[i] = static_cast<uint8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qu8_f32_vcvt_ukernel_fn vcvt, xnn_init_qu8_f32_cvt_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<float> output(batch_size());
    std::vector<float> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      union xnn_qu8_f32_cvt_params params;
      init_params(&params, scale(), input_zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint8_t), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref[i] = float(int16_t(input[i]) - input_zero_point()) * scale();
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(output[i], output_ref[i])
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

 private:
  float scale_ = 1.75f;
  int16_t input_zero_point_ = 1;
  int16_t output_zero_point_ = 5;
  int16_t qmin_ = std::numeric_limits<int16_t>::min();
  int16_t qmax_ = std::numeric_limits<int16_t>::max();
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};
