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
#include <limits>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


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

  inline VCvtMicrokernelTester& zero_point(int16_t zero_point) {
    this->zero_point_ = zero_point;
    return *this;
  }

  inline int16_t zero_point() const {
    return this->zero_point_;
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

  void Test(xnn_f16_f32_vcvt_ukernel_function vcvt, xnn_init_f16_f32_cvt_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-100.0f, 100.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<float> output(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f16rng));
      std::fill(output.begin(), output.end(), nanf(""));

      union xnn_f16_f32_cvt_params params;
      if (init_params) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint16_t), input.data(), output.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(fp32_to_bits(output[i]), fp32_to_bits(fp16_ieee_to_fp32_value(input[i])))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(4) << std::setfill('0') << input[i];
      }
    }
  }

  void Test(xnn_f32_f16_vcvt_ukernel_function vcvt, xnn_init_f32_f16_cvt_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-100.0f, 100.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<uint16_t> output(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), UINT16_C(0x7E));

      union xnn_f32_f16_cvt_params params;
      if (init_params) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(output[i], fp16_ieee_from_fp32_value(input[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

  void Test(xnn_f32_qs8_vcvt_ukernel_function vcvt, xnn_init_f32_qs8_cvt_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      union xnn_f32_qs8_cvt_params params;
      if (init_params) {
        init_params(&params, scale(), zero_point(), qmin(), qmax());
      }

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        float scaled_input = input[i] * scale();
        scaled_input = std::min<float>(scaled_input, float(qmax() - zero_point()));
        scaled_input = std::max<float>(scaled_input, float(qmin() - zero_point()));
        output_ref[i] = int8_t(std::lrintf(scaled_input) + long(zero_point()));
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

  void Test(xnn_f32_qu8_vcvt_ukernel_function vcvt, xnn_init_f32_qu8_cvt_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<uint8_t> output(batch_size());
    std::vector<uint8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      union xnn_f32_qu8_cvt_params params;
      init_params(&params, scale(), zero_point(), qmin(), qmax());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(float), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        float scaled_input = input[i] * scale();
        scaled_input = std::min<float>(scaled_input, float(qmax() - zero_point()));
        scaled_input = std::max<float>(scaled_input, float(qmin() - zero_point()));
        output_ref[i] = uint8_t(std::lrintf(scaled_input) + long(zero_point()));
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = 0x" << std::hex << std::setw(8) << std::setfill('0') << fp32_to_bits(input[i])
          << " (" << input[i] << ")";
      }
    }
  }

  void Test(xnn_qs8_f32_vcvt_ukernel_function vcvt, xnn_init_qs8_f32_cvt_params_fn init_params) const {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    auto i8rng = std::bind(distribution, std::ref(rng));

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<float> output(batch_size());
    std::vector<float> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i8rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      union xnn_qs8_f32_cvt_params params;
      init_params(&params, scale(), zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref[i] = float(int16_t(input[i]) - zero_point()) * scale();
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(output[i], output_ref[i])
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qu8_f32_vcvt_ukernel_function vcvt, xnn_init_qu8_f32_cvt_params_fn init_params) const {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    auto u8rng = std::bind(distribution, std::ref(rng));

    std::vector<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<float> output(batch_size());
    std::vector<float> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      union xnn_qu8_f32_cvt_params params;
      init_params(&params, scale(), zero_point());

      // Call optimized micro-kernel.
      vcvt(batch_size() * sizeof(uint8_t), input.data(), output.data(), &params);

      // Compute reference results
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref[i] = float(int16_t(input[i]) - zero_point()) * scale();
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(output[i], output_ref[i])
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

 private:
  float scale_ = 1.75f;
  int16_t zero_point_ = 1;
  int16_t qmin_ = std::numeric_limits<int16_t>::min();
  int16_t qmax_ = std::numeric_limits<int16_t>::max();
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};
