// Copyright 2021 Google LLC
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

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/requantization.h>


class VMulMicrokernelTester {
 public:
  inline VMulMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VMulMicrokernelTester& inplace_a(bool inplace_a) {
    this->inplace_a_ = inplace_a;
    return *this;
  }

  inline bool inplace_a() const {
    return this->inplace_a_;
  }

  inline VMulMicrokernelTester& inplace_b(bool inplace_b) {
    this->inplace_b_ = inplace_b;
    return *this;
  }

  inline bool inplace_b() const {
    return this->inplace_b_;
  }

  inline VMulMicrokernelTester& a_scale(float a_scale) {
    assert(a_scale > 0.0f);
    assert(std::isnormal(a_scale));
    this->a_scale_ = a_scale;
    return *this;
  }

  inline float a_scale() const {
    return this->a_scale_;
  }

  inline VMulMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  inline uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  inline VMulMicrokernelTester& b_scale(float b_scale) {
    assert(b_scale > 0.0f);
    assert(std::isnormal(b_scale));
    this->b_scale_ = b_scale;
    return *this;
  }

  inline float b_scale() const {
    return this->b_scale_;
  }

  inline VMulMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  inline uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  inline VMulMicrokernelTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  inline float y_scale() const {
    return this->y_scale_;
  }

  inline VMulMicrokernelTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  inline uint8_t y_zero_point() const {
    return this->y_zero_point_;
  }

  inline VMulMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VMulMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VMulMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(
      xnn_qu8_vmul_minmax_ukernel_function vmul_minmax,
      xnn_init_qu8_mul_minmax_params_fn init_params,
      xnn_qu8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
    std::vector<float> y_fp(batch_size());
    std::vector<uint8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* a_data = inplace_a() ? y.data() : a.data();
      const uint8_t* b_data = inplace_b() ? y.data() : b.data();

      // Prepare parameters.
      const float product_scale = a_scale() * b_scale();
      const float product_output_scale = product_scale / y_scale();
      xnn_qu8_mul_minmax_params quantization_params;
      init_params(
        &quantization_params,
        a_zero_point(), b_zero_point(), y_zero_point(),
        product_output_scale, qmin(), qmax());

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t acc =
          (int32_t(a_data[i]) - int32_t(a_zero_point())) * (int32_t(b_data[i]) - int32_t(b_zero_point()));
        y_fp[i] = float(y_zero_point()) + product_output_scale * float(acc);
        y_fp[i] = std::min<float>(y_fp[i], float(int32_t(qmax())));
        y_fp[i] = std::max<float>(y_fp[i], float(int32_t(qmin())));
        y_ref[i] = requantize(
          acc, product_output_scale, y_zero_point(), qmin(), qmax());
      }

      // Call optimized micro-kernel.
      vmul_minmax(batch_size(), a_data, b_data, y.data(), &quantization_params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
          << "at element " << i << " / " << batch_size();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
          << "at element " << i << " / " << batch_size();
        ASSERT_NEAR(float(int32_t(y[i])), y_fp[i], 0.6f)
          << "at element " << i << " / " << batch_size();
        ASSERT_EQ(uint32_t(y[i]), uint32_t(y_ref[i]))
          << "at element " << i << " / " << batch_size();
      }
    }
  }

  void Test(
      xnn_qs8_vmul_minmax_ukernel_function vmul_minmax,
      xnn_init_qs8_mul_minmax_params_fn init_params,
      xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      rng);

    std::vector<int8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(int8_t) : 0));
    std::vector<float> y_fp(batch_size());
    std::vector<int8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(i8rng));
      std::generate(b.begin(), b.end(), std::ref(i8rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(i8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const int8_t* a_data = inplace_a() ? y.data() : a.data();
      const int8_t* b_data = inplace_b() ? y.data() : b.data();

      // Prepare parameters.
      const float product_scale = a_scale() * b_scale();
      const float product_output_scale = product_scale / y_scale();
      EXPECT_GE(product_output_scale, 0x1.0p-32f);
      xnn_qs8_mul_minmax_params quantization_params;
      init_params(
        &quantization_params,
        int8_t(a_zero_point() - 0x80), int8_t(b_zero_point() - 0x80), int8_t(y_zero_point() - 0x80),
        product_output_scale, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t acc =
          (int32_t(a_data[i]) - int32_t(a_zero_point() - 0x80)) * (int32_t(b_data[i]) - int32_t(b_zero_point() - 0x80));
        y_fp[i] = float(y_zero_point() - 0x80) + product_output_scale * float(acc);
        y_fp[i] = std::min<float>(y_fp[i], float(int32_t(qmax() - 0x80)));
        y_fp[i] = std::max<float>(y_fp[i], float(int32_t(qmin() - 0x80)));
        y_ref[i] = requantize(
          acc, product_output_scale, int8_t(y_zero_point() - 0x80), int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
      }

      // Call optimized micro-kernel.
      vmul_minmax(batch_size(), a_data, b_data, y.data(), &quantization_params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_LE(int32_t(y[i]), int32_t(qmax() - 0x80))
          << "at element " << i << " / " << batch_size();
        ASSERT_GE(int32_t(y[i]), int32_t(qmin() - 0x80))
          << "at element " << i << " / " << batch_size();
        ASSERT_EQ(int32_t(y_ref[i]), int32_t(y[i]))
          << "at element " << i << " / " << batch_size();
        ASSERT_NEAR(float(int32_t(y[i])), y_fp[i], 0.6f)
          << "at element " << i << " / " << batch_size();
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_a_{false};
  bool inplace_b_{false};
  float a_scale_{0.75f};
  float b_scale_{1.25f};
  float y_scale_{0.96875f};
  uint8_t a_zero_point_{121};
  uint8_t b_zero_point_{127};
  uint8_t y_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
