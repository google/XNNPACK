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

#include <xnnpack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class VAddMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline VAddMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline VAddMicrokernelTester& inplace_a(bool inplace_a) {
    this->inplace_a_ = inplace_a;
    return *this;
  }

  inline bool inplace_a() const {
    return this->inplace_a_;
  }

  inline VAddMicrokernelTester& inplace_b(bool inplace_b) {
    this->inplace_b_ = inplace_b;
    return *this;
  }

  inline bool inplace_b() const {
    return this->inplace_b_;
  }

  inline VAddMicrokernelTester& a_scale(float a_scale) {
    assert(a_scale > 0.0f);
    assert(std::isnormal(a_scale));
    this->a_scale_ = a_scale;
    return *this;
  }

  inline float a_scale() const {
    return this->a_scale_;
  }

  inline VAddMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  inline uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  inline VAddMicrokernelTester& b_scale(float b_scale) {
    assert(b_scale > 0.0f);
    assert(std::isnormal(b_scale));
    this->b_scale_ = b_scale;
    return *this;
  }

  inline float b_scale() const {
    return this->b_scale_;
  }

  inline VAddMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  inline uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  inline VAddMicrokernelTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  inline float y_scale() const {
    return this->y_scale_;
  }

  inline VAddMicrokernelTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  inline uint8_t y_zero_point() const {
    return this->y_zero_point_;
  }

  inline VAddMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VAddMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VAddMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_vadd_minmax_ukernel_function vadd_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> a(n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> b(n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(n() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
    std::vector<float> y_fp(n());
    std::vector<uint8_t> y_ref(n());
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

      // Prepare quantization parameters.
      xnn_q8_add_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_add_params(
            a_zero_point(), b_zero_point(), y_zero_point(),
            a_scale() / y_scale(), b_scale() / y_scale(),
            qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_add_params(
            a_zero_point(), b_zero_point(), y_zero_point(),
            a_scale() / y_scale(), b_scale() / y_scale(),
            qmin(), qmax());
          break;
      }
      const xnn_q8_add_params scalar_quantization_params =
          xnn_init_scalar_q8_add_params(
            a_zero_point(), b_zero_point(), y_zero_point(),
            a_scale() / y_scale(), b_scale() / y_scale(),
            qmin(), qmax());

      // Compute reference results.
      for (size_t i = 0; i < n(); i++) {
        y_fp[i] = float(y_zero_point()) +
          float(int32_t(a_data[i]) - int32_t(a_zero_point())) * (a_scale() / y_scale()) +
          float(int32_t(b_data[i]) - int32_t(b_zero_point())) * (b_scale() / y_scale());
        y_fp[i] = std::min<float>(y_fp[i], float(qmax()));
        y_fp[i] = std::max<float>(y_fp[i], float(qmin()));
        y_ref[i] = xnn_add_quantize(a_data[i], b_data[i], scalar_quantization_params);
      }

      // Call optimized micro-kernel.
      vadd_minmax(n(), a_data, b_data, y.data(), &quantization_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
          << "at " << i << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
          << "at " << i << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), y_fp[i], 0.6f)
          << "at " << i << ", n = " << n();
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t n_{1};
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
