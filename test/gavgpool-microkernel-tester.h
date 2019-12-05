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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class GAvgPoolMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline GAvgPoolMicrokernelTester& m(size_t m) {
    assert(m != 0);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GAvgPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GAvgPoolMicrokernelTester& nr(size_t nr) {
    assert(nr != 0);
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }

  inline GAvgPoolMicrokernelTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return n();
    } else {
      assert(this->x_stride_ >= n());
      return this->x_stride_;
    }
  }

  inline GAvgPoolMicrokernelTester& x_scale(float x_scale) {
    assert(x_scale > 0.0f);
    assert(std::isnormal(x_scale));
    this->x_scale_ = x_scale;
    return *this;
  }

  inline float x_scale() const {
    return this->x_scale_;
  }

  inline GAvgPoolMicrokernelTester& x_zero_point(uint8_t x_zero_point) {
    this->x_zero_point_ = x_zero_point;
    return *this;
  }

  inline uint8_t x_zero_point() const {
    return this->x_zero_point_;
  }

  inline GAvgPoolMicrokernelTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  inline float y_scale() const {
    return this->y_scale_;
  }

  inline GAvgPoolMicrokernelTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  inline uint8_t y_zero_point() const {
    return this->y_zero_point_;
  }

  inline GAvgPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GAvgPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GAvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_gavgpool_up_ukernel_function gavgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x((m() - 1) * x_stride() + n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> zero(n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> y_ref(n());
    std::vector<float> y_fp(n());
    std::vector<int32_t> y_acc(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      // Prepare quantization parameters.
      union xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(m()),
            x_scale() / (y_scale() * float(m())),
            y_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(m()),
            x_scale() / (y_scale() * float(m())),
            y_zero_point(), qmin(), qmax());
          break;
      }
      const union xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(x_zero_point()) * int32_t(m()),
          x_scale() / (y_scale() * float(m())),
          y_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t j = 0; j < n(); j++) {
        int32_t acc = scalar_quantization_params.scalar.bias;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * x_stride() + j];
        }
        y_acc[j] = acc;
        y_ref[j] = xnn_avgpool_quantize(acc, scalar_quantization_params);
        y_fp[j] = float(acc) * (x_scale() / (y_scale() * float(m()))) + float(y_zero_point());
        y_fp[j] = std::min<float>(y_fp[j], float(qmax()));
        y_fp[j] = std::max<float>(y_fp[j], float(qmin()));
      }

      // Call optimized micro-kernel.
      gavgpool(m(), n(),
        x.data(), x_stride() * sizeof(uint8_t),
        zero.data(),
        y.data(),
        &quantization_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), y_fp[i], 0.5f)
          << "at position " << i << ", m = " << m() << ", n = " << n() << ", acc = " << y_acc[i];
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at position " << i << ", m = " << m() << ", n = " << n() << ", acc = " << y_acc[i];
      }
    }
  }

  void Test(xnn_q8_gavgpool_mp_ukernel_function gavgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x((m() - 1) * x_stride() + n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buf(n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> zero(n() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> y_ref(n());
    std::vector<float> y_fp(n());
    std::vector<int32_t> y_acc(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      // Prepare quantization parameters.
      union xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(m()),
            x_scale() / (y_scale() * float(m())),
            y_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(m()),
            x_scale() / (y_scale() * float(m())),
            y_zero_point(), qmin(), qmax());
          break;
      }
      const union xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(x_zero_point()) * int32_t(m()),
          x_scale() / (y_scale() * float(m())),
          y_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t j = 0; j < n(); j++) {
        int32_t acc = scalar_quantization_params.scalar.bias;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * x_stride() + j];
        }

        y_acc[j] = acc;
        y_ref[j] = xnn_avgpool_quantize(acc, scalar_quantization_params);
        y_fp[j] = float(acc) * (x_scale() / (y_scale() * float(m()))) + float(y_zero_point());
        y_fp[j] = std::min<float>(y_fp[j], float(qmax()));
        y_fp[j] = std::max<float>(y_fp[j], float(qmin()));
      }

      // Call optimized micro-kernel.
      gavgpool(m(), n(),
        x.data(), x_stride() * sizeof(uint8_t),
        zero.data(),
        buf.data(),
        y.data(),
        &quantization_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(float(int32_t(y[i])), y_fp[i], 0.5f)
          << "at position " << i << ", m = " << m() << ", n = " << n() << ", acc = " << y_acc[i];
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at position " << i << ", m = " << m() << ", n = " << n() << ", acc = " << y_acc[i];
      }
    }
  }

  void Test(xnn_f32_gavgpool_up_ukernel_function gavgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> x((m() - 1) * x_stride() + n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(n());
    std::vector<float> y_ref(n());

    std::fill(zero.begin(), zero.end(), 0.0f);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t j = 0; j < n(); j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * x_stride() + j];
        }
        y_ref[j] = acc / float(m());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float y_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Prepare micro-kernel parameters.
      union xnn_f32_avgpool_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_avgpool_params(
            1.0f / float(m()), y_min, y_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_avgpool_params(
            1.0f / float(m()), y_min, y_max);
          break;
      }

      // Call optimized micro-kernel.
      gavgpool(m(), n(),
        x.data(), x_stride() * sizeof(float),
        zero.data(),
        y.data(),
        &params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(y[i], y_max)
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(y[i], y_min)
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at position " << i << ", m = " << m() << ", n = " << n();
      }
    }
  }

  void Test(xnn_f32_gavgpool_mp_ukernel_function gavgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> x((m() - 1) * x_stride() + n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> buf(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero(n() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(n());
    std::vector<float> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t j = 0; j < n(); j++) {
        float acc = 0.0f;
        for (size_t i = 0; i < m(); i++) {
          acc += x[i * x_stride() + j];
        }
        y_ref[j] = acc / float(m());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float y_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Prepare micro-kernel parameters.
      union xnn_f32_avgpool_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_avgpool_params(
            1.0f / float(m()), y_min, y_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_avgpool_params(
            1.0f / float(m()), y_min, y_max);
          break;
      }

      // Clamp reference results.
      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Call optimized micro-kernel.
      gavgpool(m(), n(),
        x.data(), x_stride() * sizeof(float),
        zero.data(),
        buf.data(),
        y.data(),
        &params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(y[i], y_max)
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_GE(y[i], y_min)
          << "at position " << i << ", m = " << m() << ", n = " << n();
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at position " << i << ", m = " << m() << ", n = " << n();
      }
    }
  }

 private:
  size_t m_{1};
  size_t n_{1};
  size_t nr_{1};
  size_t x_stride_{0};
  float x_scale_{1.25f};
  float y_scale_{0.75f};
  uint8_t x_zero_point_{121};
  uint8_t y_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
