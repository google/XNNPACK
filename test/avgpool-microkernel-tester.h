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


class AvgPoolMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline AvgPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline AvgPoolMicrokernelTester& s(size_t s) {
    assert(s != 0);
    this->s_ = s;
    return *this;
  }

  inline size_t s() const {
    return this->s_;
  }

  inline AvgPoolMicrokernelTester& kh(size_t kh) {
    assert(kh != 0);
    this->kh_ = kh;
    return *this;
  }

  inline size_t kh() const {
    return this->kh_;
  }

  inline AvgPoolMicrokernelTester& kw(size_t kw) {
    assert(kw != 0);
    this->kw_ = kw;
    return *this;
  }

  inline size_t kw() const {
    return this->kw_;
  }

  inline size_t ks() const {
    return kh() * kw();
  }

  inline size_t packed_ks() const {
    if (ks() <= mr()) {
      return mr();
    } else {
      return (ks() - mr()) % qr() == 0 ? ks() : ((ks() - mr()) / qr() + 1) * qr() + mr();
    }
  }

  inline AvgPoolMicrokernelTester& mr(size_t mr) {
    assert(mr != 0);
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline AvgPoolMicrokernelTester& qr(size_t qr) {
    assert(qr != 0);
    this->qr_ = qr;
    return *this;
  }

  inline size_t qr() const {
    return this->qr_;
  }

  inline AvgPoolMicrokernelTester& kc(size_t kc) {
    assert(kc != 0);
    this->kc_ = kc;
    return *this;
  }

  inline size_t kc() const {
    return this->kc_;
  }

  inline AvgPoolMicrokernelTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return kc();
    } else {
      assert(this->x_stride_ >= kc());
      return this->x_stride_;
    }
  }

  inline AvgPoolMicrokernelTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return kc();
    } else {
      assert(this->y_stride_ >= kc());
      return this->y_stride_;
    }
  }

  inline AvgPoolMicrokernelTester& x_scale(float x_scale) {
    assert(x_scale > 0.0f);
    assert(std::isnormal(x_scale));
    this->x_scale_ = x_scale;
    return *this;
  }

  inline float x_scale() const {
    return this->x_scale_;
  }

  inline AvgPoolMicrokernelTester& x_zero_point(uint8_t x_zero_point) {
    this->x_zero_point_ = x_zero_point;
    return *this;
  }

  inline uint8_t x_zero_point() const {
    return this->x_zero_point_;
  }

  inline AvgPoolMicrokernelTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  inline float y_scale() const {
    return this->y_scale_;
  }

  inline AvgPoolMicrokernelTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  inline uint8_t y_zero_point() const {
    return this->y_zero_point_;
  }

  inline AvgPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline AvgPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_avgpool_up_ukernel_function avgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<const uint8_t*> indirect_x(packed_ks() + (n() - 1) * s() * kh());
    std::vector<uint8_t> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(uint8_t));

    std::vector<uint8_t> zero(kc() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y((n() - 1) * y_stride() + kc());
    std::vector<uint8_t> y_ref(n() * kc());
    std::vector<float> y_fp(n() * kc());
    std::vector<int32_t> y_acc(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      for (size_t i = 0; i < indirect_x.size(); i++) {
        indirect_x[i] = x.data() + i * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Prepare quantization parameters.
      xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(ks()),
            x_scale() / (y_scale() * float(ks())),
            y_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(ks()),
            x_scale() / (y_scale() * float(ks())),
            y_zero_point(), qmin(), qmax());
          break;
      }
      const xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(x_zero_point()) * int32_t(ks()),
          x_scale() / (y_scale() * float(ks())),
          y_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          int32_t acc = scalar_quantization_params.scalar.bias;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirect_x[i * s() * kh() + j][k];
          }
          y_acc[i * kc() + k] = acc;
          y_ref[i * kc() + k] = xnn_avgpool_quantize(acc, scalar_quantization_params);
          y_fp[i * kc() + k] = float(acc) * (x_scale() / (y_scale() * float(ks()))) + float(y_zero_point());
          y_fp[i * kc() + k] = std::min<float>(y_fp[i * kc() + k], float(qmax()));
          y_fp[i * kc() + k] = std::max<float>(y_fp[i * kc() + k], float(qmin()));
        }
      }

      // Call optimized micro-kernel.
      avgpool(n(), ks(), kc(),
        indirect_x.data(), zero.data(), y.data(),
        kh() * s() * sizeof(void*),
        (y_stride() - kc()) * sizeof(uint8_t),
        &quantization_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(uint32_t(y[i * y_stride() + k]), uint32_t(qmax()))
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_GE(uint32_t(y[i * y_stride() + k]), uint32_t(qmin()))
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_NEAR(float(int32_t(y[i * y_stride() + k])), y_fp[i * kc() + k], 0.5f)
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc()
            << ", acc = " << y_acc[i * kc() + k];
          ASSERT_EQ(uint32_t(y_ref[i * kc() + k]), uint32_t(y[i * y_stride() + k]))
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc()
            << ", acc = " << y_acc[i * kc() + k];
        }
      }
    }
  }

  void Test(xnn_q8_avgpool_mp_ukernel_function avgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<const uint8_t*> indirect_x(packed_ks() + (n() - 1) * s() * kh());
    std::vector<uint8_t> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buf(kc() + XNN_EXTRA_BYTES / sizeof(uint8_t));

    std::vector<uint8_t> zero(kc() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y((n() - 1) * y_stride() + kc());
    std::vector<uint8_t> y_ref(n() * kc());
    std::vector<float> y_fp(n() * kc());
    std::vector<int32_t> y_acc(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      for (size_t i = 0; i < indirect_x.size(); i++) {
        indirect_x[i] = x.data() + i * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Prepare quantization parameters.
      xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(ks()),
            x_scale() / (y_scale() * float(ks())),
            y_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(x_zero_point()) * int32_t(ks()),
            x_scale() / (y_scale() * float(ks())),
            y_zero_point(), qmin(), qmax());
          break;
      }
      const xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(x_zero_point()) * int32_t(ks()),
          x_scale() / (y_scale() * float(ks())),
          y_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          int32_t acc = scalar_quantization_params.scalar.bias;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirect_x[i * s() * kh() + j][k];
          }
          y_acc[i * kc() + k] = acc;
          y_ref[i * kc() + k] = xnn_avgpool_quantize(acc, scalar_quantization_params);
          y_fp[i * kc() + k] = float(acc) * (x_scale() / (y_scale() * float(ks()))) + float(y_zero_point());
          y_fp[i * kc() + k] = std::min<float>(y_fp[i * kc() + k], float(qmax()));
          y_fp[i * kc() + k] = std::max<float>(y_fp[i * kc() + k], float(qmin()));
        }
      }

      // Call optimized micro-kernel.
      avgpool(n(), ks(), kc(),
        indirect_x.data(), zero.data(), buf.data(), y.data(),
        (kh() * s() - (packed_ks() - qr())) * sizeof(void*),
        (y_stride() - kc()) * sizeof(uint8_t),
        &quantization_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(uint32_t(y[i * y_stride() + k]), uint32_t(qmax()))
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_GE(uint32_t(y[i * y_stride() + k]), uint32_t(qmin()))
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_NEAR(float(int32_t(y[i * y_stride() + k])), y_fp[i * kc() + k], 0.5f)
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc()
            << ", acc = " << y_acc[i * kc() + k];
          ASSERT_EQ(uint32_t(y_ref[i * kc() + k]), uint32_t(y[i * y_stride() + k]))
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc()
            << ", acc = " << y_acc[i * kc() + k];
        }
      }
    }
  }

  void Test(xnn_f32_avgpool_up_ukernel_function avgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<const float*> indirect_x(packed_ks() + (n() - 1) * s() * kh());
    std::vector<float> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(float));

    std::vector<float> zero(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y((n() - 1) * y_stride() + kc());
    std::vector<float> y_ref(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), std::nanf(""));

      for (size_t i = 0; i < indirect_x.size(); i++) {
        indirect_x[i] = x.data() + i * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          float acc = 0.0f;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirect_x[i * s() * kh() + j][k];
          }
          y_ref[i * kc() + k] = acc / float(ks());
        }
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

      // Prepare output parameters.
      xnn_f32_avgpool_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_avgpool_params(
            1.0f / float(ks()), y_min, y_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_avgpool_params(
            1.0f / float(ks()), y_min, y_max);
          break;
      }

      // Call optimized micro-kernel.
      avgpool(n(), ks(), kc(),
        indirect_x.data(), zero.data(), y.data(),
        kh() * s() * sizeof(void*),
        (y_stride() - kc()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(y[i * y_stride() + k], y_max)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_GE(y[i * y_stride() + k], y_min)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_NEAR(y[i * y_stride() + k], y_ref[i * kc() + k], std::abs(y_ref[i * kc() + k]) * 1.0e-6)
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
        }
      }
    }
  }

  void Test(xnn_f32_avgpool_mp_ukernel_function avgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<const float*> indirect_x(packed_ks() + (n() - 1) * s() * kh());
    std::vector<float> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> buf(kc() + XNN_EXTRA_BYTES / sizeof(float));

    std::vector<float> zero(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y((n() - 1) * y_stride() + kc());
    std::vector<float> y_ref(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), std::nanf(""));

      for (size_t i = 0; i < indirect_x.size(); i++) {
        indirect_x[i] = x.data() + i * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          float acc = 0.0f;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirect_x[i * s() * kh() + j][k];
          }
          y_ref[i * kc() + k] = acc / float(ks());
        }
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

      // Prepare output parameters.
      xnn_f32_avgpool_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_avgpool_params(
            1.0f / float(ks()), y_min, y_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_avgpool_params(
            1.0f / float(ks()), y_min, y_max);
          break;
      }

      // Call optimized micro-kernel.
      avgpool(n(), ks(), kc(),
        indirect_x.data(), zero.data(), buf.data(), y.data(),
        (kh() * s() - (packed_ks() - qr())) * sizeof(void*),
        (y_stride() - kc()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(y[i * y_stride() + k], y_max)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_GE(y[i * y_stride() + k], y_min)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_NEAR(y[i * y_stride() + k], y_ref[i * kc() + k], std::abs(y_ref[i * kc() + k]) * 1.0e-6)
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
        }
      }
    }
  }

  void Test(xnn_f32_pavgpool_up_ukernel_function pavgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f32mrng = std::bind(std::uniform_real_distribution<float>(0.1f, 0.5f), rng);

    std::vector<const float*> indirect_x(packed_ks() + (n() - 1) * s() * kh());
    std::vector<float> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(float));

    std::vector<float> zero(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> m(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y((n() - 1) * y_stride() + kc());
    std::vector<float> y_ref(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32irng));
      std::generate(m.begin(), m.end(), std::ref(f32mrng));
      std::fill(y.begin(), y.end(), std::nanf(""));

      for (size_t i = 0; i < indirect_x.size(); i++) {
        indirect_x[i] = x.data() + i * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          float acc = 0.0f;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirect_x[i * s() * kh() + j][k];
          }
          y_ref[i * kc() + k] = acc * m[i];
        }
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

      // Prepare output parameters.
      xnn_f32_output_params output_params = { };
      switch (variant) {
        case Variant::Native:
          output_params = xnn_init_f32_output_params(y_min, y_max);
          break;
        case Variant::Scalar:
          output_params = xnn_init_scalar_f32_output_params(y_min, y_max);
          break;
      }

      // Call optimized micro-kernel.
      pavgpool(n(), ks(), kc(),
        indirect_x.data(), zero.data(), m.data(), y.data(),
        kh() * s() * sizeof(void*),
        (y_stride() - kc()) * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(y[i * y_stride() + k], y_max)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_GE(y[i * y_stride() + k], y_min)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_NEAR(y[i * y_stride() + k], y_ref[i * kc() + k], std::abs(y_ref[i * kc() + k]) * 1.0e-6)
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
        }
      }
    }
  }

  void Test(xnn_f32_pavgpool_mp_ukernel_function pavgpool, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f32mrng = std::bind(std::uniform_real_distribution<float>(0.1f, 0.5f), rng);

    std::vector<const float*> indirect_x(packed_ks() + (n() - 1) * s() * kh());
    std::vector<float> x((indirect_x.size() - 1) * x_stride() + kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> buf(kc() + XNN_EXTRA_BYTES / sizeof(float));

    std::vector<float> zero(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> m(kc() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y((n() - 1) * y_stride() + kc());
    std::vector<float> y_ref(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32irng));
      std::generate(m.begin(), m.end(), std::ref(f32mrng));
      std::fill(y.begin(), y.end(), std::nanf(""));

      for (size_t i = 0; i < indirect_x.size(); i++) {
        indirect_x[i] = x.data() + i * x_stride();
      }
      std::shuffle(indirect_x.begin(), indirect_x.end(), rng);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          float acc = 0.0f;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirect_x[i * s() * kh() + j][k];
          }
          y_ref[i * kc() + k] = acc * m[i];
        }
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

      // Prepare output parameters.
      xnn_f32_output_params output_params = { };
      switch (variant) {
        case Variant::Native:
          output_params = xnn_init_f32_output_params(y_min, y_max);
          break;
        case Variant::Scalar:
          output_params = xnn_init_scalar_f32_output_params(y_min, y_max);
          break;
      }

      // Call optimized micro-kernel.
      pavgpool(n(), ks(), kc(),
        indirect_x.data(), zero.data(), m.data(), buf.data(), y.data(),
        (kh() * s() - (packed_ks() - qr())) * sizeof(void*),
        (y_stride() - kc()) * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(y[i * y_stride() + k], y_max)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_GE(y[i * y_stride() + k], y_min)
            << "at pixel " << i << ", channel " << k << ", n = " << n() << ", kc = " << kc();
          ASSERT_NEAR(y[i * y_stride() + k], y_ref[i * kc() + k], std::abs(y_ref[i * kc() + k]) * 1.0e-6)
            << "at pixel " << i << ", channel " << k << ", n = " << n()
            << ", ks = " << kh() << "x" << kw() << " (" << ks() << "), kc = " << kc();
        }
      }
    }
  }

 private:
  size_t n_{1};
  size_t s_{1};
  size_t kh_{1};
  size_t kw_{1};
  size_t mr_{1};
  size_t qr_{1};
  size_t kc_{1};
  size_t x_stride_{0};
  size_t y_stride_{0};
  float x_scale_{1.25f};
  float y_scale_{0.75f};
  uint8_t x_zero_point_{121};
  uint8_t y_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
