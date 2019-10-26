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
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class VMulCAddCMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline VMulCAddCMicrokernelTester& cr(size_t cr) {
    this->cr_ = cr;
    return *this;
  }

  inline size_t cr() const {
    return this->cr_;
  }

  inline VMulCAddCMicrokernelTester& c(size_t c) {
    assert(c != 0);
    this->c_ = c;
    return *this;
  }

  inline size_t c() const {
    return this->c_;
  }

  inline size_t packed_c() const {
    return c() % cr() == 0 ? c() : (c() / cr() + 1) * cr();
  }

  inline VMulCAddCMicrokernelTester& m(size_t m) {
    assert(m != 0);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline VMulCAddCMicrokernelTester& x_stride(size_t x_stride) {
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    return this->x_stride_ == 0 ? c() : this->x_stride_;
  }

  inline VMulCAddCMicrokernelTester& y_stride(size_t y_stride) {
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    return this->y_stride_ == 0 ? c() : this->y_stride_;
  }

  inline VMulCAddCMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VMulCAddCMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VMulCAddCMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VMulCAddCMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vmulcaddc_ukernel_function vmulcaddc, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    if (inplace()) {
      ASSERT_EQ(x_stride(), y_stride());
    }

    std::vector<float> x((m() - 1) * x_stride() + c() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> scale(c());
    std::vector<float> bias(c());
    std::vector<float, AlignedAllocator<float, 32>> packed_w(packed_c() * 2);
    std::vector<float> y((m() - 1) * y_stride() + c() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(m() * c());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      std::fill(packed_w.begin(), packed_w.end(), nanf(""));
      xnn_pack_f32_vmulcaddc_w(c(), cr(),
        scale.data(), bias.data(), packed_w.data());

      // Compute reference results.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < c(); j++) {
          y_ref[i * c() + j] = x_data[i * x_stride() + j] * scale[j] + bias[j];
        }
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      const float y_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      for (float& y_value : y_ref) {
        y_value = std::max<float>(std::min<float>(y_value, y_max), y_min);
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
      vmulcaddc(m(), c() * sizeof(float),
        x_data, x_stride() * sizeof(float),
        packed_w.data(),
        y.data(), y_stride() * sizeof(float),
        &output_params);

      // Verify results.
      for (size_t i = 0; i < m(); i++) {
        for (size_t j = 0; j < c(); j++) {
          ASSERT_NEAR(y[i * y_stride() + j], y_ref[i * c() + j], std::abs(y_ref[i * c() + j]) * 1.0e-6f)
            << "at pixel " << i << " / " << m()
            << ", channel = " << j << " / " << c();
        }
      }
    }
  }

 private:
  size_t cr_{1};
  size_t c_{1};
  size_t m_{1};
  size_t x_stride_{0};
  size_t y_stride_{0};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
