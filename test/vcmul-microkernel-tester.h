// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class VCMulMicrokernelTester {
 public:
  VCMulMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  VCMulMicrokernelTester& inplace_a(bool inplace_a) {
    this->inplace_a_ = inplace_a;
    return *this;
  }

  bool inplace_a() const {
    return this->inplace_a_;
  }

  VCMulMicrokernelTester& inplace_b(bool inplace_b) {
    this->inplace_b_ = inplace_b;
    return *this;
  }

  bool inplace_b() const {
    return this->inplace_b_;
  }

  VCMulMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_vbinary_ukernel_fn vcmul, xnn_init_f16_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32rdist(1.0f, 10.0f);
    std::uniform_real_distribution<float> f32idist(0.01f, 0.1f);

    std::vector<uint16_t> a(2 * batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> b(2 * batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(2 * batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(2 * batch_size());
    std::fill(a.begin(), a.end(), UINT16_C(0x7E00) /* NaN */);
    std::fill(b.begin(), b.end(), UINT16_C(0x7E00) /* NaN */);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(a.begin(), batch_size(), [&]() { return fp16_ieee_from_fp32_value(f32rdist(rng)); });
      std::generate_n(a.begin() + batch_size(), batch_size(), [&]() { return fp16_ieee_from_fp32_value(f32idist(rng)); });
      std::generate_n(b.begin(), batch_size(), [&]() { return fp16_ieee_from_fp32_value(f32rdist(rng)); });
      std::generate_n(b.begin() + batch_size(), batch_size(), [&]() { return fp16_ieee_from_fp32_value(f32idist(rng)); });
      if (inplace_a()) {
        std::copy(a.cbegin(), a.cend(), y.begin());
      } else if (inplace_b()) {
        std::copy(b.cbegin(), b.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* a_data = inplace_a() ? y.data() : a.data();
      const uint16_t* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        float a0 = fp16_ieee_to_fp32_value(a_data[i]);
        float b0 = fp16_ieee_to_fp32_value(b_data[i]);
        float a1 = fp16_ieee_to_fp32_value(a_data[i + batch_size()]);
        float b1 = fp16_ieee_to_fp32_value(b_data[i + batch_size()]);
        y_ref[i] = a0 * b0 - a1 * b1;
        y_ref[i + batch_size()] = a0 * b1 + a1 * b0;
      }

      // Prepare parameters.
      xnn_f16_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vcmul(batch_size() * sizeof(uint16_t), a_data, b_data, y.data(), init_params != nullptr ? &params : nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        const float tolerance = std::abs(y_ref[i]) * 1.0e-2f;
        EXPECT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i], tolerance)
          << "at " << i << " / " << batch_size();
      }
    }
  }

  void Test(xnn_f32_vbinary_ukernel_fn vcmul, xnn_init_f32_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32rdist(1.0f, 10.0f);
    std::uniform_real_distribution<float> f32idist(0.01f, 0.1f);

    std::vector<float> a(2 * batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(2 * batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(2 * batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<double> y_ref(2 * batch_size());
    std::fill(a.begin(), a.end(), std::nanf(""));
    std::fill(b.begin(), b.end(), std::nanf(""));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(a.begin(), batch_size(), [&]() { return f32rdist(rng); });
      std::generate_n(a.begin() + batch_size(), batch_size(), [&]() { return f32idist(rng); });
      std::generate_n(b.begin(), batch_size(), [&]() { return f32rdist(rng); });
      std::generate_n(b.begin() + batch_size(), batch_size(), [&]() { return f32idist(rng); });
      if (inplace_a()) {
        std::copy(a.cbegin(), a.cend(), y.begin());
      } else if (inplace_b()) {
        std::copy(b.cbegin(), b.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* a_data = inplace_a() ? y.data() : a.data();
      const float* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = double(a_data[i]) * double(b_data[i]) - double(a_data[i + batch_size()]) * double(b_data[i + batch_size()]);
        y_ref[i + batch_size()] = double(a_data[i]) * double(b_data[i + batch_size()]) + double(a_data[i + batch_size()]) * double(b_data[i]);
      }

      // Prepare parameters.
      xnn_f32_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vcmul(batch_size() * sizeof(float), a_data, b_data, y.data(), init_params != nullptr ? &params : nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-4f)
          << "at " << i << " / " << batch_size();
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_a_{false};
  bool inplace_b_{false};
  size_t iterations_{15};
};
