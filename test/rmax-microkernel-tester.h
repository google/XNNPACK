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
#include <limits>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>


class RMaxMicrokernelTester {
 public:
  inline RMaxMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline RMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_rmax_ukernel_function rmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<uint16_t> x(n() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });

      // Compute reference results.
      float y_ref = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < n(); i++) {
        y_ref = std::max(y_ref, fp16_ieee_to_fp32_value(x[i]));
      }

      // Call optimized micro-kernel.
      uint16_t y = UINT16_C(0x7E00) /* NaN */;
      rmax(n() * sizeof(uint16_t), x.data(), &y);

      // Verify results.
      ASSERT_EQ(fp16_ieee_to_fp32_value(y), y_ref)
        << "batch " << n() << " y = " << y;
    }
  }

  void Test(xnn_f32_rmax_ukernel_function rmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> x(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      float y_ref = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < n(); i++) {
        y_ref = std::max(y_ref, x[i]);
      }

      // Call optimized micro-kernel.
      float y = std::nanf("");
      rmax(n() * sizeof(float), x.data(), &y);

      // Verify results.
      ASSERT_EQ(y_ref, y)
        << "batch " << n();
    }
  }

  void Test(xnn_u8_rmax_ukernel_function rmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> x(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return u8dist(rng); });

      // Compute reference results.
      uint8_t y_ref = 0;
      for (size_t i = 0; i < n(); i++) {
        y_ref = std::max(y_ref, x[i]);
      }

      // Call optimized micro-kernel.
      uint8_t y = u8dist(rng);
      rmax(n() * sizeof(uint8_t), x.data(), &y);

      // Verify results.
      ASSERT_EQ(int32_t(y_ref), int32_t(y))
        << "batch " << n();
    }
  }

 private:
  size_t n_{1};
  size_t iterations_{15};
};
