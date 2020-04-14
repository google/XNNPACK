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

#include <xnnpack/params.h>


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

  void Test(xnn_u8_rmax_ukernel_function rmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> x(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));

      // Compute reference results.
      uint8_t y_ref = 0;
      for (size_t i = 0; i < n(); i++) {
        y_ref = std::max(y_ref, x[i]);
      }

      // Call optimized micro-kernel.
      uint8_t y = u8rng();
      rmax(n() * sizeof(uint8_t), x.data(), &y);

      // Verify results.
      ASSERT_EQ(y_ref, y) << "n = " << n();
    }
  }

  void Test(xnn_f32_rmax_ukernel_function rmax) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> x(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));

      // Compute reference results.
      float y_ref = 0;
      for (size_t i = 0; i < n(); i++) {
        y_ref = std::max(y_ref, x[i]);
      }

      // Call optimized micro-kernel.
      float y = std::nanf("");
      rmax(n() * sizeof(float), x.data(), &y);

      // Verify results.
      ASSERT_EQ(y_ref, y) << "n = " << n();
    }
  }

 private:
  size_t n_{1};
  size_t iterations_{15};
};
