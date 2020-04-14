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


class LUTMicrokernelTester {
 public:
  inline LUTMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline LUTMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline LUTMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x8_lut_ukernel_function lut) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> x(n());
    std::vector<uint8_t> t(256);
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::generate(t.begin(), t.end(), std::ref(u8rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = t[x_data[i]];
      }

      // Call optimized micro-kernel.
      lut(n(), x_data, t.data(), y.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at position " << i << ", n = " << n();
      }
    }
  }

 private:
  size_t n_{1};
  bool inplace_{false};
  size_t iterations_{15};
};
