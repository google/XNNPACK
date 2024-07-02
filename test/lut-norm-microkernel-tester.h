// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class LUTNormMicrokernelTester {
 public:
  LUTNormMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  size_t n() const {
    return this->n_;
  }

  LUTNormMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  LUTNormMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u8_lut32norm_ukernel_fn lutnorm) const {
    xnnpack::ReplicableRandomDevice rng;
    auto u8rng = [&rng]() {
      return std::uniform_int_distribution<uint32_t>(
          0, std::numeric_limits<uint8_t>::max())(rng);
    };
    auto u32rng = [&]() {
      return std::uniform_int_distribution<uint32_t>(
          1, std::numeric_limits<uint32_t>::max() / (257 * n()))(rng);
    };

    std::vector<uint8_t> x(n());
    std::vector<uint32_t> t(256);
    std::vector<uint8_t> y(n());
    std::vector<float> y_ref(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::generate(t.begin(), t.end(), std::ref(u32rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      uint32_t sum = 0;
      for (size_t i = 0; i < n(); i++) {
        sum += t[x_data[i]];
      }
      for (size_t i = 0; i < n(); i++) {
        y_ref[i] = 256.0f * float(t[x_data[i]]) / float(sum);
        y_ref[i] = std::min(y_ref[i], 255.0f);
      }

      // Call optimized micro-kernel.
      lutnorm(n(), x_data, t.data(), y.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        EXPECT_NEAR(y_ref[i], float(y[i]), 0.5f)
          << "at position " << i << ", n = " << n() << ", sum = " << sum;
      }
    }
  }

 private:
  size_t n_{1};
  bool inplace_{false};
  size_t iterations_{15};
};
