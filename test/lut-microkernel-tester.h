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
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>


class LUTMicrokernelTester {
 public:
  inline LUTMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
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

  void Test(xnn_x8_lut_ukernel_fn lut) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(
      std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

    std::vector<uint8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    XNN_ALIGN(64) std::array<uint8_t, 256> t;
    std::vector<uint8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
    std::vector<uint8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::generate(t.begin(), t.end(), std::ref(u8rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* x_data = x.data();
      if (inplace()) {
        std::copy(y.cbegin(), y.cend(), x.begin());
        x_data = y.data();
      }

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = t[x_data[i]];
      }

      // Call optimized micro-kernel.
      lut(batch_size(), x_data, y.data(), t.data());

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at position " << i << " / " << batch_size()
          << ", input " << uint32_t(x[i]);
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_{false};
  size_t iterations_{15};
};
