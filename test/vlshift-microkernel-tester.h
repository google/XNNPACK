// Copyright 2022 Google LLC
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
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>


class VLShiftMicrokernelTester {
 public:
  inline VLShiftMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  inline size_t batch() const {
    return this->batch_;
  }

  inline VLShiftMicrokernelTester& shift(uint32_t shift) {
    assert(shift < 32);
    this->shift_ = shift;
    return *this;
  }

  inline uint32_t shift() const {
    return this->shift_;
  }

  inline VLShiftMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VLShiftMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s16_vlshift_ukernel_function vlshift) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> x(batch() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y(batch() + (inplace() ? XNN_EXTRA_BYTES / sizeof(int16_t) : 0));
    std::vector<int16_t> y_ref(batch());
    const int16_t* x_data = inplace() ? y.data() : x.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i16rng));
      std::generate(y.begin(), y.end(), std::ref(i16rng));
      std::generate(y_ref.begin(), y_ref.end(), std::ref(i16rng));

      // Compute reference results.
      for (size_t n = 0; n < batch(); n++) {
        const uint16_t i = static_cast<uint16_t>(x_data[n]);
        uint16_t value = i << shift();
        y_ref[n] = reinterpret_cast<uint16_t>(value);
      }

      // Call optimized micro-kernel.
      vlshift(batch(), x_data, y.data(), shift());

      // Verify results.
      for (size_t n = 0; n < batch(); n++) {
        ASSERT_EQ(y[n], y_ref[n])
          << ", shift " << shift()
          << ", batch " << n << " / " << batch();
      }
    }
  }

 private:
  size_t batch_{1};
  uint32_t shift_{12};
  bool inplace_{false};
  size_t iterations_{15};
};
