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
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/params.h>


class VRShiftMicrokernelTester {
 public:
  inline VRShiftMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline VRShiftMicrokernelTester& shift(uint32_t shift) {
    assert(shift < 32);
    this->shift_ = shift;
    return *this;
  }

  inline uint32_t shift() const {
    return this->shift_;
  }

  inline VRShiftMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VRShiftMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s16_vrshift_ukernel_function vrshift) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> x(channels() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y(channels() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y_ref(channels());
    const int16_t* x_data = inplace() ? y.data() : x.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i16rng));
      std::fill(y.begin(), y.end(), INT32_C(0x12345678));

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        const uint16_t i = static_cast<uint16_t>(x_data[c]);
        uint16_t value = i << shift();
        y_ref[c] = reinterpret_cast<uint16_t>(value);
      }

      // Call optimized micro-kernel.
      vrshift(channels(), x_data, shift(), y.data());

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_EQ(y[c], y_ref[c])
          << "shift " << shift()
          << ", channel " << c << " / " << channels();
      }
    }
  }

 private:
  size_t channels_{1};
  uint32_t shift_{12};
  bool inplace_{false};
  size_t iterations_{15};
};
