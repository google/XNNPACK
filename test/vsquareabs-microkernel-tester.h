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


class VSquareAbsMicrokernelTester {
 public:
  inline VSquareAbsMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_;
  }

  inline VSquareAbsMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_cs16_vsquareabs_ukernel_function vsquareabs) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> x(batch_size() * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<uint32_t> y(batch_size());
    std::vector<uint32_t> y_ref(batch_size());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i16rng));
      std::fill(y.begin(), y.end(), INT32_C(0x12345678));

      // Compute reference results.
      for (size_t n = 0; n < batch_size(); n++) {
        const int16_t r = x[n * 2];
        const int16_t i = x[n * 2 + 1];
        uint32_t rsquare = static_cast<uint32_t>(static_cast<int32_t>(r) * static_cast<int32_t>(r));
        uint32_t isquare = static_cast<uint32_t>(static_cast<int32_t>(i) * static_cast<int32_t>(i));
        uint32_t value = rsquare + isquare;
        y_ref[n] = value;
      }

      // Call optimized micro-kernel.
      vsquareabs(batch_size(), x.data(), y.data());

      // Verify results.
      for (size_t n = 0; n < batch_size(); n++) {
        ASSERT_EQ(y[n], y_ref[n])
          << ", batch " << n << " / " << batch_size();
      }
    }
  }

 private:
  size_t batch_{1};
  size_t iterations_{15};
};
