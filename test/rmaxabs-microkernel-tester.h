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


class RMaxAbsMicrokernelTester {
 public:

  inline RMaxAbsMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline RMaxAbsMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s16_rmaxabs_ukernel_function rmaxabs) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> x(channels() + XNN_EXTRA_BYTES / sizeof(int16_t));

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i16rng));
      int32_t y_ref = 0;

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        const int32_t i = static_cast<int32_t>(x[c]);
        const int32_t abs = std::abs(i);
        y_ref = std::max(y_ref, abs);
      }

      // Call optimized micro-kernel.
      uint16_t y = UINT16_C(0xDEAD);
      rmaxabs(channels(), x.data(), &y);

      // Verify results.
      ASSERT_EQ(static_cast<int32_t>(y), y_ref);
    }
  }

 private:
  size_t channels_{1};
  size_t iterations_{15};
};
