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
  inline VSquareAbsMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  inline size_t batch() const {
    return this->batch_;
  }

  inline VSquareAbsMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_cs16_vsquareabs_ukernel_fn vsquareabs) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> input(batch() * 2 + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<uint32_t> output(batch());
    std::vector<uint32_t> output_ref(batch());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i16rng));
      std::fill(output.begin(), output.end(), UINT32_C(0xDEADBEEF));

      // Compute reference results.
      for (size_t n = 0; n < batch(); n++) {
        const int32_t r = static_cast<int32_t>(input[n * 2]);
        const int32_t i = static_cast<int32_t>(input[n * 2 + 1]);
        output_ref[n] = static_cast<uint32_t>(r * r + i * i);
      }

      // Call optimized micro-kernel.
      vsquareabs(batch() * sizeof(int16_t) * 2, input.data(), output.data());

      // Verify results.
      for (size_t n = 0; n < batch(); n++) {
        EXPECT_EQ(output[n], output_ref[n])
          << ", batch " << n << " / " << batch();
      }
    }
  }

 private:
  size_t batch_{1};
  size_t iterations_{15};
};
