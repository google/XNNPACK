// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class RMaxAbsMicrokernelTester {
 public:

  RMaxAbsMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  size_t batch() const {
    return this->batch_;
  }

  RMaxAbsMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s16_rmaxabs_ukernel_fn rmaxabs) const {
    xnnpack::ReplicableRandomDevice rng;
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> input(batch() + XNN_EXTRA_BYTES / sizeof(int16_t));

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(i16rng));

      // Compute reference results.
      int32_t output_ref = 0;
      for (size_t n = 0; n < batch(); n++) {
        const int32_t input_value = static_cast<int32_t>(input[n]);
        const int32_t abs_value = std::abs(input_value);
        output_ref = std::max(output_ref, abs_value);
      }

      // Call optimized micro-kernel.
      uint16_t output = UINT16_C(0xDEAD);
      rmaxabs(batch() * sizeof(int16_t), input.data(), &output);

      // Verify results.
      ASSERT_EQ(static_cast<int32_t>(output), output_ref)
        << "batch " << batch();
    }
  }

 private:
  size_t batch_{1};
  size_t iterations_{15};
};
