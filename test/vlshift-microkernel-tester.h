// Copyright 2022 Google LLC
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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class VLShiftMicrokernelTester {
 public:
  VLShiftMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  size_t batch() const {
    return this->batch_;
  }

  VLShiftMicrokernelTester& shift(uint32_t shift) {
    assert(shift < 32);
    this->shift_ = shift;
    return *this;
  }

  uint32_t shift() const {
    return this->shift_;
  }

  VLShiftMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  VLShiftMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_i16_vlshift_ukernel_fn vlshift) const {
    xnnpack::ReplicableRandomDevice rng;
    auto u16rng = std::bind(std::uniform_int_distribution<uint16_t>(), std::ref(rng));

    std::vector<uint16_t> input(batch() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(batch() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<uint16_t> output_ref(batch());
    const uint16_t* input_data = inplace() ? output.data() : input.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u16rng));
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));

      // Compute reference results.
      for (size_t n = 0; n < batch(); n++) {
        uint16_t value = input_data[n];
        value <<= shift();
        output_ref[n] = value;
      }

      // Call optimized micro-kernel.
      vlshift(batch(), input_data, output.data(), shift());

      // Verify results.
      for (size_t n = 0; n < batch(); n++) {
        EXPECT_EQ(output[n], output_ref[n])
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
