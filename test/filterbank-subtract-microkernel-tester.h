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
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microfnptr.h>


class FilterbankSubtractMicrokernelTester {
 public:

  inline FilterbankSubtractMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  inline size_t batch() const {
    return this->batch_;
  }

  inline FilterbankSubtractMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline FilterbankSubtractMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u32_filterbank_subtract_ukernel_fn filterbank_subtract) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), std::ref(rng));
    const uint32_t smoothing = 655;
    const uint32_t alternate_smoothing = 655;
    const uint32_t one_minus_smoothing = 15729;
    const uint32_t alternate_one_minus_smoothing = 15729;
    const uint32_t min_signal_remaining = 819;
    const uint32_t smoothing_bits = 0;
    const uint32_t spectral_subtraction_bits = 14;

    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> x(batch() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> noise(batch() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> noise_ref(batch() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> y(batch() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint32_t) : 0));
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> y_ref(batch());
    const uint32_t* x_data = inplace() ? y.data() : x.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u32rng));
      std::iota(noise.begin(), noise.end(), 0);
      std::iota(noise_ref.begin(), noise_ref.end(), 0);
      std::generate(y.begin(), y.end(), std::ref(u32rng));
      std::generate(y_ref.begin(), y_ref.end(), std::ref(u32rng));

      for (size_t n = 0; n < batch(); n += 2) {
        const uint32_t vinput0 = x_data[n + 0];
        const uint32_t vinput1 = x_data[n + 1];

        uint32_t vnoise_estimate0 = noise_ref[n + 0];
        uint32_t vnoise_estimate1 = noise_ref[n + 1];

        // Scale up signa for smoothing filter computation.
        const uint32_t vsignal_scaled_up0 = vinput0 << smoothing_bits;
        const uint32_t vsignal_scaled_up1 = vinput1 << smoothing_bits;

        vnoise_estimate0 = (((uint64_t) (vsignal_scaled_up0) * smoothing) +
                            ((uint64_t) (vnoise_estimate0) * one_minus_smoothing)) >> spectral_subtraction_bits;
        vnoise_estimate1 = (((uint64_t) (vsignal_scaled_up1) * alternate_smoothing) +
                            ((uint64_t) (vnoise_estimate1) * alternate_one_minus_smoothing)) >> spectral_subtraction_bits;

        noise_ref[n + 0] = vnoise_estimate0;
        noise_ref[n + 1] = vnoise_estimate1;

        // Make sure that we can't get a negative value for the signal - estimate.
        const uint32_t estimate_scaled_up0 = std::min(vnoise_estimate0, vsignal_scaled_up0);
        const uint32_t estimate_scaled_up1 = std::min(vnoise_estimate1, vsignal_scaled_up1);
        const uint32_t vsubtracted0 = (vsignal_scaled_up0 - estimate_scaled_up0) >> smoothing_bits;
        const uint32_t vsubtracted1 = (vsignal_scaled_up1 - estimate_scaled_up1) >> smoothing_bits;

        const uint32_t vfloor0 = ((uint64_t) (vinput0) * min_signal_remaining) >> spectral_subtraction_bits;
        const uint32_t vfloor1 = ((uint64_t) (vinput1) * min_signal_remaining) >> spectral_subtraction_bits;
        const uint32_t vout0 = std::max(vsubtracted0, vfloor0);
        const uint32_t vout1 = std::max(vsubtracted1, vfloor1);

        y_ref[n + 0] = vout0;
        y_ref[n + 1] = vout1;
      }

      // Call optimized micro-kernel.
      filterbank_subtract(batch(), x_data,
          smoothing, alternate_smoothing, one_minus_smoothing, alternate_one_minus_smoothing,
          min_signal_remaining, smoothing_bits, spectral_subtraction_bits,
          noise.data(), y.data());

      // Verify results.
      for (size_t n = 0; n < batch(); n++) {
        EXPECT_EQ(y[n], y_ref[n])
            << "at n " << n << " / " << batch();
        EXPECT_EQ(noise[n], noise_ref[n])
            << "at n " << n << " / " << batch();
      }
    }
  }

 private:
  size_t batch_{48};
  bool inplace_{false};
  size_t iterations_{15};
};
