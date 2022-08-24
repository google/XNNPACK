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
#include <xnnpack/microfnptr.h>
#include <xnnpack/math.h>


void xnn_cs16_fftr_reference(
    size_t samples,
    const int16_t* input,
    int16_t* output,
    const int16_t* twiddle) {

  assert(samples >= 2);
  assert(samples % 2 == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(twiddle != NULL);

  const int16_t* il = input;
  const int16_t* ir = input + samples * 2;
  int32_t vdcr = (int32_t) il[0];
  int32_t vdci = (int32_t) il[1];
  il += 2;
  vdcr = math_asr_s32(vdcr * 16383 + 16384, 15);
  vdci = math_asr_s32(vdci * 16383 + 16384, 15);

  int16_t* outl  = output;
  int16_t* outr = output + samples * 2;
  outl[0] = vdcr + vdci;
  outl[1] = 0;
  outl  += 2;
  outr[0] = vdcr - vdci;
  outr[1] = 0;

  samples >>= 1;

  do {
    int32_t vilr = il[0];
    int32_t vili = il[1];
    il += 2;
    ir -= 2;
    int32_t virr =  (int32_t) ir[0];
    int32_t viri = -(int32_t) ir[1];
    const int32_t vtwr = twiddle[0];
    const int32_t vtwi = twiddle[1];
    twiddle += 2;

    vilr =  math_asr_s32(vilr * 16383 + 16384, 15);
    vili =  math_asr_s32(vili * 16383 + 16384, 15);
    virr = math_asr_s32(virr * 16383 + 16384, 15);
    viri = math_asr_s32(viri * 16383 + 16384, 15);
    const int32_t vacc1r = vilr + virr;
    const int32_t vacc1i = vili + viri;
    const int32_t vacc2r = vilr - virr;
    const int32_t vacc2i = vili - viri;

    const int32_t twr = math_asr_s32(vacc2r * vtwr - vacc2i * vtwi + 16384, 15);
    const int32_t twi = math_asr_s32(vacc2r * vtwi + vacc2i * vtwr + 16384, 15);

    outl[0] = math_asr_s32(vacc1r + twr, 1);
    outl[1] = math_asr_s32(vacc1i + twi, 1);
    outl += 2;
    outr -= 2;
    outr[0] = math_asr_s32(vacc1r - twr, 1);
    outr[1] = math_asr_s32(twi - vacc1i, 1);

  } while (--samples != 0);
}

class FftrMicrokernelTester {
 public:
  inline FftrMicrokernelTester& samples(size_t samples) {
    assert(samples != 0);
    this->samples_ = samples;
    return *this;
  }

  inline size_t samples() const {
    return this->samples_;
  }

  inline FftrMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_cs16_fftr_ukernel_function fftr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));
    const size_t sample_size = samples() * 2 + 2;

    std::vector<int16_t> twiddle(samples());
    std::vector<int16_t> y(sample_size);
    std::vector<int16_t> y_ref(sample_size);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(twiddle.begin(), twiddle.end(), std::ref(i16rng));
      std::generate(y.begin(), y.end(), std::ref(i16rng));
      std::copy(y.begin(), y.end(), y_ref.begin());

      // Compute reference results.
      xnn_cs16_fftr_reference(samples(), y_ref.data(), y_ref.data(), twiddle.data());

      // Call optimized micro-kernel.
      fftr(samples(), y.data(), twiddle.data());

      // Verify results.
      for (size_t n = 0; n < sample_size; n++) {
        ASSERT_EQ(y[n], y_ref[n])
            << "at sample " << n << " / " << samples();
      }
    }
  }

 private:
  size_t samples_{256};
  size_t iterations_{15};
};
