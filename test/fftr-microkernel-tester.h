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


static const int16_t xnn_reference_table_fftr_twiddle[256] = {
  -402,-32765,   -804,-32757,  -1206,-32745,  -1608,-32728,
  -2009,-32705,  -2410,-32678,  -2811,-32646,  -3212,-32609,
  -3612,-32567,  -4011,-32521,  -4410,-32469,  -4808,-32412,
  -5205,-32351,  -5602,-32285,  -5998,-32213,  -6393,-32137,
  -6786,-32057,  -7179,-31971,  -7571,-31880,  -7962,-31785,
  -8351,-31685,  -8739,-31580,  -9126,-31470,  -9512,-31356,
  -9896,-31237, -10278,-31113, -10659,-30985, -11039,-30852,
  -11417,-30714, -11793,-30571, -12167,-30424, -12539,-30273,
  -12910,-30117, -13279,-29956, -13645,-29791, -14010,-29621,
  -14372,-29447, -14732,-29268, -15090,-29085, -15446,-28898,
  -15800,-28706, -16151,-28510, -16499,-28310, -16846,-28105,
  -17189,-27896, -17530,-27683, -17869,-27466, -18204,-27245,
  -18537,-27019, -18868,-26790, -19195,-26556, -19519,-26319,
  -19841,-26077, -20159,-25832, -20475,-25582, -20787,-25329,
  -21096,-25072, -21403,-24811, -21705,-24547, -22005,-24279,
  -22301,-24007, -22594,-23731, -22884,-23452, -23170,-23170,
  -23452,-22884, -23731,-22594, -24007,-22301, -24279,-22005,
  -24547,-21705, -24811,-21403, -25072,-21096, -25329,-20787,
  -25582,-20475, -25832,-20159, -26077,-19841, -26319,-19519,
  -26556,-19195, -26790,-18868, -27019,-18537, -27245,-18204,
  -27466,-17869, -27683,-17530, -27896,-17189, -28105,-16846,
  -28310,-16499, -28510,-16151, -28706,-15800, -28898,-15446,
  -29085,-15090, -29268,-14732, -29447,-14372, -29621,-14010,
  -29791,-13645, -29956,-13279, -30117,-12910, -30273,-12539,
  -30424,-12167, -30571,-11793, -30714,-11417, -30852,-11039,
  -30985,-10659, -31113,-10278, -31237, -9896, -31356, -9512,
  -31470, -9126, -31580, -8739, -31685, -8351, -31785, -7962,
  -31880, -7571, -31971, -7179, -32057, -6786, -32137, -6393,
  -32213, -5998, -32285, -5602, -32351, -5205, -32412, -4808,
  -32469, -4410, -32521, -4011, -32567, -3612, -32609, -3212,
  -32646, -2811, -32678, -2410, -32705, -2009, -32728, -1608,
  -32745, -1206, -32757,  -804, -32765,  -402, -32767,     0,
};

static void xnn_cs16_fftr_reference(
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
    int32_t vilr = (int32_t) il[0];
    int32_t vili = (int32_t) il[1];
    il += 2;
    ir -= 2;
    int32_t virr = (int32_t) ir[0];
    int32_t viri = (int32_t) ir[1];
    const int32_t vtwr = twiddle[0];
    const int32_t vtwi = twiddle[1];
    twiddle += 2;

    vilr = math_asr_s32(vilr * 16383 + 16384, 15);
    vili = math_asr_s32(vili * 16383 + 16384, 15);
    virr = math_asr_s32(virr * 16383 + 16384, 15);
    viri = math_asr_s32(viri * 16383 + 16384, 15);
    const int16_t vacc1r = vilr + virr;
    const int16_t vacc1i = vili - viri;
    const int16_t vacc2r = vilr - virr;
    const int16_t vacc2i = vili + viri;

    const int32_t vaccr = math_asr_s32(vacc2r * vtwr - vacc2i * vtwi + 16384, 15);
    const int32_t vacci = math_asr_s32(vacc2r * vtwi + vacc2i * vtwr + 16384, 15);

    outl[0] = math_asr_s32(vacc1r + vaccr, 1);
    outl[1] = math_asr_s32(vacc1i + vacci, 1);
    outl += 2;
    outr -= 2;
    outr[0] = math_asr_s32(vacc1r - vaccr, 1);
    outr[1] = math_asr_s32(vacci - vacc1i, 1);

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

  void Test(xnn_cs16_fftr_ukernel_fn fftr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));
    const size_t sample_size = samples() * 2 + 2;

    std::vector<int16_t> y(sample_size);
    std::vector<int16_t> y_ref(sample_size);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(y.begin(), y.end(), std::ref(i16rng));
      std::copy(y.begin(), y.end(), y_ref.begin());

      // Compute reference results.
      xnn_cs16_fftr_reference(samples(), y_ref.data(), y_ref.data(), xnn_reference_table_fftr_twiddle);

      // Call optimized micro-kernel.
      fftr(samples(), y.data(), xnn_reference_table_fftr_twiddle);

      // Verify results.
      for (size_t n = 0; n < sample_size; n++) {
        EXPECT_EQ(y[n], y_ref[n])
            << "at sample " << n << " / " << sample_size;
      }
    }
  }

 private:
  size_t samples_{256};
  size_t iterations_{15};
};
