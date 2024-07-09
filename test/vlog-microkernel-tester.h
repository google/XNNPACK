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
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

extern "C" XNN_INTERNAL const uint16_t xnn_table_vlog[129];

class VLogMicrokernelTester {
 public:
  VLogMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  size_t batch() const {
    return this->batch_;
  }

  VLogMicrokernelTester& input_lshift(uint32_t input_lshift) {
    assert(input_lshift < 32);
    this->input_lshift_ = input_lshift;
    return *this;
  }

  uint32_t input_lshift() const {
    return this->input_lshift_;
  }

  VLogMicrokernelTester& output_scale(uint32_t output_scale) {
    this->output_scale_ = output_scale;
    return *this;
  }

  uint32_t output_scale() const {
    return this->output_scale_;
  }

  VLogMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  VLogMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u32_vlog_ukernel_fn vlog) const {
    xnnpack::ReplicableRandomDevice rng;
    auto i16rng = std::bind(std::uniform_int_distribution<uint16_t>(), std::ref(rng));
    auto i32rng = std::bind(std::uniform_int_distribution<uint32_t>(), std::ref(rng));

    std::vector<uint32_t> x(batch() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint16_t> y(batch() * (inplace() ? sizeof(uint32_t) / sizeof(uint16_t) : 1) + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint16_t> y_ref(batch());
    const uint32_t* x_data = inplace() ? reinterpret_cast<const uint32_t*>(y.data()) : x.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i32rng));
      std::generate(y.begin(), y.end(), std::ref(i16rng));
      std::generate(y_ref.begin(), y_ref.end(), std::ref(i16rng));

      // Compute reference results.
      for (size_t n = 0; n < batch(); n++) {
        const uint32_t x_value = x_data[n];
        const uint32_t scaled = x_value << input_lshift();
        uint32_t log_value = 0;
        if (scaled != 0) {
          const uint32_t out_scale = output_scale();

          const int log_scale = 65536;
          const int log_scale_log2 = 16;
          const int log_coeff = 45426;
          const uint32_t log2x = math_clz_nonzero_u32(scaled) ^ 31;  // log2 of scaled
          assert(log2x < 32);

          // Number of segments in the log lookup table. The table will be log_segments+1
          // in length (with some padding).
          const int log_segments_log2 = 7;

          // Part 1
          uint32_t frac = scaled - (UINT32_C(1) << log2x);

          // Shift the fractional part into msb of 16 bits
          frac =  XNN_UNPREDICTABLE(log2x < log_scale_log2) ?
              (frac << (log_scale_log2 - log2x)) :
              (frac >> (log2x - log_scale_log2));

          // Part 2
          const uint32_t base_seg = frac >> (log_scale_log2 - log_segments_log2);
          const uint32_t seg_unit = (UINT32_C(1) << log_scale_log2) >> log_segments_log2;

          assert(128 == (1 << log_segments_log2));
          assert(base_seg < (1 << log_segments_log2));

          const uint32_t c0 = xnn_table_vlog[base_seg];
          const uint32_t c1 = xnn_table_vlog[base_seg + 1];
          const uint32_t seg_base = seg_unit * base_seg;
          const uint32_t rel_pos = ((c1 - c0) * (frac - seg_base)) >> log_scale_log2;
          const uint32_t fraction =  frac + c0 + rel_pos;

          const uint32_t log2 = (log2x << log_scale_log2) + fraction;
          const uint32_t round = log_scale / 2;
          const uint32_t loge = (((uint64_t) log_coeff) * log2 + round) >> log_scale_log2;

          // Finally scale to our output scale
          log_value = (out_scale * loge + round) >> log_scale_log2;
        }

        const uint32_t vout = math_min_u32(log_value, (uint32_t) INT16_MAX);
        y_ref[n] = vout;
      }

      // Call optimized micro-kernel.
      vlog(batch(), x_data, input_lshift(), output_scale(), y.data());

      // Verify results.
      for (size_t n = 0; n < batch(); n++) {
        EXPECT_EQ(y[n], y_ref[n])
          << ", input_lshift " << input_lshift()
          << ", output_scale " << output_scale()
          << ", batch " << n << " / " << batch();
      }
    }
  }

 private:
  size_t batch_{1};
  uint32_t input_lshift_{4};
  uint32_t output_scale_{16};
  bool inplace_{false};
  size_t iterations_{15};
};
