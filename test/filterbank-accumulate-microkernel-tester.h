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
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

class FilterbankAccumulateMicrokernelTester {
 public:
  FilterbankAccumulateMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const {
    return this->rows_;
  }

  FilterbankAccumulateMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u32_filterbank_accumulate_ukernel_fn filterbank_accumulate) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(1, 10);
    std::uniform_int_distribution<uint16_t> u16dist;
    std::uniform_int_distribution<uint32_t> u32dist;

    std::vector<uint8_t> filterbank_widths(rows() + 1);
    std::vector<uint64_t> output(rows());
    std::vector<uint64_t> output_ref(rows());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(filterbank_widths.begin(), filterbank_widths.end(), [&] { return u8dist(rng); });
      const size_t num_channels = std::accumulate(filterbank_widths.cbegin(), filterbank_widths.cend(), 0);

      std::vector<uint32_t> input(num_channels);
      std::vector<uint16_t> weights(num_channels * 2);
      std::generate(input.begin(), input.end(), [&] { return u32dist(rng); });
      std::generate(weights.begin(), weights.end(), [&] { return u16dist(rng); });
      std::fill(output.begin(), output.end(), UINT64_C(0xCAFEB0BADEADBEAF));

      uint64_t weight_accumulator = 0;
      uint64_t unweight_accumulator = 0;
      size_t i = 0;
      for (size_t m = 0; m <= rows(); m++) {
        const size_t weight_width = filterbank_widths[m];
        for (size_t n = 0; n < weight_width; n++) {
          weight_accumulator += uint64_t(input[i]) * uint64_t(weights[i * 2]);
          unweight_accumulator += uint64_t(input[i]) * uint64_t(weights[i * 2 + 1]);
          i += 1;
        }
        if (m != 0) {
          output_ref[m - 1] = weight_accumulator;
        }
        weight_accumulator = unweight_accumulator;
        unweight_accumulator = 0;
      }

      // Call optimized micro-kernel.
      filterbank_accumulate(rows(), input.data(), filterbank_widths.data(), weights.data(), output.data());

      // Verify results.
      for (size_t m = 0; m < rows(); m++) {
        EXPECT_EQ(output[m], output_ref[m])
            << "at row " << m << " / " << rows();
      }
    }
  }

 private:
  size_t rows_{1};
  size_t iterations_{15};
};
