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


class FilterbankAccumulateMicrokernelTester {
 public:
  inline FilterbankAccumulateMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline FilterbankAccumulateMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  inline size_t batch() const {
    return this->batch_;
  }

  inline FilterbankAccumulateMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u32_filterbank_accumulate_ukernel_function filterbank_accumulate) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u16rng = std::bind(std::uniform_int_distribution<uint16_t>(), std::ref(rng));
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), std::ref(rng));

    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> input(batch() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> weight_widths(rows());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> weights(batch() * 2 + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> output(rows());
    std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> output_ref(rows());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::fill(weight_widths.begin(), weight_widths.end(), rows());
      std::generate(weights.begin(), weights.end(), std::ref(u16rng));
      std::iota(output.begin(), output.end(), 0);
      std::iota(output_ref.begin(), output_ref.end(), 1);

      uint64_t weight_accumulator = 0;
      uint64_t unweight_accumulator = 0;
      size_t i = 0;
      for (size_t m = 0; m < rows(); m++) {
        const size_t weight_width = (size_t) weight_widths[m];
        for (size_t n = 0; n < weight_width; n++, i++) {
          weight_accumulator += (uint64_t) input[i] * (uint64_t) weights[i * 2];
          unweight_accumulator += (uint64_t) input[i] * (uint64_t) weights[i * 2 + 1];
        }
        output_ref[m] = weight_accumulator;
        weight_accumulator = unweight_accumulator;
      }

      // Call optimized micro-kernel.
      filterbank_accumulate(rows(), batch(), input.data(), weight_widths.data(), weights.data(), output.data());

      // Verify results.
      for (size_t m = 0; m < rows(); m++) {
        ASSERT_EQ(output[m], output_ref[m])
            << "at row " << m << " / " << rows();
      }
    }
  }

 private:
  size_t rows_{1};
  size_t batch_{1};
  size_t iterations_{15};
};
