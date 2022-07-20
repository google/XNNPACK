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


class WindowMicrokernelTester {
 public:
  inline WindowMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline WindowMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline WindowMicrokernelTester& shift(uint32_t shift) {
    assert(shift < 32);
    this->shift_ = shift;
    return *this;
  }

  inline uint32_t shift() const {
    return this->shift_;
  }

  inline WindowMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline WindowMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s16_window_ukernel_function window) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i16rng = std::bind(std::uniform_int_distribution<int16_t>(), std::ref(rng));

    std::vector<int16_t> x((channels() * rows()) + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t, AlignedAllocator<int16_t, 64>> w(channels() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y((channels() * rows()) + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y_ref(channels() * rows());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i16rng));
      std::generate(w.begin(), w.end(), std::ref(i16rng));
      std::fill(y.begin(), y.end(), INT16_C(0xDEAD));
      const int16_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          const int16_t x_value = x_data[n * channels() + c];
          int32_t value = ((int32_t) x_value * (int32_t) w[c]) >> shift();
          value = std::min(value, (int32_t) std::numeric_limits<int16_t>::max());
          value = std::max(value, (int32_t) std::numeric_limits<int16_t>::min());
          y_ref[n * channels() + c] = value;
        }
      }

      // Call optimized micro-kernel.
      window(rows(), channels(), x_data, w.data(), shift(), y.data());

      // Verify results.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_EQ(y[n * channels() + c], y_ref[n * channels() + c])
            << "at row " << n << " / " << rows()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  uint32_t shift_{12};
  bool inplace_{false};
  size_t iterations_{15};
};
