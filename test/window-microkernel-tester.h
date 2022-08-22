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

  inline WindowMicrokernelTester& batch(size_t batch) {
    assert(batch != 0);
    this->batch_ = batch;
    return *this;
  }

  inline size_t batch() const {
    return this->batch_;
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

    std::vector<int16_t> x(batch() * rows() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t, AlignedAllocator<int16_t, 64>> w(batch() + XNN_EXTRA_BYTES / sizeof(int16_t));
    std::vector<int16_t> y(batch() * rows() + (inplace() ? XNN_EXTRA_BYTES / sizeof(int16_t) : 0));
    std::vector<int16_t> y_ref(batch() * rows());
    const int16_t* x_data = inplace() ? y.data() : x.data();

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i16rng));
      std::generate(w.begin(), w.end(), std::ref(i16rng));
      std::generate(y.begin(), y.end(), std::ref(i16rng));
      std::generate(y_ref.begin(), y_ref.end(), std::ref(i16rng));

      // Compute reference results.
      for (size_t m = 0; m < rows(); m++) {
        for (size_t n = 0; n < batch(); n++) {
          const int16_t x_value = x_data[m * batch() + n];
          int32_t value = ((int32_t) x_value * (int32_t) w[n]) >> shift();
          value = std::min(value, (int32_t) std::numeric_limits<int16_t>::max());
          value = std::max(value, (int32_t) std::numeric_limits<int16_t>::min());
          y_ref[m * batch() + n] = value;
        }
      }

      // Call optimized micro-kernel.
      window(rows(), batch(), x_data, w.data(), y.data(), shift());

      // Verify results.
      for (size_t m = 0; m < rows(); m++) {
        for (size_t n = 0; n < batch(); n++) {
          ASSERT_EQ(y[m * batch() + n], y_ref[m * batch() + n])
            << "at row " << m << " / " << rows()
            << ", shift " << shift()
            << ", batch " << n << " / " << batch();
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t batch_{1};
  uint32_t shift_{12};
  bool inplace_{false};
  size_t iterations_{15};
};
