// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/params.h>


class PadMicrokernelTester {
 public:
  inline PadMicrokernelTester& m(size_t m) {
    assert(m != 0);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline PadMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline PadMicrokernelTester& l(size_t l) {
    this->l_ = l;
    return *this;
  }

  inline size_t l() const {
    return this->l_;
  }

  inline PadMicrokernelTester& r(size_t r) {
    this->r_ = r;
    return *this;
  }

  inline size_t r() const {
    return this->r_;
  }

  inline PadMicrokernelTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return n();
    } else {
      assert(this->x_stride_ >= n());
      return this->x_stride_;
    }
  }

  inline PadMicrokernelTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return l() + n() + r();
    } else {
      assert(this->y_stride_ >= l() + n() + r());
      return this->y_stride_;
    }
  }

  inline PadMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_pad_ukernel_function pad) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    const uint32_t c = u32rng();
    std::vector<uint32_t> x(n() + (m() - 1) * x_stride() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t> y((l() + n() + r()) + (m() - 1) * y_stride());
    std::vector<uint32_t> y_ref(m() * (l() + n() + r()));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u32rng));
      std::generate(y.begin(), y.end(), std::ref(u32rng));

      // Compute reference results.
      std::fill(y_ref.begin(), y_ref.end(), c);
      for (size_t k = 0; k < m(); k++) {
        for (size_t i = 0; i < n(); i++) {
          y_ref[k * (l() + n() + r()) + l() + i] = x[k * x_stride() + i];
        }
      }

      // Call optimized micro-kernel.
      pad(
        m(),
        n() * sizeof(uint32_t),
        l() * sizeof(uint32_t),
        r() * sizeof(uint32_t),
        c,
        x.data(), x_stride() * sizeof(float),
        y.data(), y_stride() * sizeof(float));

      // Verify results.
      for (size_t k = 0; k < m(); k++) {
        for (size_t i = 0; i < l() + n() + r(); i++) {
          ASSERT_EQ(y_ref[k * (l() + n() + r()) + i], y[k * y_stride() + i])
            << "at (" << k << ", " << i << "), "
            << "m = " << m() << ", n = " << n() << ", c = " << c;
        }
      }
    }
  }

 private:
  size_t m_{1};
  size_t n_{1};
  size_t l_{0};
  size_t r_{0};
  size_t x_stride_{0};
  size_t y_stride_{0};
  size_t iterations_{15};
};
