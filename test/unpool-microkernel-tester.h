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

#include <xnnpack/microfnptr.h>


class UnpoolMicrokernelTester {
 public:
  inline UnpoolMicrokernelTester& p(size_t p) {
    assert(p != 0);
    this->p_ = p;
    return *this;
  }

  inline size_t p() const {
    return this->p_;
  }

  inline UnpoolMicrokernelTester& c(size_t c) {
    assert(c != 0);
    this->c_ = c;
    return *this;
  }

  inline size_t c() const {
    return this->c_;
  }

  inline UnpoolMicrokernelTester& f(uint32_t f) {
    this->f_ = f;
    return *this;
  }

  inline uint32_t f() const {
    return this->f_;
  }

  inline UnpoolMicrokernelTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return c();
    } else {
      assert(this->y_stride_ >= c());
      return this->y_stride_;
    }
  }

  inline UnpoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x32_unpool_ukernel_fn unpool) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto x_rng = std::bind(std::uniform_int_distribution<uint32_t>(), std::ref(rng));
    auto i_rng = std::bind(std::uniform_int_distribution<uint32_t>(0, uint32_t(p() - 1)), std::ref(rng));

    std::vector<uint32_t> x(c());
    std::vector<uint32_t> i(c());
    std::vector<uint32_t> y((p() - 1) * y_stride() + c());
    std::vector<uint32_t*> indirect_y(p());
    std::vector<uint32_t> y_ref((p() - 1) * y_stride() + c());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(x_rng));
      std::generate(i.begin(), i.end(), std::ref(i_rng));
      std::generate(y.begin(), y.end(), std::ref(x_rng));

      for (size_t i = 0; i < indirect_y.size(); i++) {
        indirect_y[i] = y.data() + i * y_stride();
      }
      std::shuffle(indirect_y.begin(), indirect_y.end(), rng);

      // Compute reference output.
      std::fill(y_ref.begin(), y_ref.end(), f());
      for (size_t k = 0; k < c(); k++) {
        const uint32_t idx = i[k];
        (indirect_y[idx] - y.data() + y_ref.data())[k] = x[k];
      }

      // Call optimized micro-kernel.
      unpool(p(), c(), f(), x.data(), i.data(), indirect_y.data());

      // Verify results.
      for (size_t i = 0; i < p(); i++) {
        for (size_t k = 0; k < c(); k++) {
          EXPECT_EQ(y_ref[i * y_stride() + k], y[i * y_stride() + k])
            << "at pixel " << i << ", channel " << k
            << ", p = " << p() << ", c = " << c();
        }
      }
    }
  }

 private:
  size_t p_{1};
  size_t c_{1};
  uint32_t f_{0};
  size_t y_stride_{0};
  size_t iterations_{15};
};
