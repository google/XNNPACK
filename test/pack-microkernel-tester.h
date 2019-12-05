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
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/params.h>


class PackMicrokernelTester {
 public:
  inline PackMicrokernelTester& mr(size_t mr) {
    assert(mr != 0);
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline PackMicrokernelTester& m(size_t m) {
    assert(m != 0);
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline PackMicrokernelTester& k(size_t k) {
    assert(k != 0);
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline PackMicrokernelTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  inline size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return k();
    } else {
      assert(this->x_stride_ >= k());
      return this->x_stride_;
    }
  }

  inline PackMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x32_packx_ukernel_function packx) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    const uint32_t c = u32rng();
    std::vector<uint32_t> x(k() + (m() - 1) * x_stride() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> y(mr() * k());
    std::vector<uint32_t> y_ref(mr() * k());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u32rng));
      std::generate(y.begin(), y.end(), std::ref(u32rng));

      // Compute reference results.
      std::fill(y_ref.begin(), y_ref.end(), c);
      for (size_t i = 0; i < mr(); i++) {
        for (size_t j = 0; j < k(); j++) {
          y_ref[j * mr() + i] = x[std::min(i, m() - 1) * x_stride() + j];
        }
      }

      // Call optimized micro-kernel.
      packx(
        m(), k(),
        x.data(), x_stride() * sizeof(uint32_t),
        y.data());

      // Verify results.
      for (size_t i = 0; i < mr(); i++) {
        for (size_t j = 0; j < k(); j++) {
          ASSERT_EQ(y_ref[j * mr() + i], y[j * mr() + i])
            << "at pixel = " << i << ", channel = " << j << ", "
            << "m = " << m() << ", k = " << k();
        }
      }
    }
  }

 private:
  size_t mr_{1};
  size_t m_{1};
  size_t k_{1};
  size_t x_stride_{0};
  size_t iterations_{1};
};
