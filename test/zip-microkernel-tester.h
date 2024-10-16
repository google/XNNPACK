// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/microfnptr.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

class ZipMicrokernelTester {
 public:
  ZipMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  size_t n() const {
    return this->n_;
  }

  ZipMicrokernelTester& g(size_t g) {
    assert(g != 0);
    this->g_ = g;
    return *this;
  }

  size_t g() const {
    return this->g_;
  }

  ZipMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x8_zipc_ukernel_fn zip) const {
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<uint8_t> x(n() * g());
    xnnpack::Buffer<uint8_t> x_ref(g() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      xnnpack::fill_uniform_random_bits(x.data(), x.size(), rng);

      // Call optimized micro-kernel.
      zip(n() * sizeof(uint8_t), x.data(), x_ref.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          EXPECT_EQ(uint32_t(x_ref[i * g() + j]), uint32_t(x[j * n() + i]))
            << "at element " << i << ", group " << j;
        }
      }
    }
  }

  void Test(xnn_x8_zipv_ukernel_fn zip) const {
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<uint8_t> x(n() * g());
    xnnpack::Buffer<uint8_t> x_ref(g() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      xnnpack::fill_uniform_random_bits(x.data(), x.size(), rng);

      // Call optimized micro-kernel.
      zip(n() * sizeof(uint8_t), g(), x.data(), x_ref.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          EXPECT_EQ(uint32_t(x_ref[i * g() + j]), uint32_t(x[j * n() + i]))
            << "at element " << i << ", group " << j;
        }
      }
    }
  }

  void Test(xnn_x32_zipc_ukernel_fn zip) const {
    xnnpack::ReplicableRandomDevice rng;
    auto u32rng = [&rng]() {
      return std::uniform_int_distribution<uint32_t>()(rng);
    };

    xnnpack::Buffer<uint32_t> x(n() * g());
    xnnpack::Buffer<uint32_t> x_ref(g() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u32rng));

      // Call optimized micro-kernel.
      zip(n() * sizeof(uint32_t), x.data(), x_ref.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          EXPECT_EQ(x_ref[i * g() + j], x[j * n() + i])
            << "at element " << i << ", group " << j;
        }
      }
    }
  }

  void Test(xnn_x32_zipv_ukernel_fn zip) const {
    xnnpack::ReplicableRandomDevice rng;
    auto u32rng = [&rng]() {
      return std::uniform_int_distribution<uint32_t>()(rng);
    };

    xnnpack::Buffer<uint32_t> x(n() * g());
    xnnpack::Buffer<uint32_t> x_ref(g() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u32rng));

      // Call optimized micro-kernel.
      zip(n() * sizeof(uint32_t), g(), x.data(), x_ref.data());

      // Verify results.
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          EXPECT_EQ(x_ref[i * g() + j], x[j * n() + i])
            << "at element " << i << ", group " << j;
        }
      }
    }
  }

 private:
  size_t n_{1};
  size_t g_{1};
  size_t iterations_{3};
};
