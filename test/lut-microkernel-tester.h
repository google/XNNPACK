// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

class LUTMicrokernelTester {
 public:
  LUTMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  LUTMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  LUTMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x8_lut_ukernel_fn lut) const {
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<uint8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    XNN_ALIGN(64) std::array<uint8_t, 256> t;
    xnnpack::Buffer<uint8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
    xnnpack::Buffer<uint8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      xnnpack::fill_uniform_random_bits(x.data(), x.size(), rng);
      xnnpack::fill_uniform_random_bits(t.data(), t.size(), rng);
      if (inplace()) {
        xnnpack::fill_uniform_random_bits(y.data(), y.size(), rng);
      }
      const uint8_t* x_data = x.data();
      if (inplace()) {
        std::copy(y.cbegin(), y.cend(), x.begin());
        x_data = y.data();
      }

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = t[x_data[i]];
      }

      // Call optimized micro-kernel.
      lut(batch_size(), x_data, y.data(), t.data());

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at position " << i << " / " << batch_size()
          << ", input " << uint32_t(x[i]);
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_{false};
  size_t iterations_{15};
};
