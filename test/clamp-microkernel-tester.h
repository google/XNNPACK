// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class ClampMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline ClampMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ClampMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline ClampMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ClampMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ClampMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_u8_clamp_ukernel_function clamp, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
    std::vector<uint8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* x_data = inplace() ? y.data() : x.data();

      // Prepare clamping parameters.
      union xnn_u8_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_u8_minmax_params(qmin(), qmax());
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_u8_minmax_params(qmin(), qmax());
          break;
      }

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max(std::min(x_data[i], qmax()), qmin());
      }

      // Call optimized micro-kernel.
      clamp(batch_size() * sizeof(uint8_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
          << "at position " << i << ", batch_size = " << batch_size();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
          << "at position " << i << ", batch_size = " << batch_size();
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at position " << i << ", batch_size = " << batch_size()
          << ", qmin = " << uint32_t(qmin()) << ", qmax = " << uint32_t(qmax());
      }
    }
  }

  void Test(xnn_f32_clamp_ukernel_function clamp, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 255.0f), rng);

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f32rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::fill(y.begin(), y.end(), std::nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Prepare output parameters.
      xnn_f32_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_minmax_params(float(qmin()), float(qmax()));
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_minmax_params(float(qmin()), float(qmax()));
          break;
      }

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max(std::min(x_data[i], float(qmax())), float(qmin()));
      }

      // Call optimized micro-kernel.
      clamp(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_LE(y[i], float(qmax()))
          << "at position " << i << ", batch_size = " << batch_size();
        ASSERT_GE(y[i], float(qmin()))
          << "at position " << i << ", batch_size = " << batch_size();
        ASSERT_EQ(y_ref[i], y[i])
          << "at position " << i << ", batch_size = " << batch_size()
          << ", qmin = " << uint32_t(qmin()) << ", qmax = " << uint32_t(qmax());
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_{false};
  uint8_t qmin_{50};
  uint8_t qmax_{200};
  size_t iterations_{15};
};
