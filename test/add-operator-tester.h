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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>


class AddOperatorTester {
 public:
  inline AddOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AddOperatorTester& a_stride(size_t a_stride) {
    assert(a_stride != 0);
    this->a_stride_ = a_stride;
    return *this;
  }

  inline size_t a_stride() const {
    if (this->a_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->a_stride_ >= this->channels_);
      return this->a_stride_;
    }
  }

  inline AddOperatorTester& b_stride(size_t b_stride) {
    assert(b_stride != 0);
    this->b_stride_ = b_stride;
    return *this;
  }

  inline size_t b_stride() const {
    if (this->b_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->b_stride_ >= this->channels_);
      return this->b_stride_;
    }
  }

  inline AddOperatorTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  inline size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->y_stride_ >= this->channels_);
      return this->y_stride_;
    }
  }

  inline AddOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline AddOperatorTester& a_scale(float a_scale) {
    assert(a_scale > 0.0f);
    assert(std::isnormal(a_scale));
    this->a_scale_ = a_scale;
    return *this;
  }

  inline float a_scale() const {
    return this->a_scale_;
  }

  inline AddOperatorTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  inline uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  inline AddOperatorTester& b_scale(float b_scale) {
    assert(b_scale > 0.0f);
    assert(std::isnormal(b_scale));
    this->b_scale_ = b_scale;
    return *this;
  }

  inline float b_scale() const {
    return this->b_scale_;
  }

  inline AddOperatorTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  inline uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  inline AddOperatorTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  inline float y_scale() const {
    return this->y_scale_;
  }

  inline AddOperatorTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  inline uint8_t y_zero_point() const {
    return this->y_zero_point_;
  }

  inline AddOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline AddOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AddOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> a(XNN_EXTRA_BYTES / sizeof(uint8_t) + (batch_size() - 1) * a_stride() + channels());
    std::vector<uint8_t> b(XNN_EXTRA_BYTES / sizeof(uint8_t) + (batch_size() - 1) * b_stride() + channels());
    std::vector<uint8_t> y((batch_size() - 1) * y_stride() + channels());
    std::vector<float> y_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      if (batch_size() * channels() > 3) {
        ASSERT_NE(*std::max_element(a.cbegin(), a.cend()), *std::min_element(a.cbegin(), a.cend()));
        ASSERT_NE(*std::max_element(b.cbegin(), b.cend()), *std::min_element(b.cbegin(), b.cend()));
      }

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          y_ref[i * channels() + c] = float(y_zero_point()) +
            float(int32_t(a[i * a_stride() + c]) - int32_t(a_zero_point())) * (a_scale() / y_scale()) +
            float(int32_t(b[i * b_stride() + c]) - int32_t(b_zero_point())) * (b_scale() / y_scale());
          y_ref[i * channels() + c] = std::min<float>(y_ref[i * channels() + c], float(qmax()));
          y_ref[i * channels() + c] = std::max<float>(y_ref[i * channels() + c], float(qmin()));
        }
      }

      // Create, setup, run, and destroy Add operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t add_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_add_nc_q8(
          channels(), a_stride(), b_stride(), y_stride(),
          a_zero_point(), a_scale(),
          b_zero_point(), b_scale(),
          y_zero_point(), y_scale(),
          qmin(), qmax(),
          0, &add_op));
      ASSERT_NE(nullptr, add_op);

      // Smart pointer to automatically delete add_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_add_op(add_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_add_nc_q8(
          add_op,
          batch_size(),
          a.data(), b.data(), y.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(add_op, nullptr /* thread pool */));

      /// Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(uint32_t(y[i * y_stride() + c]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(y[i * y_stride() + c]), uint32_t(qmin()));
          ASSERT_NEAR(float(int32_t(y[i * y_stride() + c])), y_ref[i * channels() + c], 0.6f);
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<float> a(XNN_EXTRA_BYTES / sizeof(float) + (batch_size() - 1) * a_stride() + channels());
    std::vector<float> b(XNN_EXTRA_BYTES / sizeof(float) + (batch_size() - 1) * b_stride() + channels());
    std::vector<float> y((batch_size() - 1) * y_stride() + channels());
    std::vector<float> y_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      std::fill(y.begin(), y.end(), nanf(""));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          y_ref[i * channels() + c] = a[i * a_stride() + c] + b[i * b_stride() + c];
        }
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = batch_size() * channels() == 1 ?
        -std::numeric_limits<float>::infinity() : accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float y_max = batch_size() * channels() == 1 ?
        +std::numeric_limits<float>::infinity() : accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          y_ref[i * channels() + c] = std::min(std::max(y_ref[i * channels() + c], y_min), y_max);
        }
      }

      // Create, setup, run, and destroy Add operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t add_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_add_nc_f32(
          channels(), a_stride(), b_stride(), y_stride(),
          y_min, y_max,
          0, &add_op));
      ASSERT_NE(nullptr, add_op);

      // Smart pointer to automatically delete add_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_add_op(add_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_add_nc_f32(
          add_op,
          batch_size(),
          a.data(), b.data(), y.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(add_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(y[i * y_stride() + c], y_ref[i * channels() + c], 1.0e-6f * y_ref[i * channels() + c])
            << "i = " << i << ", c = " << c;
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t a_stride_{0};
  size_t b_stride_{0};
  size_t y_stride_{0};
  float a_scale_{0.75f};
  float b_scale_{1.25f};
  float y_scale_{0.96875f};
  uint8_t a_zero_point_{121};
  uint8_t b_zero_point_{127};
  uint8_t y_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
