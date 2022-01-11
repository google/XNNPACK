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

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class VBinaryMicrokernelTester {
 public:
  enum class OpType {
    Add,
    Div,
    Max,
    Min,
    Mul,
    Sub,
    SqrDiff,
  };

  inline VBinaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VBinaryMicrokernelTester& inplace_a(bool inplace_a) {
    this->inplace_a_ = inplace_a;
    return *this;
  }

  inline bool inplace_a() const {
    return this->inplace_a_;
  }

  inline VBinaryMicrokernelTester& inplace_b(bool inplace_b) {
    this->inplace_b_ = inplace_b;
    return *this;
  }

  inline bool inplace_b() const {
    return this->inplace_b_;
  }

  inline VBinaryMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VBinaryMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VBinaryMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_vbinary_ukernel_function vbinary, OpType op_type) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), rng);
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f16rng));
      std::generate(b.begin(), b.end(), std::ref(f16rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(f16rng));
      } else {
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* a_data = inplace_a() ? y.data() : a.data();
      const uint16_t* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::Add:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) + fp16_ieee_to_fp32_value(b_data[i]);
            break;
          case OpType::Div:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) / fp16_ieee_to_fp32_value(b_data[i]);
            break;
          case OpType::Max:
            y_ref[i] = std::max<float>(fp16_ieee_to_fp32_value(a_data[i]), fp16_ieee_to_fp32_value(b_data[i]));
            break;
          case OpType::Min:
            y_ref[i] = std::min<float>(fp16_ieee_to_fp32_value(a_data[i]), fp16_ieee_to_fp32_value(b_data[i]));
            break;
          case OpType::Mul:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) * fp16_ieee_to_fp32_value(b_data[i]);
            break;
          case OpType::SqrDiff:
          {
            const float diff = fp16_ieee_to_fp32_value(a_data[i]) - fp16_ieee_to_fp32_value(b_data[i]);
            y_ref[i] = diff * diff;
            break;
          }
          case OpType::Sub:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) - fp16_ieee_to_fp32_value(b_data[i]);
            break;
        }
      }

      // Call optimized micro-kernel.
      vbinary(batch_size() * sizeof(uint16_t), a_data, b_data, y.data(), nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i], std::max(1.0e-4f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size();
      }
    }
  }

  void Test(xnn_f16_vbinary_minmax_ukernel_function vbinary_minmax, OpType op_type, xnn_init_f16_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), rng);
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f16rng));
      std::generate(b.begin(), b.end(), std::ref(f16rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(f16rng));
      } else {
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* a_data = inplace_a() ? y.data() : a.data();
      const uint16_t* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::Add:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) + fp16_ieee_to_fp32_value(b_data[i]);
            break;
          case OpType::Div:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) / fp16_ieee_to_fp32_value(b_data[i]);
            break;
          case OpType::Max:
            y_ref[i] = std::max<float>(fp16_ieee_to_fp32_value(a_data[i]), fp16_ieee_to_fp32_value(b_data[i]));
            break;
          case OpType::Min:
            y_ref[i] = std::min<float>(fp16_ieee_to_fp32_value(a_data[i]), fp16_ieee_to_fp32_value(b_data[i]));
            break;
          case OpType::Mul:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) * fp16_ieee_to_fp32_value(b_data[i]);
            break;
          case OpType::SqrDiff:
          {
            const float diff = fp16_ieee_to_fp32_value(a_data[i]) - fp16_ieee_to_fp32_value(b_data[i]);
            y_ref[i] = diff * diff;
            break;
          }
          case OpType::Sub:
            y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) - fp16_ieee_to_fp32_value(b_data[i]);
            break;
        }
      }

      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_range > 0.0f ?
        (accumulated_max - accumulated_range / 255.0f * float(255 - qmax())) :
        +std::numeric_limits<float>::infinity()));
      const float y_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_range > 0.0f ?
        (accumulated_min + accumulated_range / 255.0f * float(qmin())) :
        -std::numeric_limits<float>::infinity()));
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max<float>(std::min<float>(y_ref[i], y_max), y_min);
      }

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params,
        fp16_ieee_from_fp32_value(y_min), fp16_ieee_from_fp32_value(y_max));

      // Call optimized micro-kernel.
      vbinary_minmax(batch_size() * sizeof(uint16_t), a_data, b_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i], std::max(1.0e-4f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size();
      }
    }
  }

  void Test(xnn_f32_vbinary_ukernel_function vbinary, OpType op_type, xnn_init_f32_default_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), rng);

    std::vector<float> a(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* a_data = inplace_a() ? y.data() : a.data();
      const float* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::Add:
            y_ref[i] = a_data[i] + b_data[i];
            break;
          case OpType::Div:
            y_ref[i] = a_data[i] / b_data[i];
            break;
          case OpType::Max:
            y_ref[i] = std::max<float>(a_data[i], b_data[i]);
            break;
          case OpType::Min:
            y_ref[i] = std::min<float>(a_data[i], b_data[i]);
            break;
          case OpType::Mul:
            y_ref[i] = a_data[i] * b_data[i];
            break;
          case OpType::SqrDiff:
          {
            const float diff = a_data[i] - b_data[i];
            y_ref[i] = diff * diff;
            break;
          }
          case OpType::Sub:
            y_ref[i] = a_data[i] - b_data[i];
            break;
        }
      }

      // Prepare parameters.
      xnn_f32_default_params params;
      if (init_params) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vbinary(batch_size() * sizeof(float), a_data, b_data, y.data(), init_params != nullptr ? &params : nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << " / " << batch_size();
      }
    }
  }

  void Test(xnn_f32_vbinary_relu_ukernel_function vbinary_relu, OpType op_type) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), rng);

    std::vector<float> a(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* a_data = inplace_a() ? y.data() : a.data();
      const float* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::Add:
            y_ref[i] = a_data[i] + b_data[i];
            break;
          case OpType::Div:
            y_ref[i] = a_data[i] / b_data[i];
            break;
          case OpType::Max:
            y_ref[i] = std::max<float>(a_data[i], b_data[i]);
            break;
          case OpType::Min:
            y_ref[i] = std::min<float>(a_data[i], b_data[i]);
            break;
          case OpType::Mul:
            y_ref[i] = a_data[i] * b_data[i];
            break;
          case OpType::SqrDiff:
          {
            const float diff = a_data[i] - b_data[i];
            y_ref[i] = diff * diff;
            break;
          }
          case OpType::Sub:
            y_ref[i] = a_data[i] - b_data[i];
            break;
        }
      }
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max(y_ref[i], 0.0f);
      }

      // Call optimized micro-kernel.
      vbinary_relu(batch_size() * sizeof(float), a_data, b_data, y.data(), nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_GE(y[i], 0.0f)
          << "at " << i << " / " << batch_size();
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << " / " << batch_size();
      }
    }
  }

  void Test(xnn_f32_vbinary_minmax_ukernel_function vbinary_minmax, OpType op_type, xnn_init_f32_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), rng);

    std::vector<float> a(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> b(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(f32rng));
      std::generate(b.begin(), b.end(), std::ref(f32rng));
      if (inplace_a() || inplace_b()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* a_data = inplace_a() ? y.data() : a.data();
      const float* b_data = inplace_b() ? y.data() : b.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::Add:
            y_ref[i] = a_data[i] + b_data[i];
            break;
          case OpType::Div:
            y_ref[i] = a_data[i] / b_data[i];
            break;
          case OpType::Max:
            y_ref[i] = std::max<float>(a_data[i], b_data[i]);
            break;
          case OpType::Min:
            y_ref[i] = std::min<float>(a_data[i], b_data[i]);
            break;
          case OpType::Mul:
            y_ref[i] = a_data[i] * b_data[i];
            break;
          case OpType::SqrDiff:
          {
            const float diff = a_data[i] - b_data[i];
            y_ref[i] = diff * diff;
            break;
          }
          case OpType::Sub:
            y_ref[i] = a_data[i] - b_data[i];
            break;
        }
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = accumulated_range > 0.0f ?
        (accumulated_max - accumulated_range / 255.0f * float(255 - qmax())) :
        +std::numeric_limits<float>::infinity();
      const float y_min = accumulated_range > 0.0f ?
        (accumulated_min + accumulated_range / 255.0f * float(qmin())) :
        -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max<float>(std::min<float>(y_ref[i], y_max), y_min);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, y_min, y_max);

      // Call optimized micro-kernel.
      vbinary_minmax(batch_size() * sizeof(float), a_data, b_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << " / " << batch_size();
      }
    }
  }

 private:
  size_t batch_size_{1};
  bool inplace_a_{false};
  bool inplace_b_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
