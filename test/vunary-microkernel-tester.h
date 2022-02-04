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


class VUnaryMicrokernelTester {
 public:
  enum class OpType {
    ReLU,
    RoundToNearestEven,
    RoundTowardsZero,
    RoundUp,
    RoundDown,
  };

  enum class Variant {
    Native,
    Scalar,
  };

  inline VUnaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VUnaryMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VUnaryMicrokernelTester& slope(float slope) {
    this->slope_ = slope;
    return *this;
  }

  inline float slope() const {
    return this->slope_;
  }

  inline VUnaryMicrokernelTester& prescale(float prescale) {
    this->prescale_ = prescale;
    return *this;
  }

  inline float prescale() const {
    return this->prescale_;
  }

  inline VUnaryMicrokernelTester& alpha(float alpha) {
    this->alpha_ = alpha;
    return *this;
  }

  inline float alpha() const {
    return this->alpha_;
  }

  inline VUnaryMicrokernelTester& beta(float beta) {
    this->beta_ = beta;
    return *this;
  }

  inline float beta() const {
    return this->beta_;
  }

  inline VUnaryMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VUnaryMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VUnaryMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_vunary_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-125.0f, 125.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<double> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::ReLU:
            y_ref[i] = std::max(x_data[i], 0.0f);
            break;
          default:
            GTEST_FAIL() << "Unexpected operation type";
            return;
        }
      }

      // Call optimized micro-kernel.
      vunary(batch_size() * sizeof(float), x_data, y.data(), nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::max(5.0e-6, std::abs(y_ref[i]) * 1.0e-5))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vabs_ukernel_function vabs, xnn_init_f32_abs_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::abs(x_data[i]);
      }

      // Prepare parameters.
      union xnn_f32_abs_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vabs(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vclamp_ukernel_function vclamp, xnn_init_f32_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 255.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max(std::min(x_data[i], float(qmax())), float(qmin()));
      }

      // Prepare parameters.
      union xnn_f32_minmax_params params;
      init_params(&params, float(qmin()), float(qmax()));

      // Call optimized micro-kernel.
      vclamp(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_velu_ukernel_function velu, xnn_init_f32_elu_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 20.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<double> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::signbit(x_data[i]) ? alpha() * std::expm1(double(x_data[i]) * prescale()) : double(x_data[i]) * beta();
      }

      // Prepare parameters.
      union xnn_f32_elu_params params;
      init_params(&params, prescale(), alpha(), beta());

      // Call optimized micro-kernel.
      velu(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::max(5.0e-6, std::abs(y_ref[i]) * 1.0e-5))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vhswish_ukernel_function vhswish, xnn_init_f32_hswish_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-4.0f, 4.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<double> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = (x_data[i] / 6.0f) * std::max(std::min(x_data[i] + 3.0f, 6.0f), 0.0f);
      }

      // Prepare parameters.
      union xnn_f32_hswish_params params;
      init_params(&params);

      // Call optimized micro-kernel.
      vhswish(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::max(5.0e-6, std::abs(y_ref[i]) * 1.0e-5))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f16_vlrelu_ukernel_function vlrelu, xnn_init_f16_lrelu_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-125.0f, 125.0f), std::ref(rng));
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    const uint16_t slope_as_half = fp16_ieee_from_fp32_value(slope());
    const float slope_as_float = fp16_ieee_to_fp32_value(slope_as_half);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f16rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f16rng));
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const float x_value = fp16_ieee_to_fp32_value(x_data[i]);
        y_ref[i] = std::signbit(x_value) ? x_value * slope_as_float : x_value;
      }

      // Prepare parameters.
      union xnn_f16_lrelu_params params;
      init_params(&params, slope_as_half);

      // Call optimized micro-kernel.
      vlrelu(batch_size() * sizeof(uint16_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(
            fp16_ieee_to_fp32_value(y[i]),
            y_ref[i],
            std::max(1.0e-4f, std::abs(y_ref[i]) * 1.0e-3f))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << fp16_ieee_to_fp32_value(x[i]);
      }
    }
  }

  void Test(xnn_f32_vlrelu_ukernel_function vlrelu, xnn_init_f32_lrelu_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-125.0f, 125.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<double> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::signbit(x_data[i]) ? x_data[i] * slope() : x_data[i];
      }

      // Prepare parameters.
      union xnn_f32_lrelu_params params;
      init_params(&params, slope());

      // Call optimized micro-kernel.
      vlrelu(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vneg_ukernel_function vneg, xnn_init_f32_neg_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = -x_data[i];
      }

      // Prepare parameters.
      union xnn_f32_neg_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vneg(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vround_ukernel_function vrnd, OpType op_type, xnn_init_f32_rnd_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
          case OpType::RoundToNearestEven:
            y_ref[i] = std::nearbyint(double(x_data[i]));
            break;
          case OpType::RoundTowardsZero:
            y_ref[i] = std::trunc(double(x_data[i]));
            break;
          case OpType::RoundUp:
            y_ref[i] = std::ceil(double(x_data[i]));
            break;
          case OpType::RoundDown:
            y_ref[i] = std::floor(double(x_data[i]));
            break;
          default:
            GTEST_FAIL() << "Unexpected operation type";
            return;
        }
      }

      // Prepare parameters.
      xnn_f32_rnd_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vrnd(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vsigmoid_ukernel_function vsigmoid, xnn_init_f32_sigmoid_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-125.0f, 125.0f);
    auto f32rng = std::bind(distribution, std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<double> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const double e = std::exp(double(x_data[i]));
        y_ref[i] = e / (1.0 + e);
      }

      // Prepare parameters.
      union xnn_f32_sigmoid_params params;
      init_params(&params);

      // Call optimized micro-kernel.
      vsigmoid(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::max(5.0e-6, std::abs(y_ref[i]) * 1.0e-5))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vsqr_ukernel_function vsqr, xnn_init_f32_default_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = x_data[i] * x_data[i];
      }

      // Prepare parameters.
      union xnn_f32_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vsqr(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  void Test(xnn_f32_vsqrt_ukernel_function vsqrt, xnn_init_f32_sqrt_params_fn init_params = nullptr) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 10.0f), std::ref(rng));

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f32rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f32rng));
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::sqrt(x_data[i]);
      }

      // Prepare parameters.
      union xnn_f32_sqrt_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      vsqrt(batch_size() * sizeof(float), x_data, y.data(), init_params != nullptr ? &params : nullptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  inline void Test(xnn_f32_vabs_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_velu_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vneg_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vrelu_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  void Test(xnn_f16_vclamp_ukernel_function vclamp, xnn_init_f16_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 255.0f), std::ref(rng));
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f16rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f16rng));
      } else {
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::max(std::min(fp16_ieee_to_fp32_value(x_data[i]), float(qmax())), float(qmin()));
      }

      // Prepare parameters.
      union xnn_f16_minmax_params params;
      init_params(&params, fp16_ieee_from_fp32_value(float(qmin())), fp16_ieee_from_fp32_value(float(qmax())));

      // Call optimized micro-kernel.
      vclamp(batch_size() * sizeof(uint16_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y_ref[i], fp16_ieee_to_fp32_value(y[i]), std::max(1.0e-3f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << fp16_ieee_to_fp32_value(x[i]);
      }
    }
  }

  void Test(xnn_f16_vhswish_ukernel_function vhswish, xnn_init_f16_hswish_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-4.0f, 4.0f), std::ref(rng));
    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(f16rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f16rng));
      } else {
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const float x_value = fp16_ieee_to_fp32_value(x_data[i]);
        y_ref[i] = (x_value / 6.0f) * std::max(std::min(x_value + 3.0f, 6.0f), 0.0f);
      }

      // Prepare parameters.
      union xnn_f16_hswish_params params;
      init_params(&params);

      // Call optimized micro-kernel.
      vhswish(batch_size() * sizeof(uint16_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y_ref[i], fp16_ieee_to_fp32_value(y[i]), std::max(1.0e-3f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << fp16_ieee_to_fp32_value(x[i]);
      }
    }
  }

  void Test(xnn_s8_vclamp_ukernel_function vclamp, xnn_init_s8_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

    std::vector<int8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(int8_t) : 0));
    std::vector<int8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(i8rng));
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), INT8_C(0xA5));
      }
      const int8_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::min(std::max(x_data[i], int8_t(qmin() - 0x80)), int8_t(qmax() - 0x80));
      }

      // Prepare parameters.
      union xnn_s8_minmax_params params;
      init_params(&params, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Call optimized micro-kernel.
      vclamp(batch_size() * sizeof(int8_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(int32_t(y_ref[i]), int32_t(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << int32_t(x[i]);
      }
    }
  }

  void Test(xnn_u8_vclamp_ukernel_function vclamp, xnn_init_u8_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(
      std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

    std::vector<uint8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
    std::vector<uint8_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), UINT8_C(0xA5));
      }
      const uint8_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = std::min(std::max(x_data[i], qmin()), qmax());
      }

      // Prepare parameters.
      union xnn_u8_minmax_params params;
      init_params(&params, qmin(), qmax());

      // Call optimized micro-kernel.
      vclamp(batch_size() * sizeof(uint8_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << uint32_t(x[i]);
      }
    }
  }

 private:
  size_t batch_size_ = 1;
  bool inplace_ = false;
  float slope_ = 0.5f;
  float prescale_ = 1.0f;
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
  uint8_t qmin_ = 0;
  uint8_t qmax_ = 255;
  size_t iterations_ = 15;
};
