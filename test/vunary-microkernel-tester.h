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
    Abs,
    Clamp,
    ELU,
    HardSwish,
    LeakyReLU,
    Negate,
    ReLU,
    RoundToNearestEven,
    RoundTowardsZero,
    RoundUp,
    RoundDown,
    Square,
    SquareRoot,
    Sigmoid,
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
    switch (op_type) {
      case OpType::Clamp:
        distribution = std::uniform_real_distribution<float>(0.0f, 255.0f);
        break;
      case OpType::ELU:
        distribution = std::uniform_real_distribution<float>(-20.0f, 20.0f);
        break;
      case OpType::HardSwish:
        distribution = std::uniform_real_distribution<float>(-4.0f, 4.0f);
        break;
      case OpType::SquareRoot:
        distribution = std::uniform_real_distribution<float>(0.0f, 10.0f);
        break;
      default:
        break;
    }
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
          case OpType::Abs:
            y_ref[i] = std::abs(x_data[i]);
            break;
          case OpType::Clamp:
            y_ref[i] = std::max(std::min(x_data[i], float(qmax())), float(qmin()));
            break;
          case OpType::ELU:
          {
            y_ref[i] = std::signbit(x_data[i]) ? alpha() * std::expm1(double(x_data[i]) * prescale()) : double(x_data[i]) * beta();
            break;
          }
          case OpType::HardSwish:
            y_ref[i] = (x_data[i] / 6.0f) * std::max(std::min(x_data[i] + 3.0f, 6.0f), 0.0f);
            break;
          case OpType::LeakyReLU:
            y_ref[i] = std::signbit(x_data[i]) ? x_data[i] * slope() : x_data[i];
            break;
          case OpType::Negate:
            y_ref[i] = -x_data[i];
            break;
          case OpType::ReLU:
            y_ref[i] = std::max(x_data[i], 0.0f);
            break;
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
          case OpType::Square:
            y_ref[i] = double(x_data[i]) * double(x_data[i]);
            break;
          case OpType::SquareRoot:
            y_ref[i] = std::sqrt(double(x_data[i]));
            break;
          case OpType::Sigmoid:
          {
            const double e = std::exp(double(x_data[i]));
            y_ref[i] = e / (1.0 + e);
            break;
          }
        }
      }

      // Prepare parameters.
      union {
        union xnn_f32_abs_params abs;
        union xnn_f32_elu_params elu;
        union xnn_f32_hswish_params hswish;
        union xnn_f32_lrelu_params lrelu;
        union xnn_f32_minmax_params minmax;
        union xnn_f32_neg_params neg;
        union xnn_f32_relu_params relu;
        union xnn_f32_rnd_params rnd;
        union xnn_f32_sqrt_params sqrt;
      } params;
      switch (op_type) {
        case OpType::Abs:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_abs_params(&params.abs);
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_abs_params(&params.abs);
              break;
          }
          break;
        case OpType::Clamp:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_minmax_params(&params.minmax, float(qmin()), float(qmax()));
              break;
            case Variant::Scalar:
              xnn_init_f32_minmax_scalar_params(&params.minmax, float(qmin()), float(qmax()));
              break;
          }
          break;
        case OpType::ELU:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_elu_params(&params.elu, prescale(), alpha(), beta());
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_elu_params(&params.elu, prescale(), alpha(), beta());
              break;
          }
          break;
        case OpType::HardSwish:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_hswish_params(&params.hswish);
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_hswish_params(&params.hswish);
              break;
          };
          break;
        case OpType::LeakyReLU:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_lrelu_params(&params.lrelu, slope());
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_lrelu_params(&params.lrelu, slope());
              break;
          }
          break;
        case OpType::Negate:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_neg_params(&params.neg);
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_neg_params(&params.neg);
              break;
          }
          break;
        case OpType::RoundToNearestEven:
        case OpType::RoundTowardsZero:
        case OpType::RoundUp:
        case OpType::RoundDown:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_rnd_params(&params.rnd);
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_rnd_params(&params.rnd);
              break;
          }
          break;
        case OpType::ReLU:
        case OpType::Sigmoid:
        case OpType::Square:
          break;
        case OpType::SquareRoot:
          switch (variant) {
            case Variant::Native:
              xnn_init_f32_sqrt_params(&params.sqrt);
              break;
            case Variant::Scalar:
              xnn_init_scalar_f32_sqrt_params(&params.sqrt);
              break;
          }
          break;
      }

      // Call optimized micro-kernel.
      vunary(batch_size() * sizeof(float), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], std::max(5.0e-6, std::abs(y_ref[i]) * 1.0e-5))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << x[i];
      }
    }
  }

  inline void Test(xnn_f32_vabs_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vclamp_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_velu_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vhswish_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vlrelu_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vneg_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vrelu_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vround_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f32_vsqrt_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f32_vunary_ukernel_function(vunary), op_type, variant);
  }

  void Test(xnn_f16_vunary_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_real_distribution<float>(-125.0f, 125.0f);
    switch (op_type) {
      case OpType::Clamp:
        distribution = std::uniform_real_distribution<float>(0.0f, 255.0f);
        break;
      case OpType::ELU:
      case OpType::HardSwish:
        distribution = std::uniform_real_distribution<float>(-20.0f, 20.0f);
        break;
      case OpType::SquareRoot:
        distribution = std::uniform_real_distribution<float>(0.0f, 10.0f);
        break;
      default:
        break;
    }
    auto f32rng = std::bind(distribution, std::ref(rng));
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
        switch (op_type) {
          case OpType::Clamp:
            y_ref[i] = std::max(std::min(fp16_ieee_to_fp32_value(x_data[i]), float(qmax())), float(qmin()));
            break;
          case OpType::HardSwish:
          {
            const float x_value = fp16_ieee_to_fp32_value(x_data[i]);
            y_ref[i] = (x_value / 6.0f) * std::max(std::min(x_value + 3.0f, 6.0f), 0.0f);
            break;
          }
          case OpType::ReLU:
            y_ref[i] = std::max(fp16_ieee_to_fp32_value(x_data[i]), 0.0f);
            break;
          default:
            GTEST_FAIL() << "Unexpected op type";
        }
      }

      // Prepare parameters.
      union {
        struct xnn_f16_hswish_params hswish;
        struct xnn_f16_minmax_params minmax;
      } params;
      switch (op_type) {
        case OpType::HardSwish:
          xnn_init_f16_hswish_params(&params.hswish);
          break;
        case OpType::Clamp:
          xnn_init_f16_minmax_params(&params.minmax,
            fp16_ieee_from_fp32_value(float(qmin())), fp16_ieee_from_fp32_value(float(qmax())));
          break;
        case OpType::ReLU:
          break;
        default:
          GTEST_FAIL() << "Unexpected op type";
      }

      // Call optimized micro-kernel.
      vunary(batch_size() * sizeof(uint16_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y_ref[i], fp16_ieee_to_fp32_value(y[i]), std::max(1.0e-3f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << fp16_ieee_to_fp32_value(x[i]);
      }
    }
  }

  inline void Test(xnn_f16_vclamp_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f16_vunary_ukernel_function(vunary), op_type, variant);
  }

  inline void Test(xnn_f16_vhswish_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_f16_vunary_ukernel_function(vunary), op_type, variant);
  }

  void Test(xnn_u8_vunary_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    ASSERT_EQ(op_type, OpType::Clamp);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution = std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max());
    auto u8rng = std::bind(distribution, std::ref(rng));

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
        switch (op_type) {
          case OpType::Clamp:
            y_ref[i] = std::min(std::max(x_data[i], qmin()), qmax());
            break;
          default:
            GTEST_FAIL() << "Unexpected op type";
        }
      }

      // Prepare parameters.
      union {
        union xnn_u8_minmax_params minmax;
      } params;
      switch (op_type) {
        case OpType::Clamp:
          switch (variant) {
            case Variant::Native:
              xnn_init_u8_minmax_params(&params.minmax, qmin(), qmax());
              break;
            case Variant::Scalar:
              xnn_init_scalar_u8_minmax_params(&params.minmax, qmin(), qmax());
              break;
          }
          break;
        default:
          GTEST_FAIL() << "Unexpected op type";
      }

      // Call optimized micro-kernel.
      vunary(batch_size() * sizeof(uint8_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << uint32_t(x[i]);
      }
    }
  }

  inline void Test(xnn_u8_vclamp_ukernel_function vunary, OpType op_type, Variant variant = Variant::Native) const {
    Test(xnn_u8_vunary_ukernel_function(vunary), op_type, variant);
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
