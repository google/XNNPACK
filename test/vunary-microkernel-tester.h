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
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class VUnOpMicrokernelTester {
 public:
  enum class OpType {
    Abs,
    ELU,
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

  inline VUnOpMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VUnOpMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VUnOpMicrokernelTester& slope(float slope) {
    this->slope_ = slope;
    return *this;
  }

  inline float slope() const {
    return this->slope_;
  }

  inline VUnOpMicrokernelTester& prescale(float prescale) {
    this->prescale_ = prescale;
    return *this;
  }

  inline float prescale() const {
    return this->prescale_;
  }

  inline VUnOpMicrokernelTester& alpha(float alpha) {
    this->alpha_ = alpha;
    return *this;
  }

  inline float alpha() const {
    return this->alpha_;
  }

  inline VUnOpMicrokernelTester& beta(float beta) {
    this->beta_ = beta;
    return *this;
  }

  inline float beta() const {
    return this->beta_;
  }

  inline VUnOpMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VUnOpMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VUnOpMicrokernelTester& iterations(size_t iterations) {
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
      case OpType::ELU:
        distribution = std::uniform_real_distribution<float>(-20.0f, 20.0f);
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
          case OpType::ELU:
          {
            y_ref[i] = std::signbit(x_data[i]) ? alpha() * std::expm1(double(x_data[i]) * prescale()) : double(x_data[i]) * beta();
            break;
          }
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
        union xnn_f32_relu_params relu;
        union xnn_f32_lrelu_params lrelu;
        union xnn_f32_neg_params neg;
        union xnn_f32_rnd_params rnd;
        union xnn_f32_sqrt_params sqrt;
      } params;
      switch (op_type) {
        case OpType::Abs:
          switch (variant) {
            case Variant::Native:
              params.abs = xnn_init_f32_abs_params();
              break;
            case Variant::Scalar:
              params.abs = xnn_init_scalar_f32_abs_params();
              break;
          }
          break;
        case OpType::ELU:
          switch (variant) {
            case Variant::Native:
              params.elu = xnn_init_f32_elu_params(prescale(), alpha(), beta());
              break;
            case Variant::Scalar:
              params.elu = xnn_init_scalar_f32_elu_params(prescale(), alpha(), beta());
              break;
          }
          break;
        case OpType::LeakyReLU:
          switch (variant) {
            case Variant::Native:
              params.lrelu = xnn_init_f32_lrelu_params(slope());
              break;
            case Variant::Scalar:
              params.lrelu = xnn_init_scalar_f32_lrelu_params(slope());
              break;
          }
          break;
        case OpType::Negate:
          switch (variant) {
            case Variant::Native:
              params.neg = xnn_init_f32_neg_params();
              break;
            case Variant::Scalar:
              params.neg = xnn_init_scalar_f32_neg_params();
              break;
          }
          break;
        case OpType::RoundToNearestEven:
        case OpType::RoundTowardsZero:
        case OpType::RoundUp:
        case OpType::RoundDown:
          switch (variant) {
            case Variant::Native:
              params.rnd = xnn_init_f32_rnd_params();
              break;
            case Variant::Scalar:
              params.rnd = xnn_init_scalar_f32_rnd_params();
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
              params.sqrt = xnn_init_f32_sqrt_params();
              break;
            case Variant::Scalar:
              params.sqrt = xnn_init_scalar_f32_sqrt_params();
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
