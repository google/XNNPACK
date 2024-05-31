// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams.h>
#include <xnnpack/requantization.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "replicable_random_device.h"
#include <gtest/gtest.h>
#include <fp16/fp16.h>

class RSumMicrokernelTester {
 public:
  RSumMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  RSumMicrokernelTester& scale(float scale) {
    this->scale_ = scale;
    return *this;
  }

  float scale() const {
    return this->scale_;
  }

  RSumMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  RSumMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  RSumMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const {
    return this->output_scale_;
  }

  RSumMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  RSumMicrokernelTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  void Test(xnn_qs8_rsum_ukernel_fn rsum,
      xnn_init_qs8_rsum_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });

      // Compute reference results.
      int32_t output_init = i8dist(rng);
      int32_t output_ref = output_init;
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref += int32_t(input[i]);
      }

      // Prepare parameters
      union xnn_qs8_rsum_params params;
      init_params(&params);

      // Call optimized micro-kernel.
      int32_t output = output_init;
      rsum(batch_size() * sizeof(int8_t), input.data(), &output, &params);

      // Verify results.
      EXPECT_EQ(output_ref, output);
    }
  }

  void Test(xnn_f16_rsum_ukernel_fn rsum, xnn_init_f16_scale_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

    std::vector<uint16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });

      // Compute reference results.
      float output_ref = 0.0f;
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref += fp16_ieee_to_fp32_value(input[i]);
      }
      output_ref *= scale();

      // Prepare parameters.
      xnn_f16_scale_params params;
      init_params(&params, fp16_ieee_from_fp32_value(scale()));

      // Call optimized micro-kernel.
      uint16_t output = UINT16_C(0x7E00);  /* NaN */
      rsum(batch_size() * sizeof(uint16_t), input.data(), &output, &params);

      // Verify results. For an explanation of the error bound, see
      // https://epubs.siam.org/doi/pdf/10.1137/18M1226312, Eq. 2.1. We use a
      // $\lambda$ of 5.0 which should only fail with a probability of 5.7e-5.
      // We don't use the usual hard bound $\gamma_n = \frac{nu}{1 - nu}$ since
      // for `fp16` $nu > 1$ already for $n > 2000$, rendering the bound
      // meaningless.
      const float fp16_u =
          4.88281e-4;  // Half of the ULP, max relative rounding error.
      const float n = batch_size() - 1;
      const float gamma_n = std::expm1(5.0f * std::sqrt(n) * fp16_u +
                                       n * fp16_u * fp16_u / (1.0f - fp16_u));
      const float max_abs_err =
          gamma_n * output_ref;  // The random inputs are all positive, so we
                                 // can just use the "exact" result here.
      EXPECT_NEAR(fp16_ieee_to_fp32_value(output), output_ref, max_abs_err)
          << "with batch " << batch_size() << ", scale " << scale();
    }
  }

  void Test(xnn_f16_f32acc_rsum_ukernel_fn rsum, xnn_init_f16_f32acc_scale_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

    std::vector<uint16_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });

      // Compute reference results.
      float output_ref = 0.0f;
      for (size_t i = 0; i < batch_size(); i++) {
        output_ref += fp16_ieee_to_fp32_value(input[i]);
      }
      output_ref *= scale();

      // Prepare parameters.
      xnn_f16_f32acc_scale_params params;
      init_params(&params, scale());

      // Call optimized micro-kernel.
      uint16_t output = fp16_ieee_from_fp32_value(0.f);
      rsum(batch_size() * sizeof(uint16_t), input.data(), &output, &params);

      // Verify results.
      EXPECT_NEAR(fp16_ieee_to_fp32_value(output), output_ref, std::abs(output_ref) * 1.0e-3f)
        << "with batch " << batch_size() << ", scale " << scale();
    }
  }

  void Test(xnn_f32_rsum_ukernel_fn rsum, xnn_init_f32_scale_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

    std::vector<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      const double output_ref = std::accumulate(input.begin(), input.begin() + batch_size(), 0.0) * double(scale());

      // Prepare parameters.
      xnn_f32_scale_params params;
      init_params(&params, scale());

      // Call optimized micro-kernel.
      float output = 0.f;
      rsum(batch_size() * sizeof(float), input.data(), &output, &params);

      // Verify results.
      EXPECT_NEAR(output, output_ref, std::abs(output_ref) * 1.0e-6f)
        << "with batch " << batch_size() << ", scale " << scale();
    }
  }

 private:
  size_t batch_size_{1};
  float scale_{1.0f};
  size_t iterations_{15};
  float input_scale_{1.25f};
  float output_scale_{0.75f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
};
