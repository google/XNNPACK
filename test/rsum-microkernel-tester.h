// Copyright 2023 Google LLC
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
#include <numeric>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


class RSumMicrokernelTester {
 public:
  inline RSumMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline RSumMicrokernelTester& scale(float scale) {
    this->scale_ = scale;
    return *this;
  }

  inline float scale() const {
    return this->scale_;
  }

  inline RSumMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_rsum_ukernel_fn rsum, xnn_init_f16_scale_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
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

      // Verify results.
      EXPECT_NEAR(fp16_ieee_to_fp32_value(output), output_ref, std::abs(output_ref) * 2.0e-3f)
        << "with batch " << batch_size() << ", scale " << scale();
    }
  }

  void Test(xnn_f16_f32acc_rsum_ukernel_fn rsum, xnn_init_f16_f32acc_scale_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
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
      uint16_t output = UINT16_C(0x7E00);  /* NaN */
      rsum(batch_size() * sizeof(uint16_t), input.data(), &output, &params);

      // Verify results.
      EXPECT_NEAR(fp16_ieee_to_fp32_value(output), output_ref, std::abs(output_ref) * 1.0e-3f)
        << "with batch " << batch_size() << ", scale " << scale();
    }
  }

  void Test(xnn_f32_rsum_ukernel_fn rsum, xnn_init_f32_scale_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
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
      float output = std::nanf("");
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
};
