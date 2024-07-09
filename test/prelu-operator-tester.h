// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/cache.h"
#include "replicable_random_device.h"

class PReLUOperatorTester {
 public:
  enum class WeightsType {
    Default,
    FP32,
  };

  PReLUOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  PReLUOperatorTester& input_channels(size_t input_channels) {
    assert(input_channels != 0);
    this->input_channels_ = input_channels;
    return *this;
  }

  size_t input_channels() const {
    return this->input_channels_;
  }

  PReLUOperatorTester& slope_channels(size_t slope_channels) {
    assert(slope_channels != 0);
    this->slope_channels_ = slope_channels;
    return *this;
  }

  size_t slope_channels() const {
    if (this->slope_channels_ == 0) {
      return this->input_channels_;
    } else {
      return this->slope_channels_;
    }
  }

  PReLUOperatorTester& x_stride(size_t x_stride) {
    assert(x_stride != 0);
    this->x_stride_ = x_stride;
    return *this;
  }

  size_t x_stride() const {
    if (this->x_stride_ == 0) {
      return this->input_channels_;
    } else {
      assert(this->x_stride_ >= this->input_channels_);
      return this->x_stride_;
    }
  }

  PReLUOperatorTester& y_stride(size_t y_stride) {
    assert(y_stride != 0);
    this->y_stride_ = y_stride;
    return *this;
  }

  size_t y_stride() const {
    if (this->y_stride_ == 0) {
      return this->input_channels_;
    } else {
      assert(this->y_stride_ >= this->input_channels_);
      return this->y_stride_;
    }
  }

  PReLUOperatorTester& weights_type(WeightsType weights_type) {
    this->weights_type_ = weights_type;
    return *this;
  }

  WeightsType weights_type() const {
    return this->weights_type_;
  }

  PReLUOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  PReLUOperatorTester& use_weights_cache(bool use_weights_cache) {
    this->use_weights_cache_ = use_weights_cache;
    return *this;
  }

  bool use_weights_cache() const {
    return this->use_weights_cache_;
  }

  void TestF16() const {
    switch (weights_type()) {
      case WeightsType::Default:
        break;
      case WeightsType::FP32:
        break;
      default:
        GTEST_FAIL() << "unexpected weights type";
    }

    xnnpack::ReplicableRandomDevice rng;
    auto f32irng = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    auto f32wrng = std::uniform_real_distribution<float>(0.25f, 0.75f);

    std::vector<uint16_t> x((batch_size() - 1) * x_stride() + input_channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> w(input_channels());
    std::vector<float> w_as_float(input_channels());
    std::vector<uint16_t> y((batch_size() - 1) * y_stride() + input_channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<float> y_ref(batch_size() * input_channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&] { return fp16_ieee_from_fp32_value(f32irng(rng)); });
      if (slope_channels() == 1) {
        std::fill(w.begin(), w.end(), fp16_ieee_from_fp32_value(f32wrng(rng)));
      } else {
        std::generate(w.begin(), w.end(), [&] { return fp16_ieee_from_fp32_value(f32wrng(rng)); });
      }
      std::transform(w.cbegin(), w.cend(), w_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < input_channels(); c++) {
          const float x_value = fp16_ieee_to_fp32_value(x[i * x_stride() + c]);
          const float w_value = w_as_float[c];
          y_ref[i * input_channels() + c] = std::signbit(x_value) ? x_value * w_value : x_value;
        }
      }

      // Create, setup, run, and destroy PReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t prelu_op = nullptr;

      struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      const void* negative_slope_data = w.data();
      if (weights_type() == WeightsType::FP32) {
        negative_slope_data = w_as_float.data();
      }
      uint32_t flags = 0;
      if (weights_type() == WeightsType::FP32) {
        flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
      }
      ASSERT_EQ(xnn_status_success,
        xnn_create_prelu_nc_f16(
          input_channels(), slope_channels(), x_stride(), y_stride(),
          negative_slope_data,
          flags, /*code_cache=*/nullptr, auto_weights_cache.get(), &prelu_op));
      ASSERT_NE(nullptr, prelu_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete prelu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_prelu_op(prelu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_prelu_nc_f16(
          prelu_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_prelu_nc_f16(
          prelu_op,
          x.data(), y.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(prelu_op, /*threadpool=*/nullptr));

      VerifyF16(y, y_ref);

      if (use_weights_cache()) {
        xnn_operator_t prelu_op2 = nullptr;
        const size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_prelu_nc_f16(
                      input_channels(), slope_channels(), x_stride(), y_stride(),
                      negative_slope_data,
                      flags, /*code_cache=*/nullptr, auto_weights_cache.get(), &prelu_op2));
        ASSERT_NE(nullptr, prelu_op2);

        // Smart pointer to automatically delete prelu_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_prelu_op(prelu_op2, xnn_delete_operator);

        std::vector<uint16_t> y2(y.size(), UINT16_C(0x7E00) /* NaN */);
        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_prelu_nc_f16(
                      prelu_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_prelu_nc_f16(
                      prelu_op2,
                      x.data(), y2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(prelu_op2, /*threadpool=*/nullptr));

        VerifyF16(y2, y_ref);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyF16(const std::vector<uint16_t>& y, const std::vector<float>& y_ref) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < input_channels(); c++) {
        ASSERT_NEAR(
            fp16_ieee_to_fp32_value(y[i * y_stride() + c]),
            y_ref[i * input_channels() + c],
            std::max(1.0e-4f, std::abs(y_ref[i * input_channels() + c]) * 1.0e-3f))
            << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << input_channels();
      }
    }
  }

  void TestF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    auto f32irng = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    auto f32wrng = std::uniform_real_distribution<float>(0.25f, 0.75f);

    std::vector<float> x((batch_size() - 1) * x_stride() + input_channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> w(input_channels());
    std::vector<float> y((batch_size() - 1) * y_stride() + input_channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y_ref(batch_size() * input_channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&] { return f32irng(rng);} );
      if (slope_channels() == 1) {
        std::fill(w.begin(), w.end(), f32wrng(rng));
      } else {
        std::generate(w.begin(), w.end(), [&] { return f32wrng(rng);} );
      }
      std::fill(y.begin(), y.end(), nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < input_channels(); c++) {
          y_ref[i * input_channels() + c] = std::signbit(x[i * x_stride() + c]) ? x[i * x_stride() + c] * w[c] : x[i * x_stride() + c];
        }
      }

      // Create, setup, run, and destroy PReLU operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t prelu_op = nullptr;

     struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
      std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
        nullptr, xnn_delete_weights_cache);
      if (use_weights_cache()) {
        xnn_weights_cache_t weights_cache = nullptr;
        xnn_create_weights_cache(&weights_cache);
        auto_weights_cache.reset(weights_cache);
        if (weights_cache) {
          internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
        }
      }

      ASSERT_EQ(xnn_status_success,
        xnn_create_prelu_nc_f32(
          input_channels(), slope_channels(), x_stride(), y_stride(),
          w.data(),
          0, /*code_cache=*/nullptr, auto_weights_cache.get(), &prelu_op));
      ASSERT_NE(nullptr, prelu_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete prelu_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_prelu_op(prelu_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_prelu_nc_f32(
          prelu_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_prelu_nc_f32(
          prelu_op,
          x.data(), y.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(prelu_op, /*threadpool=*/nullptr));

      VerifyF32(y, y_ref);

      if (use_weights_cache()) {
        xnn_operator_t prelu_op2 = nullptr;
        const size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_prelu_nc_f32(
                      input_channels(), slope_channels(), x_stride(), y_stride(),
                      w.data(),
                      0, /*code_cache=*/nullptr, auto_weights_cache.get(), &prelu_op2));
        ASSERT_NE(nullptr, prelu_op2);

        // Smart pointer to automatically delete prelu_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_prelu_op(prelu_op2, xnn_delete_operator);
        std::vector<float> y2(y.size(), nanf(""));

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_prelu_nc_f32(
                      prelu_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_prelu_nc_f32(
                      prelu_op2,
                      x.data(), y2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(prelu_op2, /*threadpool=*/nullptr));

        VerifyF32(y, y_ref);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyF32(const std::vector<float>& y, const std::vector<float>& y_ref) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < input_channels(); c++) {
        ASSERT_NEAR(
            y[i * y_stride() + c],
            y_ref[i * input_channels() + c],
            std::max(1.0e-6f, std::abs(y_ref[i * input_channels() + c]) * 1.0e-6f))
          << "at position " << i << " / " << batch_size() << ", channel " << c << " / " << input_channels();
      }
    }
  }

  void VerifyWeightsCache(const xnn_internal_weights_cache& weights_cache, size_t old_size) const {
    ASSERT_EQ(weights_cache.cache.hits, 1);
    // Ensure that we did not write more weights to the cache because it was a cache hit.
    ASSERT_EQ(old_size, weights_cache.cache.weights.size);
  };

 private:
  size_t batch_size_{1};
  size_t input_channels_{1};
  size_t slope_channels_{0};
  size_t x_stride_{0};
  size_t y_stride_{0};
  WeightsType weights_type_{WeightsType::Default};
  bool use_weights_cache_{false};
  size_t iterations_{15};
};
