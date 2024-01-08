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
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/cache.h>
#include <xnnpack/quantization.h>


class FullyConnectedOperatorTester {
 public:
  enum class WeightsType {
    Default,
    FP32,
  };

  inline FullyConnectedOperatorTester& input_channels(size_t input_channels) {
    assert(input_channels >= 1);
    this->input_channels_ = input_channels;
    return *this;
  }

  inline size_t input_channels() const {
    return this->input_channels_;
  }

  inline FullyConnectedOperatorTester& output_channels(size_t output_channels) {
    assert(output_channels >= 1);
    this->output_channels_ = output_channels;
    return *this;
  }

  inline size_t output_channels() const {
    return this->output_channels_;
  }

  inline FullyConnectedOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline FullyConnectedOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride >= 1);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return input_channels();
    } else {
      assert(this->input_stride_ >= input_channels());
      return this->input_stride_;
    }
  }

  inline FullyConnectedOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride >= 1);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return output_channels();
    } else {
      assert(this->output_stride_ >= output_channels());
      return this->output_stride_;
    }
  }

  inline FullyConnectedOperatorTester& input_zero_point(size_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline size_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline FullyConnectedOperatorTester& kernel_zero_point(size_t kernel_zero_point) {
    this->kernel_zero_point_ = kernel_zero_point;
    return *this;
  }

  inline size_t kernel_zero_point() const {
    return this->kernel_zero_point_;
  }

  inline FullyConnectedOperatorTester& output_zero_point(size_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline size_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline FullyConnectedOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline FullyConnectedOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline FullyConnectedOperatorTester& transpose_weights(bool transpose_weights) {
    this->transpose_weights_ = transpose_weights;
    return *this;
  }

  inline bool transpose_weights() const {
    return this->transpose_weights_;
  }

  inline FullyConnectedOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  inline bool has_bias() const {
    return this->has_bias_;
  }

  inline FullyConnectedOperatorTester& weights_type(WeightsType weights_type) {
    this->weights_type_ = weights_type;
    return *this;
  }

  inline WeightsType weights_type() const {
    return this->weights_type_;
  }

  inline FullyConnectedOperatorTester& use_weights_cache(bool use_weights_cache) {
    this->use_weights_cache_ = use_weights_cache;
    return *this;
  }

  inline bool use_weights_cache() const {
    return this->use_weights_cache_;
  }

#if XNN_PLATFORM_JIT
  inline FullyConnectedOperatorTester& use_jit(bool use_jit) {
    this->use_jit_ = use_jit;
    return *this;
  }

  inline bool use_jit() const {
    return this->use_jit_;
  }
#endif

  inline FullyConnectedOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  size_t calc_kernel_stride() const {
    if (transpose_weights()) {
      return (output_channels() + 1) / 2;
    } else {
      return (input_channels() + 1) / 2;
    }
  }

  void TestQD8F16QC4W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
    std::uniform_real_distribution<float> f32idist(0.5f, 2.0f);
    std::uniform_int_distribution<int32_t> w8dist(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());
    // Weights typically have a Gaussian distrubution centred on zero. A
    // standard deviation of 40 means that > 99.99% of values fall in the
    // -127->127 range. However, the reduce the chance of overflow, we constrain
    // the weights further.
    std::normal_distribution<float> normal_dist{0.f, 10.f};


    const size_t k2 =  round_up_po2(input_channels(), 2);  // tester assumes byte aligned rows

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    const size_t kernel_stride = calc_kernel_stride();
    std::vector<uint8_t> kernel((transpose_weights() ? k2 : output_channels()) * kernel_stride);
    std::vector<float> bias(output_channels());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::vector<float> kernel_scale(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return (int8_t) normal_dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(kernel_scale.begin(), kernel_scale.end(), [&]() { return f32idist(rng); });
      std::generate(quantization_params.begin(), quantization_params.end(), [&]() { return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)}; });
      for (size_t i = batch_size(); i < batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS; ++i) {
        quantization_params[i].zero_point = quantization_params[batch_size() - 1].zero_point;
        quantization_params[i].inv_scale = quantization_params[batch_size() - 1].inv_scale;
      }
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));

      // Compute reference results, without renormalization.
      std::fill(output_ref.begin(), output_ref.end(), 0);

      if (transpose_weights()) {
        for (size_t mi = 0; mi < batch_size(); mi++) {
          for (size_t ni = 0; ni < output_channels(); ni++) {
            for (size_t ki = 0; ki < input_channels(); ki++) {
              const size_t kernel_index = ki * kernel_stride + (ni / 2);
              const int8_t kernel_value =
                int8_t((ni % 2 == 0) ? (kernel[kernel_index] & 15) : (kernel[kernel_index] >> 4)) - kernel_zero_point();
              output_ref[mi * output_channels() + ni] +=
                  (int32_t(input[mi * input_stride() + ki]) - quantization_params[mi].zero_point) *
                  static_cast<float>(static_cast<int32_t>(kernel_value));
            }
            output_ref[mi * output_channels() + ni] *= quantization_params[mi].inv_scale * kernel_scale[ni];
            if (has_bias()) {
              output_ref[mi * output_channels() + ni] += bias[ni];
            }
          }
        }

      } else {
        for (size_t mi = 0; mi < batch_size(); mi++) {
          for (size_t ni = 0; ni < output_channels(); ni++) {
            for (size_t ki = 0; ki < input_channels(); ki++) {
              const size_t kernel_index = ni * kernel_stride + (ki / 2);
              const int8_t kernel_value =
                int8_t((ki % 2 == 0) ? (kernel[kernel_index] & 15) : (kernel[kernel_index] >> 4)) - kernel_zero_point();
              output_ref[mi * output_channels() + ni] +=
                  (int32_t(input[mi * input_stride() + ki]) - quantization_params[mi].zero_point) *
                  static_cast<float>(static_cast<int32_t>(kernel_value));
            }
            output_ref[mi * output_channels() + ni] *= quantization_params[mi].inv_scale * kernel_scale[ni];
            if (has_bias()) {
              output_ref[mi * output_channels() + ni] += bias[ni];
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      // Clamp reference results.
      for (size_t i = 0; i < output_ref.size(); ++i) {
        output_ref[i] = std::max(std::min(output_ref[i], output_max), output_min);
        output_ref[i] = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_ref[i]));
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qd8_f16_qc4w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel_zero_point(),
          kernel_scale.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }

      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qd8_f16_qc4w(
          fully_connected_op,
          input.data(), output.data(),
          reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyF16(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success, xnn_create_fully_connected_nc_qd8_f16_qc4w(
            input_channels(), output_channels(),
            input_stride(), output_stride(),
            kernel_zero_point(),
            kernel_scale.data(),
            kernel.data(), has_bias() ? bias.data() : nullptr,
            output_min, output_max,
            transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
            nullptr, auto_weights_cache.get(),
            &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
          xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
            fully_connected_op2,
            batch_size(),
            /*threadpool=*/nullptr));

        std::vector<float> output2(output.size(), nanf(""));
        ASSERT_EQ(xnn_status_success,
          xnn_setup_fully_connected_nc_qd8_f16_qc4w(
            fully_connected_op2,
            input.data(), output2.data(),
            reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));


        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF16(output, output_ref, output_max, output_min);
      }
    }
  }

  void TestQD8F32QC4W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
    std::uniform_real_distribution<float> f32idist(0.5f, 2.0f);
    std::uniform_int_distribution<int32_t> w8dist(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    const size_t k2 =  round_up_po2(input_channels(), 2);  // tester assumes byte aligned rows

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    const size_t kernel_stride = calc_kernel_stride();
    std::vector<uint8_t> kernel((transpose_weights() ? k2 : output_channels()) * kernel_stride);
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::vector<float> kernel_scale(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(kernel_scale.begin(), kernel_scale.end(), [&]() { return f32idist(rng); });
      std::generate(quantization_params.begin(), quantization_params.end(), [&]() { return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)}; });
      for (size_t i = batch_size(); i < batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS; ++i) {
        quantization_params[i].zero_point = quantization_params[batch_size() - 1].zero_point;
        quantization_params[i].inv_scale = quantization_params[batch_size() - 1].inv_scale;
      }
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without renormalization.
      std::fill(output_ref.begin(), output_ref.end(), 0);

      if (transpose_weights()) {
        for (size_t mi = 0; mi < batch_size(); mi++) {
          for (size_t ni = 0; ni < output_channels(); ni++) {
            for (size_t ki = 0; ki < input_channels(); ki++) {
              const size_t kernel_index = ki * kernel_stride + (ni / 2);
              const int8_t kernel_value =
                int8_t((ni % 2 == 0) ? (kernel[kernel_index] & 15) : (kernel[kernel_index] >> 4)) - kernel_zero_point();
              output_ref[mi * output_channels() + ni] +=
                  (int32_t(input[mi * input_stride() + ki]) - quantization_params[mi].zero_point) *
                  static_cast<float>(static_cast<int32_t>(kernel_value));
            }
            output_ref[mi * output_channels() + ni] *= quantization_params[mi].inv_scale * kernel_scale[ni];
            if (has_bias()) {
              output_ref[mi * output_channels() + ni] += bias[ni];
            }
          }
        }

      } else {
        for (size_t mi = 0; mi < batch_size(); mi++) {
          for (size_t ni = 0; ni < output_channels(); ni++) {
            for (size_t ki = 0; ki < input_channels(); ki++) {
              const size_t kernel_index = ni * kernel_stride + (ki / 2);
              const int8_t kernel_value =
                int8_t((ki % 2 == 0) ? (kernel[kernel_index] & 15) : (kernel[kernel_index] >> 4)) - kernel_zero_point();
              output_ref[mi * output_channels() + ni] +=
                  (int32_t(input[mi * input_stride() + ki]) - quantization_params[mi].zero_point) *
                  static_cast<float>(static_cast<int32_t>(kernel_value));
            }
            output_ref[mi * output_channels() + ni] *= quantization_params[mi].inv_scale * kernel_scale[ni];
            if (has_bias()) {
              output_ref[mi * output_channels() + ni] += bias[ni];
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel_zero_point(),
          kernel_scale.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }

      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qd8_f32_qc4w(
          fully_connected_op,
          input.data(), output.data(),
          reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success, xnn_create_fully_connected_nc_qd8_f32_qc4w(
            input_channels(), output_channels(),
            input_stride(), output_stride(),
            kernel_zero_point(),
            kernel_scale.data(),
            kernel.data(), has_bias() ? bias.data() : nullptr,
            output_min, output_max,
            transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
            nullptr, auto_weights_cache.get(),
            &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
          xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
            fully_connected_op2,
            batch_size(),
            /*threadpool=*/nullptr));

        std::vector<float> output2(output.size(), nanf(""));
        ASSERT_EQ(xnn_status_success,
          xnn_setup_fully_connected_nc_qd8_f32_qc4w(
            fully_connected_op2,
            input.data(), output2.data(),
            reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));


        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF32(output, output_ref, output_max, output_min);
      }
    }
  }

  void TestQD8F16QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> f32idist(0.1f, 1.0f);
    std::uniform_int_distribution<int32_t> w8dist(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());
    // Weights typically have a Gaussian distrubution centred on zero. A
    // standard deviation of 40 means that > 99.99% of values fall in the
    // -127->127 range. However, the reduce the chance of overflow, we constrain
    // the weights further.
    std::normal_distribution<float> normal_dist{0.f, 10.f};

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<int8_t> kernel(output_channels() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::vector<float> kernel_scale(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return (int8_t) normal_dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(kernel_scale.begin(), kernel_scale.end(), [&]() { return f32idist(rng); });
      std::generate(quantization_params.begin(), quantization_params.end(), [&]() { return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)}; });
      for (size_t i = batch_size(); i < batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS; ++i) {
        quantization_params[i].zero_point = quantization_params[batch_size() - 1].zero_point;
        quantization_params[i].inv_scale = quantization_params[batch_size() - 1].inv_scale;
      }
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));

      // Compute reference results, without renormalization.
      std::fill(output_ref.begin(), output_ref.end(), 0);
      if (transpose_weights()) {
        for (size_t m_index = 0; m_index < batch_size(); m_index++) {
          for (size_t n_index = 0; n_index < output_channels(); n_index++) {
            for (size_t k_index = 0; k_index < input_channels(); k_index++) {
              output_ref[m_index * output_channels() + n_index] +=
                  (int32_t(input[m_index * input_stride() + k_index]) - quantization_params[m_index].zero_point) *
                  int32_t(kernel[k_index * output_channels() + n_index]);
            }
            output_ref[m_index * output_channels() + n_index] *= quantization_params[m_index].inv_scale  * kernel_scale[n_index];
            if (has_bias()) {
              output_ref[m_index * output_channels() + n_index] += bias[n_index];
            }
          }
        }
      } else {
        for (size_t m_index = 0; m_index < batch_size(); m_index++) {
          for (size_t n_index = 0; n_index < output_channels(); n_index++) {
            for (size_t k_index = 0; k_index < input_channels(); k_index++) {
              output_ref[m_index * output_channels() + n_index] +=
                  (int32_t(input[m_index * input_stride() + k_index]) - quantization_params[m_index].zero_point) *
                  int32_t(kernel[n_index * input_channels() + k_index]);
            }
            output_ref[m_index * output_channels() + n_index] *= quantization_params[m_index].inv_scale * kernel_scale[n_index];
            if (has_bias()) {
              output_ref[m_index * output_channels() + n_index] += bias[n_index];
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      // Clamp reference results.
      for (size_t i = 0; i < output_ref.size(); ++i) {
        output_ref[i] = std::max(std::min(output_ref[i], output_max), output_min);
        output_ref[i] = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_ref[i]));
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qd8_f16_qc8w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel_scale.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qd8_f16_qc8w(
          fully_connected_op,
          input.data(), output.data(),
          reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyF16(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success, xnn_create_fully_connected_nc_qd8_f16_qc8w(
            input_channels(), output_channels(),
            input_stride(), output_stride(),
            kernel_scale.data(),
            kernel.data(), has_bias() ? bias.data() : nullptr,
            output_min, output_max,
            transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
            nullptr, auto_weights_cache.get(),
            &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
          xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
            fully_connected_op2,
            batch_size(),
            /*threadpool=*/nullptr));

        std::vector<uint16_t> output2(output.size(), UINT16_C(0xDEAD));
        ASSERT_EQ(xnn_status_success,
          xnn_setup_fully_connected_nc_qd8_f16_qc8w(
            fully_connected_op2,
            input.data(), output2.data(),
            reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));


        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF16(output, output_ref, output_max, output_min);
      }
    }
  }

  void TestQD8F32QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
    std::uniform_real_distribution<float> f32idist(0.5f, 2.0f);
    std::uniform_int_distribution<int32_t> w8dist(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<int8_t> kernel(output_channels() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::vector<float> kernel_scale(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(kernel_scale.begin(), kernel_scale.end(), [&]() { return f32idist(rng); });
      std::generate(quantization_params.begin(), quantization_params.end(), [&]() { return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)}; });
      for (size_t i = batch_size(); i < batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS; ++i) {
        quantization_params[i].zero_point = quantization_params[batch_size() - 1].zero_point;
        quantization_params[i].inv_scale = quantization_params[batch_size() - 1].inv_scale;
      }
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without renormalization.
      std::fill(output_ref.begin(), output_ref.end(), 0);
      if (transpose_weights()) {
        for (size_t m_index = 0; m_index < batch_size(); m_index++) {
          for (size_t n_index = 0; n_index < output_channels(); n_index++) {
            for (size_t k_index = 0; k_index < input_channels(); k_index++) {
              output_ref[m_index * output_channels() + n_index] +=
                  (int32_t(input[m_index * input_stride() + k_index]) - quantization_params[m_index].zero_point) *
                  int32_t(kernel[k_index * output_channels() + n_index]);
            }
            output_ref[m_index * output_channels() + n_index] *= quantization_params[m_index].inv_scale  * kernel_scale[n_index];
            if (has_bias()) {
              output_ref[m_index * output_channels() + n_index] += bias[n_index];
            }
          }
        }
      } else {
        for (size_t m_index = 0; m_index < batch_size(); m_index++) {
          for (size_t n_index = 0; n_index < output_channels(); n_index++) {
            for (size_t k_index = 0; k_index < input_channels(); k_index++) {
              output_ref[m_index * output_channels() + n_index] +=
                  (int32_t(input[m_index * input_stride() + k_index]) - quantization_params[m_index].zero_point) *
                  int32_t(kernel[n_index * input_channels() + k_index]);
            }
            output_ref[m_index * output_channels() + n_index] *= quantization_params[m_index].inv_scale * kernel_scale[n_index];
            if (has_bias()) {
              output_ref[m_index * output_channels() + n_index] += bias[n_index];
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qd8_f32_qc8w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel_scale.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qd8_f32_qc8w(
          fully_connected_op,
          input.data(), output.data(),
          reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success, xnn_create_fully_connected_nc_qd8_f32_qc8w(
            input_channels(), output_channels(),
            input_stride(), output_stride(),
            kernel_scale.data(),
            kernel.data(), has_bias() ? bias.data() : nullptr,
            output_min, output_max,
            transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
            nullptr, auto_weights_cache.get(),
            &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
          xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
            fully_connected_op2,
            batch_size(),
            /*threadpool=*/nullptr));

        std::vector<float> output2(output.size(), nanf(""));
        ASSERT_EQ(xnn_status_success,
          xnn_setup_fully_connected_nc_qd8_f32_qc8w(
            fully_connected_op2,
            input.data(), output2.data(),
            reinterpret_cast<const struct xnn_dynamic_quantization_params*>(quantization_params.data())));


        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF32(output, output_ref, output_max, output_min);
      }
    }
  }

  void TestQS8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<int8_t> kernel(output_channels() * input_channels());
    std::vector<int32_t> bias(output_channels());
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_channels());
    std::vector<double> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            accumulators[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[ic * output_channels() + oc]);
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[oc * input_channels() + ic]);
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const int8_t output_zero_point = int8_t(std::max(std::min(
        lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax() - 0x80) - output_zero_point), double(qmin() - 0x80) - output_zero_point);
        });

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qs8(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          int8_t(input_zero_point() - 0x80), /*input_scale=*/1.0f,
          /*kernel_scale=*/1.0f,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qs8(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qs8(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyQS8(output, output_ref, double(output_zero_point));

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_qs8(
                      input_channels(), output_channels(), input_stride(),
                      output_stride(), int8_t(input_zero_point() - 0x80), /*input_scale=*/1.0f,
                      /*kernel_scale=*/1.0f, kernel.data(),
                      has_bias() ? bias.data() : nullptr, output_zero_point,
                      output_scale, int8_t(qmin() - 0x80),
                      int8_t(qmax() - 0x80),
                      transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
                      nullptr, auto_weights_cache.get(), &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);
        std::vector<int8_t> output2(output.size(), INT8_C(0xA5));

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_fully_connected_nc_qs8(
                      fully_connected_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_fully_connected_nc_qs8(
                      fully_connected_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyQS8(output, output_ref, double(output_zero_point));
      }
    }
  }

  void VerifyQS8(const std::vector<int8_t>& output,
                 const std::vector<double>& output_ref,
                 double output_zero_point) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        ASSERT_LE(int32_t(output[i * output_stride() + c]), int32_t(qmax() - 0x80))
            << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(int32_t(output[i * output_stride() + c]), int32_t(qmin() - 0x80))
            << "batch index = " << i << ", channel = " << c;
        EXPECT_NEAR(output_ref[i * output_channels() + c],
                    double(output[i * output_stride() + c]) - output_zero_point,
                    0.9)
            << "batch index = " << i << ", channel = " << c;
      }
    }
  }

  void TestQS8QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32idist(0.1f, 1.0f);
    std::uniform_real_distribution<float> f32wdist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<int8_t> kernel(output_channels() * input_channels());
    std::vector<int32_t> bias(output_channels());
    std::vector<int8_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_channels());
    std::vector<double> output_ref(batch_size() * output_channels());
    std::vector<float> requantization_scales(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      for (size_t oc = 0; oc < output_channels(); oc++) {
        // Make filter weights within the same output channel to use the same sign as the bias.
        // This ensures that no catastrophic cancellation occurs, but test covers both positive and negative outputs.
        if (bias[oc] < 0) {
          for (size_t ic = 0; ic < input_channels(); ic++) {
            kernel[oc * input_channels() + ic] = -std::abs(kernel[oc * input_channels() + ic]);
          }
        } else {
          for (size_t ic = 0; ic < input_channels(); ic++) {
            kernel[oc * input_channels() + ic] = std::abs(kernel[oc * input_channels() + ic]);
          }
        }
      }

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            accumulators[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[ic * output_channels() + oc]);
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[oc * input_channels() + ic]);
            }
          }
        }
      }

      // Compute renormalization parameters.
      for (size_t oc = 0; oc < output_channels(); oc++) {
        int32_t accumulated_min = accumulators[oc];
        int32_t accumulated_max = accumulators[oc];
        for (size_t i = 0; i < batch_size(); i++) {
          accumulated_min = std::min(accumulated_min, accumulators[i * output_channels() + oc]);
          accumulated_max = std::max(accumulated_max, accumulators[i * output_channels() + oc]);
        }

        float requantization_scale = 0x1.0p-32f;
        if (accumulated_max != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::max()) - int32_t(output_zero_point() - 0x80)) / float(accumulated_max));
        }
        if (accumulated_min != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::min()) - int32_t(output_zero_point() - 0x80)) / float(accumulated_min));
        }
        requantization_scale = std::min(requantization_scale, 0x1.FFFFFEp-1f);
        requantization_scales[oc] = requantization_scale;
      }

      // Renormalize reference results.
      for (size_t oc = 0; oc < output_channels(); oc++) {
        for (size_t i = 0; i < batch_size(); i++) {
          output_ref[i * output_channels() + oc] = double(int32_t(output_zero_point() - 0x80)) +
            double(accumulators[i * output_channels() + oc]) * double(requantization_scales[oc]);
        }
      }
      std::transform(output_ref.cbegin(), output_ref.cend(), output_ref.begin(),
        [this](double x) -> double {
          return std::max<double>(std::min<double>(x, double(qmax() - 0x80)), double(qmin() - 0x80));
        });

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qs8_qc8w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          int8_t(input_zero_point() - 0x80), /*input_scale=*/1.0f,
          requantization_scales.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          int8_t(output_zero_point() - 0x80), /*output_scale=*/1.0f, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qs8_qc8w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qs8_qc8w(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyQC8(output, output_ref);

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_qs8_qc8w(
                      input_channels(), output_channels(), input_stride(),
                      output_stride(), (int8_t) (input_zero_point() - 0x80), /*input_scale=*/1.0f,
                      requantization_scales.data(), kernel.data(),
                      has_bias() ? bias.data() : nullptr, int8_t(output_zero_point() - 0x80),
                      /*output_scale=*/1.0f, int8_t(qmin() - 0x80),
                      int8_t(qmax() - 0x80),
                      transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
                      nullptr, auto_weights_cache.get(), &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);
        std::vector<int8_t> output2(output.size(), INT8_C(0xA5));

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_fully_connected_nc_qs8_qc8w(
                      fully_connected_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_fully_connected_nc_qs8_qc8w(
                      fully_connected_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyQC8(output, output_ref);
      }
    }
  }

  void VerifyQC8(const std::vector<int8_t>& output,
                 const std::vector<double>& output_ref) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        ASSERT_LE(int32_t(output[i * output_stride() + c]), int32_t(qmax() - 0x80))
            << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(int32_t(output[i * output_stride() + c]), int32_t(qmin() - 0x80))
            << "batch index = " << i << ", channel = " << c;
        EXPECT_NEAR(output_ref[i * output_channels() + c], double(output[i * output_stride() + c]), 0.9)
            << "batch index = " << i << ", channel = " << c;
      }
    }
  }


  void TestQU8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<uint8_t> kernel(output_channels() * input_channels());
    std::vector<int32_t> bias(output_channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_channels());
    std::vector<double> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            accumulators[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point())) *
                (int32_t(kernel[ic * output_channels() + oc]) - int32_t(kernel_zero_point()));
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              accumulators[i * output_channels() + oc] +=
                (int32_t(input[i * input_stride() + ic]) - int32_t(input_zero_point())) *
                (int32_t(kernel[oc * input_channels() + ic]) - int32_t(kernel_zero_point()));
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const xnn_status status = xnn_create_fully_connected_nc_qu8(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          input_zero_point(), /*input_scale=*/1.0f,
          uint8_t(kernel_zero_point()), /*kernel_scale=*/1.0f,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_qu8(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_qu8(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      VerifyQU8(output, output_ref, double(output_zero_point));

      if (use_weights_cache()) {
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_qu8(
                      input_channels(), output_channels(), input_stride(),
                      output_stride(), input_zero_point(), /*input_scale=*/1.0f,
                      uint8_t(kernel_zero_point()), /*kernel_scale=*/1.0f, kernel.data(),
                      has_bias() ? bias.data() : nullptr, output_zero_point,
                      output_scale, qmin(), qmax(),
                      transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
                      nullptr, auto_weights_cache.get(), &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);
        std::vector<uint8_t> output2(output.size(), UINT8_C(0xA5));

        ASSERT_EQ(
          xnn_status_success,
          xnn_reshape_fully_connected_nc_qu8(fully_connected_op2, batch_size(), /*threadpool=*/nullptr));

        ASSERT_EQ(
          xnn_status_success, xnn_setup_fully_connected_nc_qu8(fully_connected_op2, input.data(), output2.data()));

        ASSERT_EQ(
            xnn_status_success,
            xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyQU8(output2, output_ref, double(output_zero_point));
      }

    }
  }

  void VerifyQU8(const std::vector<uint8_t>& output,
                 const std::vector<double>& output_ref,
                 double output_zero_point) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        ASSERT_LE(int32_t(output[i * output_stride() + c]), int32_t(qmax()))
            << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(int32_t(output[i * output_stride() + c]), int32_t(qmin()))
            << "batch index = " << i << ", channel = " << c;
        EXPECT_NEAR(output_ref[i * output_channels() + c],
                    double(output[i * output_stride() + c]) - output_zero_point,
                    0.9)
            << "batch index = " << i << ", channel = " << c;
      }
    }
  }

  void TestF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<float> kernel(output_channels() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                input[i * input_stride() + ic] * kernel[ic * output_channels() + oc];
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                input[i * input_stride() + ic] * kernel[oc * input_channels() + ic];
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

      std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache(
          nullptr, xnn_release_code_cache);
      #if XNN_PLATFORM_JIT
        xnn_code_cache code_cache;
        if (use_jit()) {
          xnn_init_code_cache(&code_cache);
          auto_code_cache.reset(&code_cache);
        }
      #endif
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

      const xnn_status status = xnn_create_fully_connected_nc_f32(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          auto_code_cache.get(), auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      #if XNN_PLATFORM_JIT
        if (use_jit()) {
          // Check that we actually generated code.
          ASSERT_GT(code_cache.cache.code.size, 0);
          xnn_finalize_code_memory(&code_cache.cache.code);
        }
      #endif

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_f32(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_f32(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // We already finalized the code cache, so create a new code cache if we are testing JIT.
        std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_inner_code_cache(
            nullptr, xnn_release_code_cache);
        #if XNN_PLATFORM_JIT
          xnn_code_cache inner_code_cache;
          if (use_jit()) {
            xnn_init_code_cache(&inner_code_cache);
            auto_inner_code_cache.reset(&inner_code_cache);
          }
        #endif
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_f32(
                      input_channels(), output_channels(), input_stride(),
                      output_stride(), kernel.data(),
                      has_bias() ? bias.data() : nullptr, output_min,
                      output_max,
                      transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
                      auto_inner_code_cache.get(), auto_weights_cache.get(),
                      &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        #if XNN_PLATFORM_JIT
          if (use_jit()) {
            // Check that we actually generated code.
            ASSERT_GT(inner_code_cache.cache.code.size, 0);
            xnn_finalize_code_memory(&inner_code_cache.cache.code);
          }
        #endif

        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_fully_connected_nc_f32(
                      fully_connected_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        std::vector<float> output2(output.size(), nanf(""));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_fully_connected_nc_f32(
                      fully_connected_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF32(output, output_ref, output_max, output_min);
      }
    }
  }

  void VerifyF32(const std::vector<float>& output,
                 const std::vector<float>& output_ref,
                 float output_max,
                 float output_min) const {
    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        ASSERT_LE(output[i * output_stride() + c], output_max)
            << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(output[i * output_stride() + c], output_min)
            << "batch index = " << i << ", channel = " << c;
        EXPECT_NEAR(output_ref[i * output_channels() + c],
                    output[i * output_stride() + c],
                    std::max(1.0e-4 * std::abs(output_ref[i * output_channels() + c]), 1.0e-4))
            << "batch index = " << i << ", channel = " << c;
      }
    }
  }

  void TestF32QC4W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32idist(0.1f, 1.0f);
    std::uniform_real_distribution<float> f32wdist(-1.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> i8wdist(-1, std::numeric_limits<uint8_t>::max());

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<uint8_t> kernel(output_channels() * ((input_channels() + 1) / 2));
    const size_t kernel_stride = (input_channels() + 1) / 2;
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());
    std::vector<float> scale(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32idist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return i8wdist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32wdist(rng); });
      std::generate(scale.begin(), scale.end(), [&]() { return f32idist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      if (input_channels() % 2 == 1) {
        // For odd number of input channels, ensure that last nibble is padded with 0.
        for (size_t oc = 0; oc < output_channels(); oc++) {
          const size_t nb_index = oc * kernel_stride + kernel_stride - 1;
          kernel[nb_index] &= 0xF;
        }
      }

      // Compute reference results.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }

      ASSERT_FALSE(transpose_weights()) << "F32-QC4W does not support transposed weights";

      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oc = 0; oc < output_channels(); oc++) {
          for (size_t ic = 0; ic < input_channels(); ic++) {
            const size_t kernel_index = oc * kernel_stride + (ic / 2);
            const int8_t kernel_value =
              int8_t((ic % 2 == 0) ? (kernel[kernel_index] & 15) : (kernel[kernel_index] >> 4)) - kernel_zero_point();

            output_ref[i * output_channels() + oc] +=
              input[i * input_stride() + ic] * static_cast<float>(static_cast<int32_t>(kernel_value));
          }
          output_ref[i * output_channels() + oc] *= scale[oc];
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

      std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache(
          nullptr, xnn_release_code_cache);
      #if XNN_PLATFORM_JIT
        xnn_code_cache code_cache;
        if (use_jit()) {
          xnn_init_code_cache(&code_cache);
          auto_code_cache.reset(&code_cache);
        }
      #endif
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

      const xnn_status status = xnn_create_fully_connected_nc_f32_qc4w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel_zero_point(), scale.data(), kernel.data(),
          has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          auto_code_cache.get(), auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      #if XNN_PLATFORM_JIT
        if (use_jit()) {
          // Check that we actually generated code.
          ASSERT_GT(code_cache.cache.code.size, 0);
          xnn_finalize_code_memory(&code_cache.cache.code);
        }
      #endif

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_f32_qc4w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_f32_qc4w(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // We already finalized the code cache, so create a new code cache if we are testing JIT.
        std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_inner_code_cache(
            nullptr, xnn_release_code_cache);
        #if XNN_PLATFORM_JIT
          xnn_code_cache inner_code_cache;
          if (use_jit()) {
            xnn_init_code_cache(&inner_code_cache);
            auto_inner_code_cache.reset(&inner_code_cache);
          }
        #endif
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_f32_qc4w(
                      input_channels(), output_channels(),
                      input_stride(), output_stride(),
                      kernel_zero_point(), scale.data(), kernel.data(),
                      has_bias() ? bias.data() : nullptr,
                      output_min,
                      output_max,
                      transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
                      auto_inner_code_cache.get(), auto_weights_cache.get(),
                      &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        #if XNN_PLATFORM_JIT
          if (use_jit()) {
            // Check that we actually generated code.
            ASSERT_GT(inner_code_cache.cache.code.size, 0);
            xnn_finalize_code_memory(&inner_code_cache.cache.code);
          }
        #endif

        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_fully_connected_nc_f32_qc4w(
                      fully_connected_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        std::vector<float> output2(output.size(), nanf(""));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_fully_connected_nc_f32_qc4w(
                      fully_connected_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF32(output, output_ref, output_max, output_min);
      }
    }
  }

  void TestF32QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32idist(0.1f, 1.0f);
    std::uniform_real_distribution<float> f32wdist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> i8wdist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<int8_t> kernel(output_channels() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());
    std::vector<float> scale(output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32idist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return i8wdist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32wdist(rng); });
      std::generate(scale.begin(), scale.end(), [&]() { return f32idist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t oc = 0; oc < output_channels(); oc++) {
        // Make filter weights within the same output channel to use the same sign as the bias.
        // This ensures that no catastrophic cancellation occurs, but test covers both positive and negative outputs.
        if (std::signbit(bias[oc])) {
          for (size_t ic = 0; ic < input_channels(); ic++) {
            kernel[oc * input_channels() + ic] = -std::abs(kernel[oc * input_channels() + ic]);
          }
        } else {
          for (size_t ic = 0; ic < input_channels(); ic++) {
            kernel[oc * input_channels() + ic] = std::abs(kernel[oc * input_channels() + ic]);
          }
        }
      }

      // Compute reference results.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                  input[i * input_stride() + ic] *
                  static_cast<float>(static_cast<int32_t>(kernel[ic * output_channels() + oc]));
            }
            output_ref[i * output_channels() + oc] *= scale[oc];
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                  input[i * input_stride() + ic] *
                  static_cast<float>(static_cast<int32_t>(kernel[oc * input_channels() + ic]));
            }
            output_ref[i * output_channels() + oc] *= scale[oc];
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

      std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_code_cache(
          nullptr, xnn_release_code_cache);
      #if XNN_PLATFORM_JIT
        xnn_code_cache code_cache;
        if (use_jit()) {
          xnn_init_code_cache(&code_cache);
          auto_code_cache.reset(&code_cache);
        }
      #endif
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

      const xnn_status status = xnn_create_fully_connected_nc_f32_qc8w(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          scale.data(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
          auto_code_cache.get(), auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      #if XNN_PLATFORM_JIT
        if (use_jit()) {
          // Check that we actually generated code.
          ASSERT_GT(code_cache.cache.code.size, 0);
          xnn_finalize_code_memory(&code_cache.cache.code);
        }
      #endif

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_f32_qc8w(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_f32_qc8w(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // We already finalized the code cache, so create a new code cache if we are testing JIT.
        std::unique_ptr<xnn_code_cache, decltype(&xnn_release_code_cache)> auto_inner_code_cache(
            nullptr, xnn_release_code_cache);
        #if XNN_PLATFORM_JIT
          xnn_code_cache inner_code_cache;
          if (use_jit()) {
            xnn_init_code_cache(&inner_code_cache);
            auto_inner_code_cache.reset(&inner_code_cache);
          }
        #endif
        // Create another operator with the same weights cache.
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_f32_qc8w(
                      input_channels(), output_channels(),
                      input_stride(), output_stride(),
                      scale.data(),
                      kernel.data(),
                      has_bias() ? bias.data() : nullptr,
                      output_min,
                      output_max,
                      transpose_weights() ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0,
                      auto_inner_code_cache.get(), auto_weights_cache.get(),
                      &fully_connected_op2));
        ASSERT_NE(nullptr, fully_connected_op2);

        #if XNN_PLATFORM_JIT
          if (use_jit()) {
            // Check that we actually generated code.
            ASSERT_GT(inner_code_cache.cache.code.size, 0);
            xnn_finalize_code_memory(&inner_code_cache.cache.code);
          }
        #endif

        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_fully_connected_nc_f32_qc8w(
                      fully_connected_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        std::vector<float> output2(output.size(), nanf(""));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_fully_connected_nc_f32_qc8w(
                      fully_connected_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);

        VerifyF32(output, output_ref, output_max, output_min);
      }
    }
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

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<uint16_t> kernel(output_channels() * input_channels());
    std::vector<float> kernel_as_float(kernel.size());
    std::vector<uint16_t> bias(output_channels());
    std::vector<float> bias_as_float(bias.size());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(kernel.cbegin(), kernel.cend(), kernel_as_float.begin(), fp16_ieee_to_fp32_value);
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(bias.cbegin(), bias.cend(), bias_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = fp16_ieee_to_fp32_value(bias[oc]);
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                fp16_ieee_to_fp32_value(input[i * input_stride() + ic]) * fp16_ieee_to_fp32_value(kernel[ic * output_channels() + oc]);
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                fp16_ieee_to_fp32_value(input[i * input_stride() + ic]) * fp16_ieee_to_fp32_value(kernel[oc * input_channels() + ic]);
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
      xnn_operator_t fully_connected_op = nullptr;

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

      const void* kernel_data = kernel.data();
      const void* bias_data = bias.data();
      if (weights_type() == WeightsType::FP32) {
        kernel_data = kernel_as_float.data();
        bias_data = bias_as_float.data();
      }
      uint32_t flags = 0;
      if (transpose_weights()) {
        flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
      }
      if (weights_type() == WeightsType::FP32) {
        flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
      }
      const xnn_status status = xnn_create_fully_connected_nc_f16(
          input_channels(), output_channels(),
          input_stride(), output_stride(),
          kernel_data, has_bias() ? bias_data : nullptr,
          output_min, output_max,
          flags,
          nullptr, auto_weights_cache.get(),
          &fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, fully_connected_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_fully_connected_nc_f16(
          fully_connected_op,
          batch_size(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_fully_connected_nc_f16(
          fully_connected_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(fully_connected_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyF16(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        xnn_operator_t fully_connected_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;
        ASSERT_EQ(xnn_status_success,
                  xnn_create_fully_connected_nc_f16(
                      input_channels(), output_channels(), input_stride(),
                      output_stride(), kernel_data,
                      has_bias() ? bias_data : nullptr, output_min, output_max,
                      flags, nullptr, auto_weights_cache.get(), &fully_connected_op2));
        if (status == xnn_status_unsupported_hardware) {
          GTEST_SKIP();
        }
        ASSERT_EQ(xnn_status_success, status);
        ASSERT_NE(nullptr, fully_connected_op2);

        // Smart pointer to automatically delete fully_connected_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_fully_connected_op(fully_connected_op2, xnn_delete_operator);
        std::vector<uint16_t> output2(output.size(), UINT16_C(0x7E00) /* NaN */);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_fully_connected_nc_f16(
                      fully_connected_op2,
                      batch_size(),
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_fully_connected_nc_f16(
                      fully_connected_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(fully_connected_op2, /*threadpool=*/nullptr));

        // Verify results.
        VerifyF16(output2, output_ref, output_max, output_min);
        VerifyWeightsCache(*internal_weights_cache, old_weights_cache_size);
      }
    }
  }

  void VerifyF16(const std::vector<uint16_t>& output,
                 const std::vector<float>& output_ref,
                 const float output_max,
                 const float output_min) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        // FP16 overflows, it's the nature of the beast. If both reference and
        // actual are infinity, then consider the output to be correct.
        const bool reference_infinity = std::isinf(output_ref[i * output_channels() + c]);
        const bool actual_infinity = std::isinf(fp16_ieee_to_fp32_value(output[i * output_stride() + c]));
        if (reference_infinity && actual_infinity) {
          continue;
        }
        ASSERT_LE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_max)
          << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_min)
          << "batch index = " << i << ", channel = " << c;
        const float tolerance = std::max(1e-4f, 1.0e-2f * std::abs(output_ref[i * output_channels() + c]));
        EXPECT_NEAR(
            output_ref[i * output_channels() + c],
            fp16_ieee_to_fp32_value(output[i * output_stride() + c]),
            tolerance)
          << "batch index = " << i << ", channel = " << c;
      }
    }
  }

  void VerifyWeightsCache(const xnn_internal_weights_cache& weights_cache, size_t old_size) const {
    ASSERT_EQ(weights_cache.cache.hits, 1);
    // Ensure that we did not write more weights to the cache because it was a cache hit.
    ASSERT_EQ(old_size, weights_cache.cache.weights.size);
  };

 private:
  size_t input_channels_{1};
  size_t input_stride_{0};
  size_t output_channels_{1};
  size_t output_stride_{0};
  size_t batch_size_{1};
  uint8_t input_zero_point_{127};
  uint8_t output_zero_point_{127};
  uint8_t kernel_zero_point_{127};  // set qc4w kernel zero point to invalid
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool transpose_weights_{false};
  bool has_bias_{true};
  WeightsType weights_type_{WeightsType::Default};
  bool use_weights_cache_{false};
#if XNN_PLATFORM_JIT
  bool use_jit_{false};
#endif
  size_t iterations_{1};
};
