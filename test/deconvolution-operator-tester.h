// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class DeconvolutionOperatorTester {
 public:
  enum class WeightsType {
    Default,
    FP32,
  };

  enum class Activation {
    MinMax,  // Default activation used in tests. If tests do not specify
             // qmin/qmax, it is equivalent to linear activation.
    Relu,
  };

  DeconvolutionOperatorTester& padding(uint32_t padding) {
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  DeconvolutionOperatorTester& padding_height(uint32_t padding_height) {
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  uint32_t padding_height() const {
    return this->padding_top_ + this->padding_bottom_;
  }

  DeconvolutionOperatorTester& padding_width(uint32_t padding_width) {
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  uint32_t padding_width() const {
    return this->padding_left_ + this->padding_right_;
  }

  DeconvolutionOperatorTester& padding_top(uint32_t padding_top) {
    this->padding_top_ = padding_top;
    return *this;
  }

  uint32_t padding_top() const { return this->padding_top_; }

  DeconvolutionOperatorTester& padding_right(uint32_t padding_right) {
    this->padding_right_ = padding_right;
    return *this;
  }

  uint32_t padding_right() const { return this->padding_right_; }

  DeconvolutionOperatorTester& padding_bottom(uint32_t padding_bottom) {
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  uint32_t padding_bottom() const { return this->padding_bottom_; }

  DeconvolutionOperatorTester& padding_left(uint32_t padding_left) {
    this->padding_left_ = padding_left;
    return *this;
  }

  uint32_t padding_left() const { return this->padding_left_; }

  DeconvolutionOperatorTester& adjustment_height(uint32_t adjustment_height) {
    this->adjustment_height_ = adjustment_height;
    return *this;
  }

  uint32_t adjustment_height() const {
    return this->adjustment_height_;
  }

  DeconvolutionOperatorTester& adjustment_width(uint32_t adjustment_width) {
    this->adjustment_width_ = adjustment_width;
    return *this;
  }

  uint32_t adjustment_width() const {
    return this->adjustment_width_;
  }

  DeconvolutionOperatorTester& input_size(uint32_t input_height, uint32_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  DeconvolutionOperatorTester& input_height(uint32_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  uint32_t input_height() const {
    return this->input_height_;
  }

  DeconvolutionOperatorTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  uint32_t input_width() const {
    return this->input_width_;
  }

  DeconvolutionOperatorTester& groups(uint32_t groups) {
    assert(groups >= 1);
    this->groups_ = groups;
    return *this;
  }

  uint32_t groups() const {
    return this->groups_;
  }

  DeconvolutionOperatorTester& group_input_channels(size_t group_input_channels) {
    assert(group_input_channels >= 1);
    this->group_input_channels_ = group_input_channels;
    return *this;
  }

  size_t group_input_channels() const {
    return this->group_input_channels_;
  }

  DeconvolutionOperatorTester& group_output_channels(size_t group_output_channels) {
    assert(group_output_channels >= 1);
    this->group_output_channels_ = group_output_channels;
    return *this;
  }

  size_t group_output_channels() const {
    return this->group_output_channels_;
  }

  DeconvolutionOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  DeconvolutionOperatorTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size >= 1);
    this->kernel_height_ = kernel_size;
    this->kernel_width_ = kernel_size;
    return *this;
  }

  DeconvolutionOperatorTester& kernel_size(uint32_t kernel_height, uint32_t kernel_width) {
    assert(kernel_height >= 1);
    assert(kernel_width >= 1);
    this->kernel_height_ = kernel_height;
    this->kernel_width_ = kernel_width;
    return *this;
  }

  DeconvolutionOperatorTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height >= 1);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  DeconvolutionOperatorTester& kernel_width(uint32_t kernel_width) {
    assert(kernel_width >= 1);
    this->kernel_width_ = kernel_width;
    return *this;
  }

  uint32_t kernel_width() const {
    return this->kernel_width_;
  }

  DeconvolutionOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilation_height_ = dilation;
    this->dilation_width_ = dilation;
    return *this;
  }

  DeconvolutionOperatorTester& dilation(uint32_t dilation_height, uint32_t dilation_width) {
    assert(dilation_height >= 1);
    assert(dilation_width >= 1);
    this->dilation_height_ = dilation_height;
    this->dilation_width_ = dilation_width;
    return *this;
  }

  DeconvolutionOperatorTester& dilation_height(uint32_t dilation_height) {
    assert(dilation_height >= 1);
    this->dilation_height_ = dilation_height;
    return *this;
  }

  uint32_t dilation_height() const {
    return this->dilation_height_;
  }

  DeconvolutionOperatorTester& dilation_width(uint32_t dilation_width) {
    assert(dilation_width >= 1);
    this->dilation_width_ = dilation_width;
    return *this;
  }

  uint32_t dilation_width() const {
    return this->dilation_width_;
  }

  DeconvolutionOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->stride_height_ = stride;
    this->stride_width_ = stride;
    return *this;
  }

  DeconvolutionOperatorTester& stride(uint32_t stride_height, uint32_t stride_width) {
    assert(stride_height >= 1);
    assert(stride_width >= 1);
    this->stride_height_ = stride_height;
    this->stride_width_ = stride_width;
    return *this;
  }

  DeconvolutionOperatorTester& stride_height(uint32_t stride_height) {
    assert(stride_height >= 1);
    this->stride_height_ = stride_height;
    return *this;
  }

  uint32_t stride_height() const {
    return this->stride_height_;
  }

  DeconvolutionOperatorTester& stride_width(uint32_t stride_width) {
    assert(stride_width >= 1);
    this->stride_width_ = stride_width;
    return *this;
  }

  uint32_t stride_width() const {
    return this->stride_width_;
  }

  DeconvolutionOperatorTester& input_pixel_stride(size_t input_pixel_stride) {
    assert(input_pixel_stride >= 1);
    this->input_pixel_stride_ = input_pixel_stride;
    return *this;
  }

  size_t input_pixel_stride() const {
    if (this->input_pixel_stride_ == 0) {
      return group_input_channels() * groups();
    } else {
      assert(this->input_pixel_stride_ >= group_input_channels() * groups());
      return this->input_pixel_stride_;
    }
  }

  DeconvolutionOperatorTester& output_pixel_stride(size_t output_pixel_stride) {
    assert(output_pixel_stride >= 1);
    this->output_pixel_stride_ = output_pixel_stride;
    return *this;
  }

  size_t output_pixel_stride() const {
    if (this->output_pixel_stride_ == 0) {
      return group_output_channels() * groups();
    } else {
      assert(this->output_pixel_stride_ >= group_output_channels() * groups());
      return this->output_pixel_stride_;
    }
  }

  uint32_t dilated_kernel_height() const {
    return (kernel_height() - 1) * dilation_height() + 1;
  }

  uint32_t dilated_kernel_width() const {
    return (kernel_width() - 1) * dilation_width() + 1;
  }

  size_t output_height() const {
    return stride_height() * (input_height() - 1) + adjustment_height() + dilated_kernel_height() - padding_height();
  }

  size_t output_width() const {
    return stride_width() * (input_width() - 1) + adjustment_width() + dilated_kernel_width() - padding_width();
  }

  DeconvolutionOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  DeconvolutionOperatorTester& next_input_height(uint32_t next_input_height) {
    assert(next_input_height >= 1);
    this->next_input_height_ = next_input_height;
    return *this;
  }

  uint32_t next_input_height() const {
    if (this->next_input_height_ == 0) {
      return input_height();
    } else {
      return this->next_input_height_;
    }
  }

  DeconvolutionOperatorTester& next_input_width(uint32_t next_input_width) {
    assert(next_input_width >= 1);
    this->next_input_width_ = next_input_width;
    return *this;
  }

  uint32_t next_input_width() const {
    if (this->next_input_width_ == 0) {
      return input_width();
    } else {
      return this->next_input_width_;
    }
  }

  size_t next_output_height() const {
    return stride_height() * (next_input_height() - 1) + adjustment_height() + dilated_kernel_height() - padding_height();
  }

  size_t next_output_width() const {
    return stride_width() * (next_input_width() - 1) + adjustment_width() + dilated_kernel_width() - padding_width();
  }

  DeconvolutionOperatorTester& next_batch_size(size_t next_batch_size) {
    assert(next_batch_size >= 1);
    this->next_batch_size_ = next_batch_size;
    return *this;
  }

  size_t next_batch_size() const {
    if (this->next_batch_size_ == 0) {
      return batch_size();
    } else {
      return this->next_batch_size_;
    }
  }

  DeconvolutionOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  DeconvolutionOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  DeconvolutionOperatorTester& activation(Activation activation) {
    this->activation_ = activation;
    return *this;
  }

  Activation activation() const {
    return this->activation_;
  }

  DeconvolutionOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  bool has_bias() const {
    return this->has_bias_;
  }

  DeconvolutionOperatorTester& weights_type(WeightsType weights_type) {
    this->weights_type_ = weights_type;
    return *this;
  }

  WeightsType weights_type() const {
    return this->weights_type_;
  }

  DeconvolutionOperatorTester& use_weights_cache(bool use_weights_cache) {
    this->use_weights_cache_ = use_weights_cache;
    return *this;
  }

  bool use_weights_cache() const {
    return this->use_weights_cache_;
  }

  DeconvolutionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestQC8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<int8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> requantization_scales(groups() * group_output_channels());

    const int8_t input_zero_point = 1;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int8_t output_zero_point = -1;
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        int32_t accumulated_min = accumulators[c];
        int32_t accumulated_max = accumulators[c];
        for (size_t px = 0; px < batch_size() * output_height() * output_width(); px++) {
          accumulated_min = std::min(accumulated_min, accumulators[px * groups() * group_output_channels() + c]);
          accumulated_max = std::max(accumulated_max, accumulators[px * groups() * group_output_channels() + c]);
        }

        float requantization_scale = 2.328e-10f; //0x1.0p-32f;
        if (accumulated_max != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::max()) - int32_t(output_zero_point)) / float(accumulated_max));
        }
        if (accumulated_min != 0) {
          requantization_scale = std::max(requantization_scale,
            float(int32_t(std::numeric_limits<int8_t>::min()) - int32_t(output_zero_point)) / float(accumulated_min));
        }
        requantization_scale = std::min(requantization_scale, 1.f);

        requantization_scales[c] = requantization_scale;
      }

      // Renormalize reference results.
      for (size_t c = 0; c < groups() * group_output_channels(); c++) {
        for (size_t px = 0; px < batch_size() * output_height() * output_width(); px++) {
          output_ref[px * groups() * group_output_channels() + c] = double(int32_t(output_zero_point)) +
            double(accumulators[px * groups() * group_output_channels() + c]) * double(requantization_scales[c]);
        }
      }
      std::transform(output_ref.cbegin(), output_ref.cend(), output_ref.begin(),
        [this](double x) -> double {
          return std::max<double>(std::min<double>(x, double(qmax() - 0x80)), double(qmin() - 0x80));
        });

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

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

      ASSERT_EQ(
          xnn_status_success,
          xnn_create_deconvolution2d_nhwc_qs8_qc8w(
              padding_top(), padding_right(), padding_bottom(), padding_left(),
              kernel_height(), kernel_width(), stride_height(), stride_width(),
              dilation_height(), dilation_width(), groups(),
              group_input_channels(), group_output_channels(),
              input_pixel_stride(), output_pixel_stride(), input_zero_point,
              /*input_scale=*/1.0f, requantization_scales.data(), kernel.data(),
              has_bias() ? bias.data() : nullptr, output_zero_point,
              /*output_scale=*/1.f, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
              /*flags=*/0, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op));

      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }
      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qs8_qc8w(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_qs8_qc8w(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      VerifyQC8(output, output_ref);

      if (use_weights_cache()) {
        xnn_operator_t deconvolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(
            xnn_status_success,
            xnn_create_deconvolution2d_nhwc_qs8_qc8w(
                padding_top(), padding_right(), padding_bottom(), padding_left(),
                kernel_height(), kernel_width(), stride_height(), stride_width(),
                dilation_height(), dilation_width(), groups(),
                group_input_channels(), group_output_channels(),
                input_pixel_stride(), output_pixel_stride(), input_zero_point,
                1.0f /* input scale */, requantization_scales.data(), kernel.data(),
                has_bias() ? bias.data() : nullptr, output_zero_point,
                /*output_scale=*/1.0f, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
                /*flags=*/0, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op2));

        // Smart pointer to automatically delete deconvolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op2, xnn_delete_operator);
        std::vector<int8_t> output2(output.size(), INT8_C(0xA5));

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_deconvolution2d_nhwc_qs8_qc8w(
                      deconvolution_op2,
                      batch_size(), input_height(), input_width(),
                      adjustment_height(), adjustment_width(),
                      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_deconvolution2d_nhwc_qs8_qc8w(
                      deconvolution_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(deconvolution_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(internal_weights_cache, old_weights_cache_size);
        VerifyQC8(output2, output_ref);
      }
    }
  }

  void VerifyQC8(const std::vector<int8_t> &output,
                 const std::vector<double> &output_ref) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  double(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]),
                  0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void VerifyWeightsCache(struct xnn_internal_weights_cache* weights_cache, size_t old_size) const {
    ASSERT_EQ(weights_cache->cache.hits, 1);
    // Ensure that we did not write more weights to the cache because it was a cache hit.
    ASSERT_EQ(old_size, weights_cache->cache.weights.size);
  };

  void TestQU8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<uint8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<uint8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
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

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

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

      ASSERT_EQ(
          xnn_status_success,
          xnn_create_deconvolution2d_nhwc_qu8(
              padding_top(), padding_right(), padding_bottom(), padding_left(),
              kernel_height(), kernel_width(), stride_height(), stride_width(),
              dilation_height(), dilation_width(), groups(),
              group_input_channels(), group_output_channels(),
              input_pixel_stride(), output_pixel_stride(), input_zero_point,
              1.0f /* input scale */, kernel_zero_point,
              1.0f /* kernel scale */, kernel.data(),
              has_bias() ? bias.data() : nullptr, output_zero_point,
              output_scale, qmin(), qmax(),
              /*flags=*/0, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op));

      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }
      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qu8(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_qu8(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results.
      VerifyQU8(output, output_ref, output_zero_point);


      if (use_weights_cache()) {
        xnn_operator_t deconvolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(
            xnn_status_success,
            xnn_create_deconvolution2d_nhwc_qu8(
                padding_top(), padding_right(), padding_bottom(), padding_left(),
                kernel_height(), kernel_width(), stride_height(), stride_width(),
                dilation_height(), dilation_width(), groups(),
                group_input_channels(), group_output_channels(),
                input_pixel_stride(), output_pixel_stride(), input_zero_point,
                1.0f /* input scale */, kernel_zero_point,
                1.0f /* kernel scale */, kernel.data(),
                has_bias() ? bias.data() : nullptr, output_zero_point,
                output_scale, qmin(), qmax(),
                /*flags=*/0, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op2));

        // Smart pointer to automatically delete deconvolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op2, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_deconvolution2d_nhwc_qu8(
                      deconvolution_op2,
                      batch_size(), input_height(), input_width(),
                      adjustment_height(), adjustment_width(),
                      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_deconvolution2d_nhwc_qu8(
                      deconvolution_op2,
                      input.data(), output.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(deconvolution_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(internal_weights_cache, old_weights_cache_size);
        VerifyQU8(output, output_ref, output_zero_point);
      }
    }
  }

  void VerifyQU8(const std::vector<uint8_t> &output,
                 const std::vector<double> &output_ref,
                 uint8_t output_zero_point) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  double(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                  0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
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

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<uint16_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> kernel_as_float(kernel.size());
    std::vector<uint16_t> bias(groups() * group_output_channels());
    std::vector<float> bias_as_float(bias.size());
    std::vector<uint16_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(kernel.cbegin(), kernel.cend(), kernel_as_float.begin(), fp16_ieee_to_fp32_value);
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(bias.cbegin(), bias.cend(), bias_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias_as_float[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) *
                            kernel_as_float[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      float output_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min));
      output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max));
      if (accumulated_range == 0.0f) {
        output_min = -std::numeric_limits<float>::infinity();
        output_max = +std::numeric_limits<float>::infinity();
      }
      if (qmin() == std::numeric_limits<uint8_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      if (qmax() == std::numeric_limits<uint8_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

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
      if (weights_type() == WeightsType::FP32) {
        flags |= XNN_FLAG_FP32_STATIC_WEIGHTS;
      }
      const xnn_status status = xnn_create_deconvolution2d_nhwc_f16(
        padding_top(), padding_right(), padding_bottom(), padding_left(),
        kernel_height(), kernel_width(), stride_height(), stride_width(),
        dilation_height(), dilation_width(), groups(),
        group_input_channels(), group_output_channels(),
        input_pixel_stride(), output_pixel_stride(),
        kernel_data, has_bias() ? bias_data : nullptr,
        output_min, output_max,
        flags, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, deconvolution_op);
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f16(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f16(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      VerifyF16(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        xnn_operator_t deconvolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(xnn_status_success,
                  xnn_create_deconvolution2d_nhwc_f16(
                      padding_top(), padding_right(), padding_bottom(), padding_left(),
                      kernel_height(), kernel_width(), stride_height(), stride_width(),
                      dilation_height(), dilation_width(), groups(),
                      group_input_channels(), group_output_channels(),
                      input_pixel_stride(), output_pixel_stride(),
                      kernel_data, has_bias() ? bias_data : nullptr,
                      output_min, output_max,
                      flags, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op2));
        ASSERT_NE(nullptr, deconvolution_op2);

        // Smart pointer to automatically delete deconvolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op2, xnn_delete_operator);
        std::vector<uint16_t> output2(output.size(), UINT16_C(0x7E00) /* NaN */);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_deconvolution2d_nhwc_f16(
                      deconvolution_op2,
                      batch_size(), input_height(), input_width(),
                      adjustment_height(), adjustment_width(),
                      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_deconvolution2d_nhwc_f16(
                      deconvolution_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(deconvolution_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(internal_weights_cache, old_weights_cache_size);
        VerifyF16(output2, output_ref, output_max, output_min);
      }
    }
  }

  void VerifyF16(const std::vector<uint16_t> &output,
                 const std::vector<float> &output_ref,
                 float output_max,
                 float output_min) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]),
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  1.0e-2f * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestQD8F32QC8W() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
    std::uniform_real_distribution<float> f32idist(0.5f, 2.0f);
    std::uniform_int_distribution<int32_t> w8dist(
    std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<xnn_qd8_quantization_params> quantization_params(batch_size());
    std::vector<float> kernel_scale(groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return w8dist(rng); });
      // Weights in the same output channel will be all positive or all negative. This ensures that no catastrophic
      // cancellation occur, but test covers both positive and negative values.
      for (size_t g = 0; g < groups(); g++) {
        for (size_t oc = 0; oc < group_output_channels(); oc++) {
          int32_t range = w8dist(rng);
          auto weights_dist = std::uniform_int_distribution<int32_t>(std::min<int32_t>(range, 0), std::max<int32_t>(range, 0));
          bias[g * group_output_channels() + oc] = f32dist(rng);
          for (size_t y = 0; y < kernel_height(); y++) {
            for (size_t x = 0; x < kernel_width(); x++) {
              for (size_t ic = 0; ic < group_input_channels(); ic++) {
                size_t index = ((((g * group_output_channels() + oc) * kernel_height()) + y) * kernel_width() + x) *
                                 group_input_channels() + ic;
                kernel[index] = weights_dist(rng);
              }
            }
          }
        }
      }

      std::generate(kernel_scale.begin(), kernel_scale.end(),
                    [&]() { return f32idist(rng); });

      std::generate(
          quantization_params.begin(), quantization_params.end(), [&]() {
            return xnn_qd8_quantization_params{w8dist(rng), f32idist(rng)};
          });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      for (size_t i = 0; i < batch_size(); i++) {
        int32_t zero_point = quantization_params[i].zero_point;
        float inv_scale = quantization_params[i].inv_scale;
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - zero_point) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
            for (size_t g = 0; g < groups(); g++) {
              for (size_t oc = 0; oc < group_output_channels(); oc++) {
                size_t n_index = g * group_output_channels() + oc;
                output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] *=
                  (inv_scale * kernel_scale[n_index]);
                if (has_bias()) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      switch (activation()) {
        case Activation::MinMax:
          if (qmin() != 0) {
            ASSERT_THAT(output_ref, testing::Contains(testing::Lt(output_min)));
          }
          if (qmax() != 255) {
            ASSERT_THAT(output_ref, testing::Contains(testing::Gt(output_max)));
          }
          break;
        case Activation::Relu:
          output_min = 0.0f;
          output_max = std::numeric_limits<float>::infinity();
          ASSERT_THAT(output_ref, testing::Contains(testing::Lt(output_min)));
          ASSERT_THAT(output_ref, testing::Contains(testing::Gt(output_min)));
          break;
      }

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      xnn_code_cache_t auto_code_cache = nullptr;
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

      ASSERT_EQ(
          xnn_status_success,
          xnn_create_deconvolution2d_nhwc_qd8_f32_qc8w(
              padding_top(), padding_right(), padding_bottom(), padding_left(),
              kernel_height(), kernel_width(), stride_height(), stride_width(),
              dilation_height(), dilation_width(), groups(),
              group_input_channels(), group_output_channels(),
              input_pixel_stride(), output_pixel_stride(), kernel_scale.data(), kernel.data(),
              has_bias() ? bias.data() : nullptr, output_min, output_max,
              /*flags=*/0, auto_code_cache, auto_weights_cache.get(), &deconvolution_op));
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qd8_f32_qc8w(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(
          xnn_status_success,
          xnn_setup_deconvolution2d_nhwc_qd8_f32_qc8w(
              deconvolution_op, input.data(), output.data(),
              reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
                  quantization_params.data())));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // We already finalized the code cache, so create a new code cache if we are testing JIT.
        xnn_code_cache_t auto_inner_code_cache = nullptr;
        xnn_operator_t deconvolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(
            xnn_status_success,
            xnn_create_deconvolution2d_nhwc_qd8_f32_qc8w(
                padding_top(), padding_right(), padding_bottom(), padding_left(),
                kernel_height(), kernel_width(), stride_height(), stride_width(),
                dilation_height(), dilation_width(), groups(),
                group_input_channels(), group_output_channels(),
                input_pixel_stride(), output_pixel_stride(), kernel_scale.data(), kernel.data(),
                has_bias() ? bias.data() : nullptr, output_min, output_max,
                /*flags=*/0, auto_inner_code_cache, auto_weights_cache.get(), &deconvolution_op2));

        // Smart pointer to automatically delete deconvolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op2, xnn_delete_operator);
        std::vector<float> output2(output.size(), nanf(""));

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_deconvolution2d_nhwc_qd8_f32_qc8w(
                      deconvolution_op2,
                      batch_size(), input_height(), input_width(),
                      adjustment_height(), adjustment_width(),
                      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                      /*threadpool=*/nullptr));

        ASSERT_EQ(
            xnn_status_success,
            xnn_setup_deconvolution2d_nhwc_qd8_f32_qc8w(
                deconvolution_op2, input.data(), output2.data(),
                reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
                    quantization_params.data())));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(deconvolution_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(internal_weights_cache, old_weights_cache_size);
        VerifyF32(output2, output_ref, output_max, output_min);
      }
    }
  }

  void TestF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      // Weights in the same output channel will be all positive or all negative. This ensures that no catastrophic
      // cancellation occur, but test covers both positive and negative values.
      for (size_t g = 0; g < groups(); g++) {
        for (size_t oc = 0; oc < group_output_channels(); oc++) {
          float range = f32dist(rng);
          auto weights_dist = std::uniform_real_distribution<float>(std::min(range, 0.0f), std::max(range, 0.0f));
          bias[g * group_output_channels() + oc] = weights_dist(rng);
          for (size_t y = 0; y < kernel_height(); y++) {
            for (size_t x = 0; x < kernel_width(); x++) {
              for (size_t ic = 0; ic < group_input_channels(); ic++) {
                size_t index = ((((g * group_output_channels() + oc) * kernel_height()) + y) * kernel_width() + x) *
                                 group_input_channels() + ic;
                kernel[index] = weights_dist(rng);
              }
            }
          }
        }
      }

      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      switch (activation()) {
        case Activation::MinMax:
          if (qmin() != 0) {
            ASSERT_THAT(output_ref, testing::Contains(testing::Lt(output_min)));
          }
          if (qmax() != 255) {
            ASSERT_THAT(output_ref, testing::Contains(testing::Gt(output_max)));
          }
          break;
        case Activation::Relu:
          output_min = 0.0f;
          output_max = std::numeric_limits<float>::infinity();
          ASSERT_THAT(output_ref, testing::Contains(testing::Lt(output_min)));
          ASSERT_THAT(output_ref, testing::Contains(testing::Gt(output_min)));
          break;
      }

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      xnn_code_cache_t auto_code_cache = nullptr;
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

      ASSERT_EQ(
          xnn_status_success,
          xnn_create_deconvolution2d_nhwc_f32(
              padding_top(), padding_right(), padding_bottom(), padding_left(),
              kernel_height(), kernel_width(), stride_height(), stride_width(),
              dilation_height(), dilation_width(), groups(),
              group_input_channels(), group_output_channels(),
              input_pixel_stride(), output_pixel_stride(), kernel.data(),
              has_bias() ? bias.data() : nullptr, output_min, output_max,
              /*flags=*/0, auto_code_cache, auto_weights_cache.get(), &deconvolution_op));
      if (use_weights_cache()) {
        ASSERT_EQ(xnn_status_success,
                  xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));
      }

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f32(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref, output_max, output_min);

      if (use_weights_cache()) {
        // We already finalized the code cache, so create a new code cache if we are testing JIT.
        xnn_code_cache_t auto_inner_code_cache = nullptr;
        xnn_operator_t deconvolution_op2 = nullptr;
        size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

        ASSERT_EQ(
            xnn_status_success,
            xnn_create_deconvolution2d_nhwc_f32(
                padding_top(), padding_right(), padding_bottom(), padding_left(),
                kernel_height(), kernel_width(), stride_height(), stride_width(),
                dilation_height(), dilation_width(), groups(),
                group_input_channels(), group_output_channels(),
                input_pixel_stride(), output_pixel_stride(), kernel.data(),
                has_bias() ? bias.data() : nullptr, output_min, output_max,
                /*flags=*/0, auto_inner_code_cache, auto_weights_cache.get(), &deconvolution_op2));

        // Smart pointer to automatically delete deconvolution_op2.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op2, xnn_delete_operator);
        std::vector<float> output2(output.size(), nanf(""));

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_deconvolution2d_nhwc_f32(
                      deconvolution_op2,
                      batch_size(), input_height(), input_width(),
                      adjustment_height(), adjustment_width(),
                      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                      /*threadpool=*/nullptr));

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_deconvolution2d_nhwc_f32(
                      deconvolution_op2,
                      input.data(), output2.data()));

        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(deconvolution_op2, /*threadpool=*/nullptr));

        VerifyWeightsCache(internal_weights_cache, old_weights_cache_size);
        VerifyF32(output2, output_ref, output_max, output_min);
      }
    }
  }

  // A variation of TestF32 that stresses the weights cache. All the operator creation needs to happen before
  // finalization and setup.
  void StressWeightsCacheTestF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    struct xnn_internal_weights_cache* internal_weights_cache = nullptr;
    std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> auto_weights_cache(
      nullptr, xnn_delete_weights_cache);
    {
      xnn_weights_cache_t weights_cache = nullptr;
      xnn_create_weights_cache(&weights_cache);
      auto_weights_cache.reset(weights_cache);
      if (weights_cache) {
        internal_weights_cache = (struct xnn_internal_weights_cache*) weights_cache->context;
      }
    }
    size_t old_weights_cache_size = internal_weights_cache->cache.weights.size;

    std::vector<xnn_operator_t> operators;
    operators.reserve(iterations());
    std::vector<std::vector<float>> inputs;
    inputs.reserve(iterations());
    std::vector<std::vector<float>> outputs;
    outputs.reserve(iterations());
    std::vector<std::vector<float>> output_refs;
    output_refs.reserve(iterations());
    std::vector<float> output_mins;
    output_mins.reserve(iterations());
    std::vector<float> output_maxs;
    output_maxs.reserve(iterations());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
                               (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
      std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
      std::vector<float> bias(groups() * group_output_channels());
      std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
      std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
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
      output_mins.push_back(output_min);
      output_maxs.push_back(output_max);

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(
          xnn_status_success,
          xnn_create_deconvolution2d_nhwc_f32(
              padding_top(), padding_right(), padding_bottom(), padding_left(),
              kernel_height(), kernel_width(), stride_height(), stride_width(),
              dilation_height(), dilation_width(), groups(),
              group_input_channels(), group_output_channels(),
              input_pixel_stride(), output_pixel_stride(), kernel.data(),
              has_bias() ? bias.data() : nullptr, output_min, output_max,
              /*flags=*/0, /*code_cache=*/nullptr, auto_weights_cache.get(), &deconvolution_op));

      operators.push_back(std::move(deconvolution_op));
      inputs.push_back(std::move(input));
      outputs.push_back(std::move(output));
      output_refs.push_back(std::move(output_ref));
    }

    ASSERT_EQ(xnn_status_success,
              xnn_finalize_weights_cache(auto_weights_cache.get(), xnn_weights_cache_finalization_kind_soft));

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      xnn_operator_t deconvolution_op = operators[iteration];

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f32(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          inputs[iteration].data(), outputs[iteration].data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      VerifyF32(outputs[iteration],
                output_refs[iteration],
                output_maxs[iteration],
                output_mins[iteration]);
      xnn_delete_operator(deconvolution_op);
    }

    // Check that the weights cache grew. We don't check that it moved because that can be flaky (depends if initial
    // allocation is big enough, and future allocations can land on the old pointer).
    ASSERT_LT(old_weights_cache_size, internal_weights_cache->cache.weights.size);
    // Since the weights are randomized, it is very unlikely to have any hits.
    ASSERT_EQ(iterations(), internal_weights_cache->cache.misses);
    ASSERT_EQ(0, internal_weights_cache->cache.hits);
    ASSERT_EQ(iterations(), internal_weights_cache->cache.num_entries);
  }

  void VerifyF32(const std::vector<float> &output,
                 const std::vector<float> &output_ref,
                 float output_max,
                 float output_min) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          for (size_t g = 0; g < groups(); g++) {
            for (size_t c = 0; c < group_output_channels(); c++) {
              EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              EXPECT_NEAR(
                  output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                  output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c],
                  std::max(1.0e-4, 1.0e-4 * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c])))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
            }
          }
        }
      }
    }
  }

  void TestSetupQS8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + groups() * group_input_channels()));
    std::vector<int8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<int8_t> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + groups() * group_output_channels()));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<int32_t> next_accumulators(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<double> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    const int8_t input_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
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

      // Create, setup, and run Deconvolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_qs8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          input_zero_point, 1.0f /* input scale */,
          1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, nullptr, nullptr, &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qs8(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_qs8(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results for the second run, including renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_accumulators.begin(), next_accumulators.end(), 0);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      std::transform(next_accumulators.cbegin(), next_accumulators.cend(), next_output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax() - 0x80) - output_zero_point), double(qmin() - 0x80) - output_zero_point);
        });

      // Setup and run Deconvolution operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qs8(
          deconvolution_op,
          next_batch_size(), next_input_height(), next_input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_qs8(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax() - 0x80))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin() - 0x80))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupQU8() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + groups() * group_input_channels()));
    std::vector<uint8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<uint8_t> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + groups() * group_output_channels()));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<int32_t> next_accumulators(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<double> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
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

      // Create, setup, and run Deconvolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_qu8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          input_zero_point, 1.0f /* input scale */,
          kernel_zero_point, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          0, nullptr, nullptr, &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qu8(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_qu8(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results for the second run, including renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_accumulators.begin(), next_accumulators.end(), 0);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      std::transform(next_accumulators.cbegin(), next_accumulators.cend(), next_output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Setup and run Deconvolution operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_qu8(
          deconvolution_op,
          next_batch_size(), next_input_height(), next_input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_qu8(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupF16() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + groups() * group_input_channels()));
    std::vector<uint16_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<uint16_t> bias(groups() * group_output_channels());
    std::vector<uint16_t> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    fp16_ieee_to_fp32_value(bias[g * group_output_channels() + oc]);
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) *
                            fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      float output_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min));
      output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max));
      if (accumulated_range == 0.0f) {
        output_min = -std::numeric_limits<float>::infinity();
        output_max = +std::numeric_limits<float>::infinity();
      }
      if (qmin() == std::numeric_limits<uint8_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      if (qmax() == std::numeric_limits<uint8_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, and run Deconvolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      const xnn_status status = xnn_create_deconvolution2d_nhwc_f16(
        padding_top(), padding_right(), padding_bottom(), padding_left(),
        kernel_height(), kernel_width(),
        stride_height(), stride_width(),
        dilation_height(), dilation_width(),
        groups(), group_input_channels(), group_output_channels(),
        input_pixel_stride(), output_pixel_stride(),
        kernel.data(), has_bias() ? bias.data() : nullptr,
        output_min, output_max,
        0, nullptr, nullptr, &deconvolution_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, deconvolution_op);

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f16(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f16(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]),
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    1.0e-2f * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results for the second run, including clamping.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    fp16_ieee_to_fp32_value(bias[g * group_output_channels() + oc]);
                }
              }
            }
          }
        }
      } else {
        std::fill(next_output_ref.begin(), next_output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            fp16_ieee_to_fp32_value(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) *
                            fp16_ieee_to_fp32_value(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (float& value : next_output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Setup and run Deconvolution operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f16(
          deconvolution_op,
          next_batch_size(), next_input_height(), next_input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f16(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    fp16_ieee_to_fp32_value(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]),
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    1.0e-2f * std::abs(next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupF32() const {
    ASSERT_EQ(weights_type(), WeightsType::Default);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + groups() * group_input_channels()));
    std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, and run Deconvolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          0, nullptr, nullptr, &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f32(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results for the second run, including clamping.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_output_ref.begin(), next_output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (float& value : next_output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Setup and run Deconvolution operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_deconvolution2d_nhwc_f32(
          deconvolution_op,
          next_batch_size(), next_input_height(), next_input_width(),
          adjustment_height(), adjustment_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                EXPECT_GE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                EXPECT_LE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

 private:
  uint32_t padding_top_{0};
  uint32_t padding_right_{0};
  uint32_t padding_bottom_{0};
  uint32_t padding_left_{0};
  size_t input_height_{1};
  size_t input_width_{1};
  uint32_t groups_{1};
  size_t group_input_channels_{1};
  size_t input_pixel_stride_{0};
  size_t group_output_channels_{1};
  size_t output_pixel_stride_{0};
  size_t batch_size_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint32_t adjustment_height_{0};
  uint32_t adjustment_width_{0};
  uint32_t dilation_height_{1};
  uint32_t dilation_width_{1};
  uint32_t stride_height_{1};
  uint32_t stride_width_{1};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  Activation activation_{Activation::MinMax};
  bool has_bias_{true};
  WeightsType weights_type_{WeightsType::Default};
  bool use_weights_cache_{false};
  size_t iterations_{1};
};
