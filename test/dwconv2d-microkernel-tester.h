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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/math.h>
#include <xnnpack/pack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


class DWConv2DMicrokernelTester {
 public:
  inline DWConv2DMicrokernelTester& padding_left(uint32_t padding_left) {
    this->padding_left_ = padding_left;
    return *this;
  }

  inline uint32_t padding_left() const {
    return this->padding_left_;
  }

  inline DWConv2DMicrokernelTester& padding_right(uint32_t padding_right) {
    this->padding_right_ = padding_right;
    return *this;
  }

  inline uint32_t padding_right() const {
    return this->padding_right_;
  }

  inline DWConv2DMicrokernelTester& padding_top(uint32_t padding_top) {
    this->padding_top_ = padding_top;
    return *this;
  }

  inline uint32_t padding_top() const {
    return this->padding_top_;
  }


  inline DWConv2DMicrokernelTester& padding_bottom(uint32_t padding_bottom) {
    this->padding_bottom_ = padding_bottom;
    return *this;
  }
  inline uint32_t padding_bottom() const {
    return this->padding_bottom_;
  }

  inline DWConv2DMicrokernelTester& input_height(uint32_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  inline uint32_t input_height() const {
    return this->input_height_;
  }

  inline DWConv2DMicrokernelTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline uint32_t input_width() const {
    return this->input_width_;
  }

  inline DWConv2DMicrokernelTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_ = subsampling;
    return *this;
  }

  inline uint32_t subsampling() const {
    return this->subsampling_;
  }

  inline DWConv2DMicrokernelTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height != 0);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  inline uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  inline DWConv2DMicrokernelTester& kernel_width(uint32_t kernel_width) {
    assert(kernel_width != 0);
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline uint32_t kernel_width() const {
    return this->kernel_width_;
  }

  inline uint32_t kernel_size() const {
    return kernel_height() * kernel_width();
  }

  inline uint32_t output_height() const {
    const uint32_t padded_input_height = padding_top() + input_height() + padding_bottom();
    if (padded_input_height <= kernel_height()) {
      return 1;
    } else {
      return (padded_input_height - kernel_height()) / subsampling() + 1;
    }
  }

  inline uint32_t output_width() const {
    const uint32_t padded_input_width = padding_left() + input_width() + padding_right();
    if (padded_input_width <= kernel_width()) {
      return 1;
    } else {
      return (padded_input_width - kernel_width()) / subsampling() + 1;
    }
  }

  inline DWConv2DMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DWConv2DMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DWConv2DMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_dwconv2d_chw_ukernel_fn dwconv, xnn_init_f32_chw_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float, AlignedAllocator<float, 64>> input(input_height() * input_width() + 2 * XNN_EXTRA_BYTES);
    std::vector<float> zero(input_width() + 2 * XNN_EXTRA_BYTES);
    std::vector<float> packed_weights(kernel_size() + 1);
    std::vector<float, AlignedAllocator<float, 64>> output(output_height() * output_width());
    std::vector<float> output_ref(output_height() * output_width());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      for (size_t oy = 0; oy < output_height(); oy++) {
        for (size_t ox = 0; ox < output_width(); ox++) {
          float acc = packed_weights[0];
          for (size_t ky = 0; ky < kernel_height(); ky++) {
            const size_t iy = oy * subsampling() + ky - padding_top();
            for (size_t kx = 0; kx < kernel_width(); kx++) {
              const size_t ix = ox * subsampling() + kx - padding_left();
              if (ix < input_width() && iy < input_height()) {
                const float input_val = input[iy * input_width() + ix];
                const float kernel_val = packed_weights[1 + ky * kernel_width() + kx];
                acc += input_val * kernel_val;
              }
            }
          }
          output_ref[oy * output_width() + ox] = acc;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Prepare parameters.
      xnn_f32_chw_params chw_params;
      init_params(&chw_params, input_width(), output_min, output_max);

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // Call optimized micro-kernel.
      dwconv(
        input_height(), input_width() * sizeof(float),
        input.data(), packed_weights.data(), zero.data(), output.data(),
        padding_top(),
        &chw_params);

      // Verify results.
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          ASSERT_NEAR(
              output_ref[y * output_width() + x],
              output[y * output_width() + x],
              std::abs(output_ref[y * output_width() + x]) * 1.0e-5)
            << "x = " << x << ", y = " << y;
        }
      }
    }
  }

  void Test(xnn_f16_dwconv2d_chw_ukernel_fn dwconv, xnn_init_f16_chw_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> input(input_height() * input_width() + 2 * XNN_EXTRA_BYTES);
    std::vector<uint16_t> zero(input_width() + 2 * XNN_EXTRA_BYTES);
    std::vector<uint16_t> packed_weights(kernel_size() + 1);
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> output(output_height() * output_width());
    std::vector<float> output_ref(output_height() * output_width());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(packed_weights.begin(), packed_weights.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      for (size_t oy = 0; oy < output_height(); oy++) {
        for (size_t ox = 0; ox < output_width(); ox++) {
          float acc = fp16_ieee_to_fp32_value(packed_weights[0]);
          for (size_t ky = 0; ky < kernel_height(); ky++) {
            const size_t iy = oy * subsampling() + ky - padding_top();
            for (size_t kx = 0; kx < kernel_width(); kx++) {
              const size_t ix = ox * subsampling() + kx - padding_left();
              if (ix < input_width() && iy < input_height()) {
                const float input_val = fp16_ieee_to_fp32_value(input[iy * input_width() + ix]);
                const float kernel_val = fp16_ieee_to_fp32_value(packed_weights[1 + ky * kernel_width() + kx]);
                acc += input_val * kernel_val;
              }
            }
          }
          output_ref[oy * output_width() + ox] = acc;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));

      // Prepare parameters.
      xnn_f16_chw_params chw_params;
      init_params(&chw_params, input_width(),
        fp16_ieee_from_fp32_value(output_min),
        fp16_ieee_from_fp32_value(output_max));

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // Call optimized micro-kernel.
      dwconv(
        input_height(), input_width() * sizeof(uint16_t),
        input.data(), packed_weights.data(), zero.data(), output.data(),
        padding_top(),
        &chw_params);

      // Verify results.
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          ASSERT_NEAR(
              output_ref[y * output_width() + x],
              fp16_ieee_to_fp32_value(output[y * output_width() + x]),
              std::abs(output_ref[y * output_width() + x]) * 1.0e-2f)
            << "x = " << x << ", y = " << y;
        }
      }
    }
  }

 private:
  uint32_t padding_left_{0};
  uint32_t padding_right_{0};
  uint32_t padding_top_{0};
  uint32_t padding_bottom_{0};
  uint32_t input_height_{1};
  uint32_t input_width_{1};
  uint32_t subsampling_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
