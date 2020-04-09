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
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/math.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


class DWConvSpCHWMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline DWConvSpCHWMicrokernelTester& input_tuple_size(uint32_t input_tuple_size) {
    this->input_tuple_size_ = input_tuple_size;
    return *this;
  }

  inline uint32_t input_tuple_size() const {
    return this->input_tuple_size_;
  }

  inline DWConvSpCHWMicrokernelTester& output_tuple_size(uint32_t output_tuple_size) {
    this->output_tuple_size_ = output_tuple_size;
    return *this;
  }

  inline uint32_t output_tuple_size() const {
    return this->output_tuple_size_;
  }

  inline DWConvSpCHWMicrokernelTester& padding_left(uint32_t padding_left) {
    this->padding_left_ = padding_left;
    return *this;
  }

  inline uint32_t padding_left() const {
    return this->padding_left_;
  }

  inline DWConvSpCHWMicrokernelTester& padding_right(uint32_t padding_right) {
    this->padding_right_ = padding_right;
    return *this;
  }

  inline uint32_t padding_right() const {
    return this->padding_right_;
  }

  inline uint32_t input_height() const {
    return (output_height() - 1) * subsampling() + kernel_height();
  }

  inline DWConvSpCHWMicrokernelTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline uint32_t input_width() const {
    return this->input_width_;
  }

  inline DWConvSpCHWMicrokernelTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_ = subsampling;
    return *this;
  }

  inline uint32_t subsampling() const {
    return this->subsampling_;
  }

  inline DWConvSpCHWMicrokernelTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height != 0);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  inline uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  inline DWConvSpCHWMicrokernelTester& kernel_width(uint32_t kernel_width) {
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

  inline DWConvSpCHWMicrokernelTester& output_height(uint32_t output_height) {
    assert(output_height >= 1);
    this->output_height_ = output_height;
    return *this;
  }

  inline uint32_t output_height() const {
    return this->output_height_;
  }

  inline uint32_t output_width() const {
    const uint32_t padded_input_width = padding_left() + input_width() + padding_right();
    if (padded_input_width <= kernel_width()) {
      return 1;
    } else {
      return (padded_input_width - kernel_width()) / subsampling() + 1;
    }
  }

  inline DWConvSpCHWMicrokernelTester& input_tuple_stride(uint32_t input_tuple_stride) {
    assert(input_tuple_stride != 0);
    this->input_tuple_stride_ = input_tuple_stride;
    return *this;
  }

  inline uint32_t input_tuple_stride() const {
    if (this->input_tuple_stride_ == 0) {
      return this->input_tuple_size();
    } else {
      return this->input_tuple_stride_;
    }
  }

  inline DWConvSpCHWMicrokernelTester& output_tuple_stride(uint32_t output_tuple_stride) {
    assert(output_tuple_stride != 0);
    this->output_tuple_stride_ = output_tuple_stride;
    return *this;
  }

  inline uint32_t output_tuple_stride() const {
    if (this->output_tuple_stride_ == 0) {
      return this->output_tuple_size();
    } else {
      return this->output_tuple_stride_;
    }
  }

  inline DWConvSpCHWMicrokernelTester& input_width_stride(uint32_t input_width_stride) {
    assert(input_width_stride != 0);
    this->input_width_stride_ = input_width_stride;
    return *this;
  }

  inline uint32_t input_width_stride() const {
    if (this->input_width_stride_ == 0) {
      return (this->input_width() + input_tuple_size() - 1) / input_tuple_size() * input_tuple_size();
    } else {
      return this->input_width_stride_;
    }
  }

  inline DWConvSpCHWMicrokernelTester& output_width_stride(uint32_t output_width_stride) {
    assert(output_width_stride != 0);
    this->output_width_stride_ = output_width_stride;
    return *this;
  }

  inline uint32_t output_width_stride() const {
    if (this->output_width_stride_ == 0) {
      return (this->output_width() + output_tuple_size() - 1) / output_tuple_size() * output_tuple_size();
    } else {
      return this->output_width_stride_;
    }
  }

  inline DWConvSpCHWMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DWConvSpCHWMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DWConvSpCHWMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_dwconv_spchw_ukernel_function dwconv, Variant variant = Variant::Native) const {
    ASSERT_EQ(0, input_tuple_stride() % input_tuple_size());
    ASSERT_EQ(0, output_tuple_stride() % output_tuple_size());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<float, AlignedAllocator<float, 64>> input((input_height() - 1) * input_width_stride() +
      (input_width() - 1) / input_tuple_size() * input_tuple_stride() + input_tuple_stride() + input_tuple_size());
    std::vector<float> packed_weights(kernel_size() + 1);
    std::vector<float, AlignedAllocator<float, 64>> output((output_height() - 1) * output_width_stride() +
      (output_width() - 1) / output_tuple_size() * output_tuple_stride() + output_tuple_size());
    std::vector<float> output_ref(output_height() * output_width());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(packed_weights.begin(), packed_weights.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      for (size_t oy = 0; oy < output_height(); oy++) {
        for (size_t ox = 0; ox < output_width(); ox++) {
          float acc = packed_weights[0];
          for (size_t ky = 0; ky < kernel_height(); ky++) {
            const size_t iy = oy * subsampling() + ky;
            for (size_t kx = 0; kx < kernel_width(); kx++) {
              const size_t ix = ox * subsampling() + kx - padding_left();
              if (ix < input_width()) {
                acc +=
                  input[iy * input_width_stride() + ix / input_tuple_size() * input_tuple_stride() + ix % input_tuple_size()] *
                  packed_weights[1 + ky * kernel_width() + kx];
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

      // Prepare output parameters.
      xnn_f32_spchw_params spchw_params = { };
      switch (variant) {
        case Variant::Native:
          spchw_params = xnn_init_f32_spchw_params(input_width(), output_min, output_max);
          break;
        case Variant::Scalar:
          spchw_params = xnn_init_scalar_f32_spchw_params(input_width(), output_min, output_max);
          break;
      }

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // Call optimized micro-kernel.
      dwconv(
        output_height(), input_width(),
        input.data(), packed_weights.data(), output.data(),
        input_tuple_stride() * sizeof(float), output_tuple_stride() * sizeof(float),
        input_width_stride() * sizeof(float), output_width_stride() * sizeof(float),
        &spchw_params);

      // Verify results.
      for (size_t y = 0; y < output_height(); y++) {
        for (size_t x = 0; x < output_width(); x++) {
          ASSERT_NEAR(
              output_ref[y * output_width() + x],
              output[y * output_width_stride() + x / output_tuple_size() * output_tuple_stride() + x % output_tuple_size()],
              std::abs(output_ref[y * output_width() + x]) * 1.0e-5)
            << "x = " << x << ", y = " << y;
        }
      }

      // Verify that remainder of the last tile left unchanged.
      if (output_width() % output_tuple_size() != 0) {
        for (size_t i = output.size() - output_tuple_size() + output_width() % output_tuple_size(); i < output.size(); i++) {
          ASSERT_TRUE(std::isnan(output[i]))
            << "i = " << i << ", output = " << output[i];
        }
      }
    }
  }

 private:
  uint32_t input_tuple_size_{1};
  uint32_t output_tuple_size_{1};
  uint32_t padding_left_{0};
  uint32_t padding_right_{0};
  uint32_t output_height_{1};
  uint32_t input_width_{1};
  uint32_t subsampling_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint32_t input_tuple_stride_{0};
  uint32_t output_tuple_stride_{0};
  uint32_t input_width_stride_{0};
  uint32_t output_width_stride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
