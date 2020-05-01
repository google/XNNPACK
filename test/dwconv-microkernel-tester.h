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
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class DWConvMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline DWConvMicrokernelTester& width(uint32_t width) {
    assert(width >= 1);
    this->width_ = width;
    return *this;
  }

  inline uint32_t width() const {
    return this->width_;
  }

  inline DWConvMicrokernelTester& step(uint32_t step) {
    assert(step >= 1);
    this->step_ = step;
    return *this;
  }

  inline uint32_t step() const {
    return this->step_;
  }

  inline DWConvMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  inline uint32_t channels() const {
    return this->channels_;
  }

  inline DWConvMicrokernelTester& cr(uint32_t cr) {
    assert(cr != 0);
    assert((cr & (cr - 1)) == 0);
    this->cr_ = cr;
    return *this;
  }

  inline uint32_t cr() const {
    return this->cr_;
  }

  inline DWConvMicrokernelTester& kr(uint32_t kr) {
    assert(kr != 0);
    this->kr_ = kr;
    return *this;
  }

  inline uint32_t kr() const {
    return this->kr_;
  }

  inline uint32_t packed_channels() const {
    return (channels() / cr() + !!(channels() % cr())) * cr();
  }

  inline DWConvMicrokernelTester& output_stride(uint32_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline uint32_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  inline DWConvMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline DWConvMicrokernelTester& kernel_zero_point(uint8_t kernel_zero_point) {
    this->kernel_zero_point_ = kernel_zero_point;
    return *this;
  }

  inline uint8_t kernel_zero_point() const {
    return this->kernel_zero_point_;
  }

  inline DWConvMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DWConvMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DWConvMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_dwconv_minmax_unipass_ukernel_function dwconv_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<const uint8_t*> indirection((width() - 1) * step() + kr());
    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + indirection.size() * channels());
    std::vector<uint8_t> kernel(channels() * kr());
    std::vector<int32_t> bias(channels());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_weights((kr() + sizeof(int32_t) / sizeof(uint8_t)) * packed_channels());
    std::vector<uint8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<uint8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), std::ref(u8rng));
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(output.begin(), output.end(), 0xA5);

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      xnn_pack_q8_dwconv_ghw_w(
        kr(), 1, channels(), cr(),
        input_zero_point(), kernel_zero_point(),
        kernel.data(), bias.data(), packed_weights.data());
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kr(); k++) {
            acc +=
              (int32_t(indirection[x * step() + k][c]) - int32_t(input_zero_point())) *
              (int32_t(kernel[c * kr() + k]) - int32_t(kernel_zero_point()));
          }
          accumulators[x * channels() + c] = acc;
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());
      const uint32_t accumulated_range = uint32_t(accumulated_max) - uint32_t(accumulated_min);
      const double output_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Prepare convolution parameters.
      const float requantization_scale = 1.0f / float(output_scale);
      union xnn_q8_gemm_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_gemm_params(
            input_zero_point(), kernel_zero_point(),
            requantization_scale, output_zero_point, qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_gemm_params(
            input_zero_point(), kernel_zero_point(),
            requantization_scale, output_zero_point, qmin(), qmax());
          break;
      }
      const union xnn_q31_requantization_params scalar_requantization_params =
        xnn_init_scalar_requantization_params(
          requantization_scale, output_zero_point, qmin(), qmax());

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = xnn_q31_requantize(accumulators[x * channels() + c], scalar_requantization_params);
        }
      }

      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint8_t),
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "x = " << x << ", channel = " << c;
          ASSERT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "x = " << x << ", channel = " << c;
          ASSERT_EQ(uint32_t(output[x * output_stride() + c]), uint32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c];
        }
      }
    }
  }

  void Test(xnn_f32_dwconv_unipass_ukernel_function dwconv) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<const float*> indirection((width() - 1) * step() + kr());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float> kernel(channels() * kr());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights((kr() + 1) * packed_channels());
    std::vector<float> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(output_ref.begin(), output_ref.end(), nanf(""));
      std::fill(output.begin(), output.end(), nanf(""));

      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
      xnn_pack_f32_dwconv_ghw_w(
        kr(), 1, channels(), cr(),
        kernel.data(), bias.data(), packed_weights.data());
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kr(); k++) {
            acc += indirection[x * step() + k][c] * kernel[c * kr() + k];
          }
          output_ref[x * channels() + c] = acc;
        }
      }

      // Call optimized micro-kernel.
      dwconv(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        nullptr);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              output_ref[x * channels() + c],
              output[x * output_stride() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

  void Test(xnn_f32_dwconv_minmax_unipass_ukernel_function dwconv_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

    std::vector<const float*> indirection((width() - 1) * step() + kr());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float> kernel(channels() * kr());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights((kr() + 1) * packed_channels());
    std::vector<float> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(output_ref.begin(), output_ref.end(), nanf(""));
      std::fill(output.begin(), output.end(), nanf(""));

      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
      xnn_pack_f32_dwconv_ghw_w(
        kr(), 1, channels(), cr(),
        kernel.data(), bias.data(), packed_weights.data());
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kr(); k++) {
            acc += indirection[x * step() + k][c] * kernel[c * kr() + k];
          }
          output_ref[x * channels() + c] = acc;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Prepare output parameters.
      xnn_f32_minmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_minmax_params(output_min, output_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_minmax_params(output_min, output_max);
          break;
      }

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "x = " << x << ", channel = " << c;
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "x = " << x << ", channel = " << c;
          ASSERT_NEAR(
              output_ref[x * channels() + c],
              output[x * output_stride() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

 private:
  uint32_t channels_{1};
  uint32_t cr_{1};
  uint32_t kr_{1};
  uint32_t width_{1};
  uint32_t step_{1};
  uint32_t output_stride_{0};
  uint8_t input_zero_point_{127};
  uint8_t kernel_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{3};
};
