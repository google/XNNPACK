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
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>
#include <xnnpack/requantization.h>


class GAvgPoolMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline GAvgPoolMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline GAvgPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GAvgPoolMicrokernelTester& channel_tile(size_t channel_tile) {
    assert(channel_tile != 0);
    this->channel_tile_ = channel_tile;
    return *this;
  }

  inline size_t channel_tile() const {
    return this->channel_tile_;
  }

  inline GAvgPoolMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  inline GAvgPoolMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline GAvgPoolMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline GAvgPoolMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline GAvgPoolMicrokernelTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline GAvgPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GAvgPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GAvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_gavgpool_minmax_unipass_ukernel_function gavgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (rows() - 1) * input_stride() + channels());
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(channels());
    std::vector<uint8_t> output_ref(channels());
    std::vector<float> output_fp(channels());
    std::vector<int32_t> accumulators(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Prepare quantization parameters.
      union xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(rows()),
            input_scale() / (output_scale() * float(rows())),
            output_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(rows()),
            input_scale() / (output_scale() * float(rows())),
            output_zero_point(), qmin(), qmax());
          break;
      }
      const union xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(input_zero_point()) * int32_t(rows()),
          input_scale() / (output_scale() * float(rows())),
          output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        int32_t acc = scalar_quantization_params.scalar.bias;
        for (size_t n = 0; n < rows(); n++) {
          acc += input[n * input_stride() + c];
        }
        accumulators[c] = acc;
        output_ref[c] = xnn_avgpool_quantize(acc, scalar_quantization_params);
        output_fp[c] = float(acc) * (input_scale() / (output_scale() * float(rows()))) + float(output_zero_point());
        output_fp[c] = std::min<float>(output_fp[c], float(qmax()));
        output_fp[c] = std::max<float>(output_fp[c], float(qmin()));
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(uint8_t),
        zero.data(),
        output.data(),
        &quantization_params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(uint32_t(output[c]), uint32_t(qmax()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(uint32_t(output[c]), uint32_t(qmin()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_NEAR(float(int32_t(output[c])), output_fp[c], 0.5f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
        ASSERT_EQ(uint32_t(output_ref[c]), uint32_t(output[c]))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
      }
    }
  }

  void Test(xnn_q8_gavgpool_minmax_multipass_ukernel_function gavgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (rows() - 1) * input_stride() + channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(channels());
    std::vector<uint8_t> output_ref(channels());
    std::vector<float> output_fp(channels());
    std::vector<int32_t> accumulators(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Prepare quantization parameters.
      union xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(rows()),
            input_scale() / (output_scale() * float(rows())),
            output_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(rows()),
            input_scale() / (output_scale() * float(rows())),
            output_zero_point(), qmin(), qmax());
          break;
      }
      const union xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(input_zero_point()) * int32_t(rows()),
          input_scale() / (output_scale() * float(rows())),
          output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        int32_t acc = scalar_quantization_params.scalar.bias;
        for (size_t n = 0; n < rows(); n++) {
          acc += input[n * input_stride() + c];
        }

        accumulators[c] = acc;
        output_ref[c] = xnn_avgpool_quantize(acc, scalar_quantization_params);
        output_fp[c] = float(acc) * (input_scale() / (output_scale() * float(rows()))) + float(output_zero_point());
        output_fp[c] = std::min<float>(output_fp[c], float(qmax()));
        output_fp[c] = std::max<float>(output_fp[c], float(qmin()));
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(uint8_t),
        zero.data(),
        buffer.data(),
        output.data(),
        &quantization_params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(uint32_t(output[c]), uint32_t(qmax()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(uint32_t(output[c]), uint32_t(qmin()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_NEAR(float(int32_t(output[c])), output_fp[c], 0.5f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
        ASSERT_EQ(uint32_t(output_ref[c]), uint32_t(output[c]))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
      }
    }
  }

  void Test(xnn_f32_gavgpool_minmax_unipass_ukernel_function gavgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(channels());
    std::vector<float> output_ref(channels());

    std::fill(zero.begin(), zero.end(), 0.0f);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        float acc = 0.0f;
        for (size_t n = 0; n < rows(); n++) {
          acc += input[n * input_stride() + c];
        }
        output_ref[c] = acc / float(rows());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_values : output_ref) {
        output_values = std::max(std::min(output_values, output_max), output_min);
      }

      // Prepare micro-kernel parameters.
      union xnn_f32_scaleminmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_scaleminmax_params(
            1.0f / float(rows()), output_min, output_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_scaleminmax_params(
            1.0f / float(rows()), output_min, output_max);
          break;
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(float),
        zero.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(output[c], output_max)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(output[c], output_min)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_NEAR(output[c], output_ref[c], std::abs(output_ref[c]) * 1.0e-6f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f32_gavgpool_minmax_multipass_ukernel_function gavgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> buffer(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(channels());
    std::vector<float> output_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        float acc = 0.0f;
        for (size_t n = 0; n < rows(); n++) {
          acc += input[n * input_stride() + c];
        }
        output_ref[c] = acc / float(rows());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Prepare micro-kernel parameters.
      union xnn_f32_scaleminmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_scaleminmax_params(
            1.0f / float(rows()), output_min, output_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_scaleminmax_params(
            1.0f / float(rows()), output_min, output_max);
          break;
      }

      // Clamp reference results.
      for (float& output_values : output_ref) {
        output_values = std::max(std::min(output_values, output_max), output_min);
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(float),
        zero.data(),
        buffer.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(output[c], output_max)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(output[c], output_min)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_NEAR(output[c], output_ref[c], std::abs(output_ref[c]) * 1.0e-6f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t channel_tile_{1};
  size_t input_stride_{0};
  float input_scale_{1.25f};
  float output_scale_{0.75f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
