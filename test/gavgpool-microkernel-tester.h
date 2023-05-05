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
#include <limits>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/requantization.h>


class GAvgPoolMicrokernelTester {
 public:
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

  void Test(
      xnn_qu8_gavgpool_minmax_unipass_ukernel_fn gavgpool_minmax,
      xnn_init_qu8_avgpool_minmax_params_fn init_params,
      xnn_qu8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (rows() - 1) * input_stride() + channels());
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(channels());
    std::vector<uint8_t> output_ref(channels());
    std::vector<float> output_fp(channels());
    std::vector<int32_t> accumulators(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Prepare parameters.
      union xnn_qu8_avgpool_minmax_params params;
      init_params(
        &params,
        -int32_t(input_zero_point()) * int32_t(rows()),
        input_scale() / (output_scale() * float(rows())),
        output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        int32_t acc = 0;
        for (size_t n = 0; n < rows(); n++) {
          acc += int32_t(input[n * input_stride() + c]) - int32_t(input_zero_point());
        }
        accumulators[c] = acc;
        output_ref[c] = requantize(
          acc, input_scale() / (output_scale() * float(rows())), output_zero_point(), qmin(), qmax());
        output_fp[c] = float(acc) * (input_scale() / (output_scale() * float(rows()))) + float(output_zero_point());
        output_fp[c] = std::min<float>(output_fp[c], float(qmax()));
        output_fp[c] = std::max<float>(output_fp[c], float(qmin()));
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(uint8_t),
        zero.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(uint32_t(output[c]), uint32_t(qmax()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(uint32_t(output[c]), uint32_t(qmin()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_NEAR(float(int32_t(output[c])), output_fp[c], 0.55f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
        EXPECT_EQ(uint32_t(output_ref[c]), uint32_t(output[c]))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
      }
    }
  }

  void Test(
      xnn_qu8_gavgpool_minmax_multipass_ukernel_fn gavgpool_minmax,
      xnn_init_qu8_avgpool_minmax_params_fn init_params,
      xnn_qu8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (rows() - 1) * input_stride() + channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(channels());
    std::vector<uint8_t> output_ref(channels());
    std::vector<float> output_fp(channels());
    std::vector<int32_t> accumulators(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Prepare parameters.
      union xnn_qu8_avgpool_minmax_params params;
      init_params(
        &params,
        -int32_t(input_zero_point()) * int32_t(rows()),
        input_scale() / (output_scale() * float(rows())),
        output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        int32_t acc = 0;
        for (size_t n = 0; n < rows(); n++) {
          acc += int32_t(input[n * input_stride() + c]) - int32_t(input_zero_point());
        }

        accumulators[c] = acc;
        output_ref[c] = requantize(
          acc, input_scale() / (output_scale() * float(rows())), output_zero_point(), qmin(), qmax());
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
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(uint32_t(output[c]), uint32_t(qmax()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(uint32_t(output[c]), uint32_t(qmin()))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_NEAR(float(int32_t(output[c])), output_fp[c], 0.55f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
        EXPECT_EQ(uint32_t(output_ref[c]), uint32_t(output[c]))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels()
          << ", acc = " << accumulators[c];
      }
    }
  }

  void Test(
      xnn_qs8_gavgpool_minmax_unipass_ukernel_fn gavgpool_minmax,
      xnn_init_qs8_avgpool_minmax_params_fn init_params,
      xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (rows() - 1) * input_stride() + channels());
    std::vector<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(channels());
    std::vector<int8_t> output_ref(channels());
    std::vector<float> output_fp(channels());
    std::vector<int32_t> accumulators(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Prepare parameters.
      union xnn_qs8_avgpool_minmax_params params;
      init_params(
        &params,
        -int32_t(input_zero_point() - 0x80) * int32_t(rows()),
        input_scale() / (output_scale() * float(rows())),
        int8_t(output_zero_point() - 0x80), int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        int32_t acc = 0;
        for (size_t n = 0; n < rows(); n++) {
          acc += int32_t(input[n * input_stride() + c]) - int32_t(input_zero_point() - 0x80);
        }
        accumulators[c] = acc;
        output_ref[c] = requantize(
          acc, input_scale() / (output_scale() * float(rows())), int8_t(output_zero_point() - 0x80), int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
        output_fp[c] = float(acc) * (input_scale() / (output_scale() * float(rows()))) + float(output_zero_point() - 0x80);
        output_fp[c] = std::min<float>(output_fp[c], float(qmax() - 0x80));
        output_fp[c] = std::max<float>(output_fp[c], float(qmin() - 0x80));
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(int8_t),
        zero.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(int32_t(output[c]), int32_t(qmax() - 0x80))
          << "at channel " << c << " / " << channels() << ", rows = " << rows();
        ASSERT_GE(int32_t(output[c]), int32_t(qmin() - 0x80))
          << "at channel " << c << " / " << channels() << ", rows = " << rows();
        ASSERT_NEAR(float(int32_t(output[c])), output_fp[c], 0.55f)
          << "at channel " << c << " / " << channels() << ", rows = " << rows()
          << ", accumulator = " << accumulators[c];
        EXPECT_EQ(int32_t(output_ref[c]), int32_t(output[c]))
          << "at channel " << c << " / " << channels() << ", rows = " << rows()
          << ", accumulator = " << accumulators[c];
      }
    }
  }

  void Test(
      xnn_qs8_gavgpool_minmax_multipass_ukernel_fn gavgpool_minmax,
      xnn_init_qs8_avgpool_minmax_params_fn init_params,
      xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (rows() - 1) * input_stride() + channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(channels());
    std::vector<int8_t> output_ref(channels());
    std::vector<float> output_fp(channels());
    std::vector<int32_t> accumulators(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Prepare parameters.
      union xnn_qs8_avgpool_minmax_params params;
      init_params(
        &params,
        -int32_t(input_zero_point() - 0x80) * int32_t(rows()),
        input_scale() / (output_scale() * float(rows())),
        int8_t(output_zero_point() - 0x80), int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        int32_t acc = 0;
        for (size_t n = 0; n < rows(); n++) {
          acc += int32_t(input[n * input_stride() + c]) - int32_t(input_zero_point() - 0x80);
        }
        accumulators[c] = acc;
        output_ref[c] = requantize(
          acc, input_scale() / (output_scale() * float(rows())), int8_t(output_zero_point() - 0x80), int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
        output_fp[c] = float(acc) * (input_scale() / (output_scale() * float(rows()))) + float(output_zero_point() - 0x80);
        output_fp[c] = std::min<float>(output_fp[c], float(qmax() - 0x80));
        output_fp[c] = std::max<float>(output_fp[c], float(qmin() - 0x80));
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(int8_t),
        zero.data(),
        buffer.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(int32_t(output[c]), int32_t(qmax() - 0x80))
          << "at channel " << c << " / " << channels() << ", rows = " << rows();
        ASSERT_GE(int32_t(output[c]), int32_t(qmin() - 0x80))
          << "at channel " << c << " / " << channels() << ", rows = " << rows();
        ASSERT_NEAR(float(int32_t(output[c])), output_fp[c], 0.55f)
          << "at channel " << c << " / " << channels() << ", rows = " << rows()
          << ", accumulator = " << accumulators[c];
        EXPECT_EQ(int32_t(output_ref[c]), int32_t(output[c]))
          << "at channel " << c << " / " << channels() << ", rows = " << rows()
          << ", accumulator = " << accumulators[c];
      }
    }
  }

  void Test(xnn_f16_gavgpool_minmax_unipass_ukernel_fn gavgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<uint16_t> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(channels());
    std::vector<float> output_ref(channels());

    std::fill(zero.begin(), zero.end(), 0);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        float acc = 0.0f;
        for (size_t n = 0; n < rows(); n++) {
          acc += fp16_ieee_to_fp32_value(input[n * input_stride() + c]);
        }
        output_ref[c] = acc / float(rows());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + float(qmin()) / 255.0f * accumulated_range));
      const float output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range));

      // Clamp reference results.
      for (float& output_values : output_ref) {
        output_values = std::max(std::min(output_values, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params,
        fp16_ieee_from_fp32_value(1.0f / float(rows())),
        fp16_ieee_from_fp32_value(output_min),
        fp16_ieee_from_fp32_value(output_max));

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(uint16_t),
        zero.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(fp16_ieee_to_fp32_value(output[c]), output_max)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(fp16_ieee_to_fp32_value(output[c]), output_min)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        EXPECT_NEAR(fp16_ieee_to_fp32_value(output[c]), output_ref[c], std::max(1.0e-4f, std::abs(output_ref[c]) * 1.0e-2f))
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f16_gavgpool_minmax_multipass_ukernel_fn gavgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<uint16_t> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> buffer(channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(channels());
    std::vector<float> output_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        float acc = 0.0f;
        for (size_t n = 0; n < rows(); n++) {
          acc += fp16_ieee_to_fp32_value(input[n * input_stride() + c]);
        }
        output_ref[c] = acc / float(rows());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + float(qmin()) / 255.0f * accumulated_range));
      const float output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range));

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params,
        fp16_ieee_from_fp32_value(1.0f / float(rows())),
        fp16_ieee_from_fp32_value(output_min),
        fp16_ieee_from_fp32_value(output_max));

      // Clamp reference results.
      for (float& output_values : output_ref) {
        output_values = std::max(std::min(output_values, output_max), output_min);
      }

      // Call optimized micro-kernel.
      gavgpool_minmax(rows(), channels(),
        input.data(), input_stride() * sizeof(uint16_t),
        zero.data(),
        buffer.data(),
        output.data(),
        &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_LE(fp16_ieee_to_fp32_value(output[c]), output_max)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        ASSERT_GE(fp16_ieee_to_fp32_value(output[c]), output_min)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
        EXPECT_NEAR(fp16_ieee_to_fp32_value(output[c]), output_ref[c], std::abs(output_ref[c]) * 1.0e-0f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f32_gavgpool_minmax_unipass_ukernel_fn gavgpool_minmax, xnn_init_f32_scaleminmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(channels());
    std::vector<float> output_ref(channels());

    std::fill(zero.begin(), zero.end(), 0.0f);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
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

      // Prepare parameters.
      union xnn_f32_scaleminmax_params params;
      init_params(&params, 1.0f / float(rows()), output_min, output_max);

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
        EXPECT_NEAR(output[c], output_ref[c], std::abs(output_ref[c]) * 1.0e-6f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f32_gavgpool_minmax_multipass_ukernel_fn gavgpool_minmax, xnn_init_f32_scaleminmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> buffer(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(channels());
    std::vector<float> output_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
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

      // Prepare parameters.
      union xnn_f32_scaleminmax_params params;
      init_params(&params, 1.0f / float(rows()), output_min, output_max);

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
        EXPECT_NEAR(output[c], output_ref[c], std::abs(output_ref[c]) * 1.0e-6f)
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
