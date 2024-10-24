// Copyright 2024 Google LLC
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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/requantization.h"
#include "replicable_random_device.h"

class RDSumMicrokernelTester {
 public:
  RDSumMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const {
    return this->rows_;
  }

  RDSumMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  RDSumMicrokernelTester& channel_tile(size_t channel_tile) {
    assert(channel_tile != 0);
    this->channel_tile_ = channel_tile;
    return *this;
  }

  size_t channel_tile() const {
    return this->channel_tile_;
  }

  RDSumMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  RDSumMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  RDSumMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  RDSumMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const {
    return this->output_scale_;
  }

  RDSumMicrokernelTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  RDSumMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  void Test(xnn_qs8_rdsum_ukernel_fn rdsum,
      xnn_init_qs8_rsum_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    xnnpack::Buffer<int8_t> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES);
    xnnpack::Buffer<int8_t> zero(channels() + XNN_EXTRA_BYTES, 0);
    xnnpack::Buffer<int32_t> output(channels());
    xnnpack::Buffer<int32_t> output_ref(channels());
    {//for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(output.begin(), output.end(), [&]() { return i8dist(rng); });
      // TODO: WHY?!
      std::copy(output.begin(), output.end(), output_ref.begin());

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        for (size_t n = 0; n < rows(); n++) {
          output_ref[c] += int32_t(input[n * input_stride() + c]);
        }
      }

      // Prepare parameters.
      struct xnn_qs8_rsum_params params;
      if (init_params) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      rdsum(rows(), channels(), input.data(), input_stride(), zero.data(), output.data(), &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_EQ(output[c], output_ref[c])
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_qu8_rdsum_ukernel_fn rdsum,
      xnn_init_qs8_rsum_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    xnnpack::Buffer<uint8_t> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES);
    xnnpack::Buffer<uint8_t> zero(channels() + XNN_EXTRA_BYTES, 0);
    xnnpack::Buffer<uint32_t> output(channels());
    xnnpack::Buffer<uint32_t> output_ref(channels());
    {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(output.begin(), output.end(), [&]() { return u8dist(rng); });
      // TODO: WHY?!
      std::copy(output.begin(), output.end(), output_ref.begin());

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        for (size_t n = 0; n < rows(); n++) {
          output_ref[c] += uint32_t(input[n * input_stride() + c]);
        }
      }

      // Prepare parameters.
      struct xnn_qs8_rsum_params params;
      if (init_params) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      rdsum(rows(), channels(), input.data(), input_stride(), zero.data(), output.data(), &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_EQ(output[c], output_ref[c])
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f16_f32acc_rdsum_ukernel_fn rdsum, xnn_init_f16_f32acc_scale_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

    xnnpack::Buffer<xnn_float16> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    xnnpack::Buffer<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16), 0);
    xnnpack::Buffer<float> output(channels());
    xnnpack::Buffer<float> output_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(output.begin(), output.end(), [&]() { return f32dist(rng); });
      // TODO: WHY?!
      std::copy(output.begin(), output.end(), output_ref.begin());

      // Compute reference results, without clamping.
      for (size_t c = 0; c < channels(); c++) {
        float acc = 0.0f;
        for (size_t n = 0; n < rows(); n++) {
          acc += input[n * input_stride() + c];
        }
        output_ref[c] += acc / float(rows());
      }

      // Prepare parameters.
      struct xnn_f16_f32acc_scale_params params;
      init_params(&params, 1.f / float(rows()));

      // Call optimized micro-kernel.
      rdsum(rows(), channels(), input.data(), input_stride() * sizeof(xnn_float16), zero.data(), output.data(), &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_NEAR(output[c], output_ref[c], std::abs(output_ref[c]) * 1.0e-5f)
          << "at position " << c << ", rows = " << rows() << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f32_rdsum_ukernel_fn rdsum, xnn_init_f32_scale_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<float> input((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
    xnnpack::Buffer<float> output(channels());
    xnnpack::Buffer<float> output_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(output.begin(), output.end(), [&]() { return f32dist(rng); });
      // TODO: WHY?!
      std::copy(output.begin(), output.end(), output_ref.begin());

      // Compute reference results.
      for (size_t c = 0; c < channels(); c++) {
        float acc = 0.0f;
        for (size_t n = 0; n < rows(); n++) {
          acc += input[n * input_stride() + c];
        }
        output_ref[c] += acc / static_cast<float>(rows());
      }

      // Prepare parameters.
      struct xnn_f32_scale_params params;
      init_params(&params, 1.0f / static_cast<float>(rows()));

      // Call optimized micro-kernel.
      rdsum(rows(), channels(), input.data(), input_stride() * sizeof(float), zero.data(), output.data(), &params);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
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
  size_t iterations_{3};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
};
