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
#include <xnnpack/common.h>
#include <xnnpack/microkernel-utils.h>
#include <xnnpack/pack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/requantization.h>


class DWConvMicrokernelTester {
 public:
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

  inline DWConvMicrokernelTester& channel_tile(uint32_t channel_tile) {
    assert(channel_tile != 0);
    this->channel_tile_ = channel_tile;
    return *this;
  }

  inline uint32_t channel_tile() const {
    return this->channel_tile_;
  }

  inline DWConvMicrokernelTester& channel_subtile(uint32_t channel_subtile) {
    assert(channel_subtile != 0);
    this->channel_subtile_ = channel_subtile;
    return *this;
  }

  inline uint32_t channel_subtile() const {
    return this->channel_subtile_;
  }

  inline DWConvMicrokernelTester& channel_round(uint32_t channel_round) {
    assert(channel_round != 0);
    this->channel_round_ = channel_round;
    return *this;
  }

  inline uint32_t channel_round() const {
    return this->channel_round_;
  }

  inline DWConvMicrokernelTester& kernel_tile(uint32_t kernel_tile) {
    assert(kernel_tile != 0);
    this->kernel_tile_ = kernel_tile;
    return *this;
  }

  inline uint32_t kernel_tile() const {
    return this->kernel_tile_;
  }

  inline DWConvMicrokernelTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size != 0);
    this->kernel_size_ = kernel_size;
    return *this;
  }

  inline uint32_t kernel_size() const {
    return this->kernel_size_;
  }

  inline uint32_t packed_channels() const {
    return (channels() / channel_tile() + !!(channels() % channel_tile())) * channel_tile();
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

  inline DWConvMicrokernelTester& input_offset(size_t input_offset) {
    this->input_offset_ = input_offset;
    return *this;
  }

  inline size_t input_offset() const {
    return this->input_offset_;
  }

  inline DWConvMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  inline size_t zero_index() const {
    return this->zero_index_;
  }

  inline DWConvMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  inline DWConvMicrokernelTester& first_pass_tile(size_t first_pass_tile) {
    this->first_pass_tile_ = first_pass_tile;
    return *this;
  }

  inline size_t first_pass_tile() const {
    return this->first_pass_tile_;
  }

  inline DWConvMicrokernelTester& middle_pass_tile(size_t middle_pass_tile) {
    this->middle_pass_tile_ = middle_pass_tile;
    return *this;
  }

  inline size_t middle_pass_tile() const {
    return this->middle_pass_tile_;
  }

  inline DWConvMicrokernelTester& last_pass_tile(size_t last_pass_tile) {
    this->last_pass_tile_ = last_pass_tile;
    return *this;
  }

  inline size_t last_pass_tile() const {
    return this->last_pass_tile_;
  }

  void Test(
    xnn_qu8_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_qu8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<const uint8_t*> indirection((width() - 1) * step() + kernel_tile());
    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + indirection.size() * channels());
    std::vector<uint8_t> kernel(channels() * kernel_tile());
    std::vector<int32_t> bias(channels());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_weights((kernel_tile() + sizeof(int32_t) / sizeof(uint8_t)) * packed_channels());
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<uint8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(zero.begin(), zero.end(), input_zero_point());
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      std::fill(packed_weights.begin(), packed_weights.end(), kernel_zero_point());
      const xnn_qu8_packing_params packing_params = { input_zero_point(), kernel_zero_point() };
      xnn_pack_qu8_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
        channel_tile(), channel_tile(), channel_tile(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, &packing_params);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_tile(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc +=
                (int32_t(indirection[x * step() + k][c + input_offset()]) - int32_t(input_zero_point())) *
                (int32_t(kernel[c * kernel_tile() + k]) - int32_t(kernel_zero_point()));
            }
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

      // Prepare parameters.
      const float requantization_scale = 1.0f / float(output_scale);
      union xnn_qu8_conv_minmax_params quantization_params;
      init_params(&quantization_params,
        kernel_zero_point(), requantization_scale, output_zero_point, qmin(), qmax());

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = requantize(
            accumulators[x * channels() + c], requantization_scale, output_zero_point, qmin(), qmax());
        }
      }

      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint8_t),
        input_offset() * sizeof(uint8_t), zero.data(),
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "x = " << x << ", channel = " << c;
          EXPECT_EQ(uint32_t(output[x * output_stride() + c]), uint32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c];
        }
      }
    }
  }

  void Test(
    xnn_qu8_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_qu8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
    std::vector<const uint8_t*> indirection((width() - 1) * step() + tile_size);
    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + indirection.size() * channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(XNN_MULTIPASS_EXTRA_BYTES / sizeof(uint8_t) + channels());
    std::vector<uint8_t> kernel(channels() * kernel_size());
    std::vector<int32_t> bias(channels());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> packed_weights(
      xnn_dwconv_multipass_weights_size(
        tile_size, channels(), channel_tile(), channel_subtile(), channel_round(),
        /*bias_element_size=*/4, /*log2_filter_element_size=*/0, /*extra_weights_byte=*/0));
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<uint8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(zero.begin(), zero.end(), input_zero_point());
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      std::fill(packed_weights.begin(), packed_weights.end(), kernel_zero_point());
      const xnn_qu8_packing_params packing_params = { input_zero_point(), kernel_zero_point() };
      xnn_pack_qu8_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(), 1,
        channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, &packing_params);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_size()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_size(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc +=
                (int32_t(indirection[x * step() + k][c + input_offset()]) - int32_t(input_zero_point())) *
                (int32_t(kernel[c * kernel_size() + k]) - int32_t(kernel_zero_point()));
            }
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

      // Prepare parameters.
      const float requantization_scale = 1.0f / float(output_scale);
      union xnn_qu8_conv_minmax_params quantization_params;
      init_params(&quantization_params,
        kernel_zero_point(), requantization_scale, output_zero_point, qmin(), qmax());

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = requantize(
            accumulators[x * channels() + c], requantization_scale, output_zero_point, qmin(), qmax());
        }
      }

      // input_stride is step() - first and middle pass
      size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()), middle_pass_tile());
      const int input_advanced = first_pass_tile() + num_middle_pass * middle_pass_tile();
      int input_stride_elements = step() - input_advanced;
      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        input_stride_elements * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint8_t),
        input_offset() * sizeof(uint8_t), zero.data(),
        kernel_size(),
        buffer.data(),
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "x = " << x << ", channel = " << c;
          EXPECT_EQ(uint32_t(output[x * output_stride() + c]), uint32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c];
        }
      }
    }
  }

  void Test(
    xnn_qs8_qc8w_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<const int8_t*> indirection((width() - 1) * step() + kernel_tile());
    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + indirection.size() * channels());
    std::vector<int8_t> kernel(channels() * kernel_tile());
    std::vector<int32_t> bias(channels());
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_weights((kernel_tile() + (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t)) * packed_channels());
    std::vector<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<float> scale(channels());
    std::vector<int8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(zero.begin(), zero.end(), int8_t(input_zero_point() - 0x80));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      const xnn_qs8_packing_params packing_params = { int8_t(input_zero_point() - 0x80) };
      xnn_pack_qs8_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
        channel_tile(), channel_tile(), channel_tile(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/ channel_tile() * sizeof(float),
        /*per_subtile_extra_bytes=*/ channel_tile() * sizeof(float),
        &packing_params);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_tile(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc +=
                (int32_t(indirection[x * step() + k][c + input_offset()]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[c * kernel_tile() + k]);
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }

      // Compute renormalization parameters.
      const int8_t output_zero_point = -1;
      for (size_t c = 0; c < channels(); c++) {
        int32_t accumulated_min = accumulators[c];
        int32_t accumulated_max = accumulators[c];
        for (size_t x = 0; x < width(); x++) {
          accumulated_min = std::min(accumulated_min, accumulators[x * channels() + c]);
          accumulated_max = std::max(accumulated_max, accumulators[x * channels() + c]);
        }
        const uint32_t accumulated_range = uint32_t(accumulated_max - accumulated_min);
        const float output_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
        scale[c] = 1.0f / output_scale;
      }
      xnn_init_qs8_qc8w_scale_fp32_params(
        channels(), channel_tile(), channel_tile(),
        channel_tile() * (kernel_tile() * sizeof(int8_t) + sizeof(int32_t) + sizeof(float)),
        channel_tile() * (kernel_tile() * sizeof(int8_t) + sizeof(int32_t) + sizeof(float)),
        0,
        scale.data(),
        (void*) ((uintptr_t) packed_weights.data() + channel_tile() * (kernel_tile() * sizeof(int8_t) + sizeof(int32_t))));

      // Prepare parameters.
      union xnn_qs8_qc8w_conv_minmax_params minmax_params;
      init_params(&minmax_params,
        output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = requantize(
            accumulators[x * channels() + c], scale[c], output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
        }
      }

      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(int8_t),
        input_offset() * sizeof(int8_t), zero.data(),
        &minmax_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(int32_t(output[x * output_stride() + c]), int32_t(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(int32_t(output[x * output_stride() + c]), int32_t(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_EQ(int32_t(output[x * output_stride() + c]), int32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c];
        }
      }
    }
  }

  void Test(
    xnn_qs8_qc8w_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
    std::vector<const int8_t*> indirection((width() - 1) * step() + tile_size);
    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + indirection.size() * channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(XNN_MULTIPASS_EXTRA_BYTES / sizeof(int8_t) + channels());
    std::vector<int8_t> kernel(channels() * kernel_size());
    std::vector<int32_t> bias(channels());
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_weights(
      xnn_dwconv_multipass_weights_size(
        tile_size, channels(), channel_tile(), channel_subtile(), channel_round(),
        /*bias_element_size=*/4, /*log2_filter_element_size=*/0, /*extra_weights_byte=*/sizeof(float)));
    std::vector<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<float> scale(channels());
    std::vector<int8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(zero.begin(), zero.end(), int8_t(input_zero_point() - 0x80));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      const xnn_qs8_packing_params packing_params = { int8_t(input_zero_point() - 0x80) };
      xnn_pack_qs8_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(), 1,
        channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/channel_tile() * sizeof(float),
        /*per_subtile_extra_bytes=*/channel_subtile() * sizeof(float),
        &packing_params);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_size()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_size(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc +=
                (int32_t(indirection[x * step() + k][c + input_offset()]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[c * kernel_size() + k]);
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }

      // Compute renormalization parameters.
      const int8_t output_zero_point = -1;
      for (size_t c = 0; c < channels(); c++) {
        int32_t accumulated_min = accumulators[c];
        int32_t accumulated_max = accumulators[c];
        for (size_t x = 0; x < width(); x++) {
          accumulated_min = std::min(accumulated_min, accumulators[x * channels() + c]);
          accumulated_max = std::max(accumulated_max, accumulators[x * channels() + c]);
        }
        const uint32_t accumulated_range = uint32_t(accumulated_max - accumulated_min);
        const float output_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
        scale[c] = 1.0f / output_scale;
      }

      size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()), middle_pass_tile());
      const size_t rounded_c = round_up_po2(channels(), channel_subtile());
      const size_t packed_weights_offset_to_last_tile =
          first_pass_tile() * rounded_c * sizeof(int8_t) + rounded_c * sizeof(int32_t) +
          num_middle_pass * middle_pass_tile() * rounded_c * sizeof(int8_t) +
          last_pass_tile() * channel_tile();

      xnn_init_qs8_qc8w_scale_fp32_params(
        channels(), channel_tile(), channel_subtile(),
        channel_tile() * (last_pass_tile() * sizeof(int8_t) + sizeof(int32_t)),
        channel_subtile() * (last_pass_tile() * sizeof(int8_t) + sizeof(int32_t)),
        (channel_tile()- channel_subtile()) * last_pass_tile() * sizeof(int8_t),
        scale.data(),
        (void*) ((uintptr_t) packed_weights.data() + packed_weights_offset_to_last_tile));

      // Prepare parameters.
      union xnn_qs8_qc8w_conv_minmax_params minmax_params;
      init_params(&minmax_params,
        output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = requantize(
            accumulators[x * channels() + c], scale[c], output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
        }
      }

      // input_stride is step() - first and middle pass
      const int input_advanced = first_pass_tile() + num_middle_pass * middle_pass_tile();
      int input_stride_elements = step() - input_advanced;
      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        input_stride_elements * sizeof(void*),
        (output_stride() - channels()) * sizeof(int8_t),
        input_offset() * sizeof(int8_t), zero.data(),
        kernel_size(),
        buffer.data(),
        &minmax_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(int32_t(output[x * output_stride() + c]), int32_t(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(int32_t(output[x * output_stride() + c]), int32_t(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_EQ(int32_t(output[x * output_stride() + c]), int32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c]
            << ", channels = " << channels();
        }
      }
    }
  }

  void Test(
    xnn_qs8_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    std::vector<const int8_t*> indirection((width() - 1) * step() + kernel_tile());
    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + indirection.size() * channels());
    std::vector<int8_t> kernel(channels() * kernel_tile());
    std::vector<int32_t> bias(channels());
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_weights((kernel_tile() + sizeof(int32_t) / sizeof(int8_t)) * packed_channels());
    std::vector<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<int8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(zero.begin(), zero.end(), int8_t(input_zero_point() - 0x80));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      const xnn_qs8_packing_params packing_params = { int8_t(input_zero_point() - 0x80) };
      xnn_pack_qs8_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
        channel_tile(), channel_tile(), channel_tile(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, &packing_params);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_tile(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc +=
                (int32_t(indirection[x * step() + k][c + input_offset()]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[c * kernel_tile() + k]);
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());
      const uint32_t accumulated_range = uint32_t(accumulated_max) - uint32_t(accumulated_min);
      const double output_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
      const int8_t output_zero_point = int8_t(std::max(std::min(
        lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

      // Prepare parameters.
      const float requantization_scale = 1.0f / float(output_scale);
      union xnn_qs8_conv_minmax_params quantization_params;
      init_params(&quantization_params,
        requantization_scale, output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = requantize(
            accumulators[x * channels() + c], requantization_scale, output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
        }
      }

      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(int8_t),
        input_offset() * sizeof(int8_t), zero.data(),
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(int32_t(output[x * output_stride() + c]), int32_t(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(int32_t(output[x * output_stride() + c]), int32_t(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_EQ(int32_t(output[x * output_stride() + c]), int32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c];
        }
      }
    }
  }

  void Test(
    xnn_qs8_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

    const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
    std::vector<const int8_t*> indirection((width() - 1) * step() + tile_size);
    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + indirection.size() * channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(XNN_MULTIPASS_EXTRA_BYTES / sizeof(int8_t) + channels());
    std::vector<int8_t> kernel(channels() * kernel_size());
    std::vector<int32_t> bias(channels());
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_weights(
      xnn_dwconv_multipass_weights_size(
        tile_size, channels(), channel_tile(), channel_subtile(), channel_round(),
        /*bias_element_size=*/4, /*log2_filter_element_size=*/0, /*extra_weights_byte=*/0));
    std::vector<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((width() - 1) * output_stride() + channels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<int8_t> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      do {
        std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
      } while (kernel.size() > 1 && *std::max_element(kernel.cbegin(), kernel.cend()) == *std::min_element(kernel.cbegin(), kernel.cend()));
      std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
      std::fill(zero.begin(), zero.end(), int8_t(input_zero_point() - 0x80));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      const xnn_qs8_packing_params packing_params = { int8_t(input_zero_point() - 0x80) };
      xnn_pack_qs8_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(), 1,
        channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, &packing_params);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_size()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without renormalization.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_size(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc +=
                (int32_t(indirection[x * step() + k][c + input_offset()]) - int32_t(input_zero_point() - 0x80)) *
                int32_t(kernel[c * kernel_size() + k]);
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());
      const uint32_t accumulated_range = uint32_t(accumulated_max) - uint32_t(accumulated_min);
      const double output_scale = accumulated_range >= 256 ? double(accumulated_range) / 255.0 : 1.00001;
      const int8_t output_zero_point = int8_t(std::max(std::min(
        lrint(-0.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<int8_t>::max())), long(std::numeric_limits<int8_t>::min())));

      // Prepare parameters.
      const float requantization_scale = 1.0f / float(output_scale);
      union xnn_qs8_conv_minmax_params quantization_params;
      init_params(&quantization_params,
        requantization_scale, output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Renormalize reference results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          output_ref[x * channels() + c] = requantize(
            accumulators[x * channels() + c], requantization_scale, output_zero_point, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));
        }
      }

      // input_stride is step() - first and middle pass
      size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()), middle_pass_tile());
      const int input_advanced = first_pass_tile() + num_middle_pass * middle_pass_tile();
      int input_stride_elements = step() - input_advanced;
      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        input_stride_elements * sizeof(void*),
        (output_stride() - channels()) * sizeof(int8_t),
        input_offset() * sizeof(int8_t), zero.data(),
        kernel_size(),
        buffer.data(),
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(int32_t(output[x * output_stride() + c]), int32_t(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(int32_t(output[x * output_stride() + c]), int32_t(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
          EXPECT_EQ(int32_t(output[x * output_stride() + c]), int32_t(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c << ", accumulator = " << accumulators[x * channels() + c];
        }
      }
    }
  }

  void Test(xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv_minmax, xnn_init_f16_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<const uint16_t*> indirection((width() - 1) * step() + kernel_tile());
    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + indirection.size() * channels());
    std::vector<uint16_t> kernel(channels() * kernel_tile());
    std::vector<uint16_t> bias(channels());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_weights((kernel_tile() + 1) * packed_channels());
    std::vector<uint16_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(zero.begin(), zero.end(), 0);
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      xnn_pack_f16_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
        channel_tile(), channel_tile(), channel_tile(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = fp16_ieee_to_fp32_value(bias[c]);
          for (size_t k = 0; k < kernel_tile(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc += fp16_ieee_to_fp32_value(indirection[x * step() + k][c + input_offset()]) * fp16_ieee_to_fp32_value(kernel[c * kernel_tile() + k]);
            }
          }
          output_ref[x * channels() + c] = acc;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params,
        fp16_ieee_from_fp32_value(output_min),
        fp16_ieee_from_fp32_value(output_max));

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        reinterpret_cast<const void**>(indirection.data()), packed_weights.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint16_t),
        input_offset() * sizeof(uint16_t), zero.data(),
        &params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_min)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_max)
            << "x = " << x << ", channel = " << c;
          EXPECT_NEAR(output_ref[x * channels() + c], fp16_ieee_to_fp32_value(output[x * output_stride() + c]), std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 1.0e-2f))
            << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

  void Test(xnn_f16_dwconv_minmax_multipass_ukernel_fn dwconv_minmax, xnn_init_f16_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
    std::vector<const uint16_t*> indirection((width() - 1) * step() + tile_size);
    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + indirection.size() * channels());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> buffer(XNN_MULTIPASS_EXTRA_BYTES / sizeof(uint16_t) + channels());
    std::vector<uint16_t> kernel(channels() * kernel_size());
    std::vector<uint16_t> bias(channels());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_weights(
      xnn_dwconv_multipass_weights_size(
        tile_size, channels(), channel_tile(), channel_subtile(), channel_round(),
        /*bias_element_size=*/sizeof(uint16_t), /*log2_filter_element_size=*/1, /*extra_weights_byte=*/0) /
      sizeof(uint16_t));
    std::vector<uint16_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(zero.begin(), zero.end(), 0);
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      std::fill(packed_weights.begin(), packed_weights.end(), 0);
      xnn_pack_f16_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(), 1,
        channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_size()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = fp16_ieee_to_fp32_value(bias[c]);
          for (size_t k = 0; k < kernel_size(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc += fp16_ieee_to_fp32_value(indirection[x * step() + k][c + input_offset()]) *
                     fp16_ieee_to_fp32_value(kernel[c * kernel_size() + k]);
            }
          }
          output_ref[x * channels() + c] = acc;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = fp16_ieee_to_fp32_value(
        fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float output_max = fp16_ieee_to_fp32_value(
        fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params,
        fp16_ieee_from_fp32_value(output_min),
        fp16_ieee_from_fp32_value(output_max));

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // input_stride is step() - first and middle pass
      size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()), middle_pass_tile());
      const int input_advanced = first_pass_tile() + num_middle_pass * middle_pass_tile();
      int input_stride_elements = step() - input_advanced;
      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        reinterpret_cast<const void**>(indirection.data()), packed_weights.data(), output.data(),
        input_stride_elements * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint16_t),
        input_offset() * sizeof(uint16_t), zero.data(),
        kernel_size(),
        buffer.data(),
        &params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_min)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_max)
            << "x = " << x << ", channel = " << c;
          EXPECT_NEAR(output_ref[x * channels() + c], fp16_ieee_to_fp32_value(output[x * output_stride() + c]),
                      std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 1.0e-2f))
            << "x = " << x << ", channel = " << c << ", channels = " << channels()
            << ", kernel_size = " << kernel_size() << ", first_pass_tile = " << first_pass_tile()
            << ", middle_pass_tile = " << middle_pass_tile() << ", last_pass_tile = " << last_pass_tile();
        }
      }
    }
  }

  void Test(xnn_f32_dwconv_unipass_ukernel_fn dwconv) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<const float*> indirection((width() - 1) * step() + kernel_tile());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float> kernel(channels() * kernel_tile());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights((kernel_tile() + 1) * packed_channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(zero.begin(), zero.end(), 0.0f);
      std::fill(output_ref.begin(), output_ref.end(), nanf(""));
      std::fill(output.begin(), output.end(), nanf(""));

      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
      xnn_pack_f32_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
        channel_tile(), channel_tile(), channel_tile(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_tile(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc += indirection[x * step() + k][c + input_offset()] * kernel[c * kernel_tile() + k];
            }
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
        input_offset() * sizeof(float), zero.data(),
        nullptr);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_NEAR(
              output_ref[x * channels() + c],
              output[x * output_stride() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

  void Test(xnn_f32_dwconv_minmax_unipass_ukernel_fn dwconv_minmax, xnn_init_f32_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<const float*> indirection((width() - 1) * step() + kernel_tile());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float> kernel(channels() * kernel_tile());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights((kernel_tile() + 1) * packed_channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(zero.begin(), zero.end(), 0.0f);
      std::fill(output_ref.begin(), output_ref.end(), nanf(""));
      std::fill(output.begin(), output.end(), nanf(""));

      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
      xnn_pack_f32_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(), channel_tile(), channel_tile(), channel_tile(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_tile(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc += indirection[x * step() + k][c + input_offset()] * kernel[c * kernel_tile() + k];
            }
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

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

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
        input_offset() * sizeof(float), zero.data(),
        &params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(output[x * output_stride() + c], output_min)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(output[x * output_stride() + c], output_max)
            << "x = " << x << ", channel = " << c;
          EXPECT_NEAR(
              output_ref[x * channels() + c],
              output[x * output_stride() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

  void Test(xnn_f32_dwconv_multipass_ukernel_fn dwconv) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
    std::vector<const float*> indirection((width() - 1) * step() + tile_size);
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float, AlignedAllocator<float, 64>> buffer(XNN_MULTIPASS_EXTRA_BYTES / sizeof(float) + channels());
    std::vector<float> kernel(channels() * kernel_size());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights(
      xnn_dwconv_multipass_weights_size(
        tile_size, channels(), channel_tile(), channel_subtile(), channel_round(),
        /*bias_element_size=*/sizeof(float), /*log2_filter_element_size=*/2, /*extra_weights_byte=*/0) /
      sizeof(float));
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(zero.begin(), zero.end(), 0.0f);
      std::fill(output_ref.begin(), output_ref.end(), nanf(""));
      std::fill(output.begin(), output.end(), nanf(""));

      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
      xnn_pack_f32_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(), 1,
        channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_size()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_size(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc += indirection[x * step() + k][c + input_offset()] * kernel[c * kernel_size() + k];
            }
          }
          output_ref[x * channels() + c] = acc;
        }
      }

      // input_stride is step() - first and middle pass
      size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()), middle_pass_tile());
      const int input_advanced = first_pass_tile() + num_middle_pass * middle_pass_tile();
      int input_stride_elements = step() - input_advanced;
      // Call optimized micro-kernel.
      dwconv(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        input_stride_elements * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        input_offset() * sizeof(float), zero.data(),
        kernel_size(),
        buffer.data(),
        nullptr);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_NEAR(
              output_ref[x * channels() + c],
              output[x * output_stride() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c << " channels = " << channels()
            << " kernel_size = " << kernel_size()
            << " first_pass_tile = " << first_pass_tile()
            << " middle_pass_tile = " << middle_pass_tile()
            << " last_pass_tile = " << last_pass_tile();
        }
      }
    }
  }

  void Test(xnn_f32_dwconv_minmax_multipass_ukernel_fn dwconv_minmax, xnn_init_f32_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
    std::vector<const float*> indirection((width() - 1) * step() + tile_size);
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + indirection.size() * channels());
    std::vector<float, AlignedAllocator<float, 64>> buffer(XNN_MULTIPASS_EXTRA_BYTES / sizeof(float) + channels());
    std::vector<float> kernel(channels() * kernel_size());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights(
      xnn_dwconv_multipass_weights_size(
        tile_size, channels(), channel_tile(), channel_subtile(), channel_round(),
        /*bias_element_size=*/sizeof(float), /*log2_filter_element_size=*/2, /*extra_weights_byte=*/0) /
      sizeof(float));
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((width() - 1) * output_stride() + channels());
    std::vector<float> output_ref(width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(zero.begin(), zero.end(), 0.0f);
      std::fill(output_ref.begin(), output_ref.end(), nanf(""));
      std::fill(output.begin(), output.end(), nanf(""));

      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
      xnn_pack_f32_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(), 1,
        channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);
      if (zero_index() != SIZE_MAX) {
        for (size_t i = 0; i < indirection.size(); i += kernel_size()) {
          indirection[i + zero_index()] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = bias[c];
          for (size_t k = 0; k < kernel_size(); k++) {
            if (indirection[x * step() + k] != zero.data()) {
              acc += indirection[x * step() + k][c + input_offset()] * kernel[c * kernel_size() + k];
            }
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

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      // Clamp reference results.
      for (float& output_val : output_ref) {
        output_val = std::max(std::min(output_val, output_max), output_min);
      }

      // input_stride is step() - first and middle pass
      size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()), middle_pass_tile());
      const int input_advanced = first_pass_tile() + num_middle_pass * middle_pass_tile();
      int input_stride_elements = step() - input_advanced;
      // Call optimized micro-kernel.
      dwconv_minmax(
        channels(), width(),
        indirection.data(), packed_weights.data(), output.data(),
        input_stride_elements * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        input_offset() * sizeof(float), zero.data(),
        kernel_size(),
        buffer.data(),
        &params);

      // Verify results.
      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(output[x * output_stride() + c], output_min)
            << "x = " << x << ", channel = " << c;
          EXPECT_LE(output[x * output_stride() + c], output_max)
            << "x = " << x << ", channel = " << c;
          EXPECT_NEAR(
              output_ref[x * channels() + c],
              output[x * output_stride() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c << " channels = " << channels()
            << " kernel_size = " << kernel_size()
            << " first_pass_tile = " << first_pass_tile()
            << " middle_pass_tile = " << middle_pass_tile()
            << " last_pass_tile = " << last_pass_tile();
        }
      }
    }
  }

 private:
  uint32_t channels_{1};
  uint32_t channel_tile_{1};
  uint32_t channel_subtile_{1};
  uint32_t channel_round_{1};
  uint32_t kernel_tile_{1};
  uint32_t kernel_size_{1};
  uint32_t width_{1};
  uint32_t step_{1};
  uint32_t output_stride_{0};
  uint8_t input_zero_point_{127};
  uint8_t kernel_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t input_offset_{0};
  size_t zero_index_{SIZE_MAX};
  size_t iterations_{3};
  size_t first_pass_tile_{0};
  size_t middle_pass_tile_{0};
  size_t last_pass_tile_{0};
};
