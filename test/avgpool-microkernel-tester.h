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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/requantization.h"
#include "next_prime.h"
#include "replicable_random_device.h"

class AvgPoolMicrokernelTester {
 public:
  AvgPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  size_t output_pixels() const {
    return this->output_pixels_;
  }

  AvgPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  size_t step() const {
    return this->step_;
  }

  AvgPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const {
    return this->input_offset_;
  }

  AvgPoolMicrokernelTester& zero_index_mod2(size_t zero_index_mod2) {
    this->zero_index_mod2_ = zero_index_mod2;
    return *this;
  }

  size_t zero_index_mod2() const {
    return this->zero_index_mod2_;
  }

  AvgPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  size_t packed_pooling_elements() const {
    if (pooling_elements() <= primary_pooling_tile()) {
      return primary_pooling_tile();
    } else {
      return (pooling_elements() - primary_pooling_tile()) % incremental_pooling_tile() == 0 ? pooling_elements() : ((pooling_elements() - primary_pooling_tile()) / incremental_pooling_tile() + 1) * incremental_pooling_tile() + primary_pooling_tile();
    }
  }

  AvgPoolMicrokernelTester& pooling_tile(size_t primary_tile, size_t incremental_tile = 0) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = incremental_tile;
    return *this;
  }

  AvgPoolMicrokernelTester& primary_pooling_tile(size_t primary_pooling_tile) {
    assert(primary_pooling_tile != 0);
    this->primary_pooling_tile_ = primary_pooling_tile;
    return *this;
  }

  size_t primary_pooling_tile() const {
    return this->primary_pooling_tile_;
  }

  AvgPoolMicrokernelTester& incremental_pooling_tile(size_t incremental_pooling_tile) {
    assert(incremental_pooling_tile != 0);
    this->incremental_pooling_tile_ = incremental_pooling_tile;
    return *this;
  }

  size_t incremental_pooling_tile() const {
    return this->incremental_pooling_tile_;
  }

  AvgPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  AvgPoolMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  AvgPoolMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  AvgPoolMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  AvgPoolMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const {
    return this->output_scale_;
  }

  AvgPoolMicrokernelTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  AvgPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  AvgPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  AvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_avgpool_minmax_unipass_ukernel_fn avgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      input_offset() + indirect_input.size() * channels());
    std::vector<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return xnn_float16_from_float(f32dist(rng)); });
      std::fill(input.begin(), input.begin() + input_offset(), UINT16_C(0x7E00) /* NaN */);
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(xnn_float16), input.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const xnn_float16* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += xnn_float16_to_float(row[c + input_offset()]);
            }
          }
          output_ref[x * channels() + c] = acc / float(pooling_elements());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min_as_float = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      float output_max_as_float = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;
      const xnn_float16 output_min_as_half = xnn_float16_from_float(output_min_as_float);
      const xnn_float16 output_max_as_half = xnn_float16_from_float(output_max_as_float);
      output_min_as_float = xnn_float16_to_float(output_min_as_half);
      output_max_as_float = xnn_float16_to_float(output_max_as_half);

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max_as_float), output_min_as_float);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params, xnn_float16_from_float(1.0f / float(pooling_elements())), output_min_as_half, output_max_as_half);

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), zero.data(),
        output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(xnn_float16_to_float(output[x * output_stride() + c]), output_min_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(xnn_float16_to_float(output[x * output_stride() + c]), output_max_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              xnn_float16_to_float(output[x * output_stride() + c]),
              output_ref[x * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 3.0e-3f))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f16_avgpool_minmax_multipass_ukernel_fn avgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      input_offset() + indirect_input.size() * channels());
    std::vector<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<xnn_float16, AlignedAllocator<xnn_float16, 64>> buffer(XNN_EXTRA_BYTES / sizeof(xnn_float16) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return xnn_float16_from_float(f32dist(rng)); });
      std::fill(input.begin(), input.begin() + input_offset(), UINT16_C(0x7E00) /* NaN */);
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(xnn_float16), input.end(), UINT16_C(0x7E00) /* NaN */);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const xnn_float16* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += xnn_float16_to_float(row[c + input_offset()]);
            }
          }
          output_ref[x * channels() + c] = acc / float(pooling_elements());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min_as_float = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      float output_max_as_float = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;
      const xnn_float16 output_min_as_half = xnn_float16_from_float(output_min_as_float);
      const xnn_float16 output_max_as_half = xnn_float16_from_float(output_max_as_float);
      output_min_as_float = xnn_float16_to_float(output_min_as_half);
      output_max_as_float = xnn_float16_to_float(output_max_as_half);

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max_as_float), output_min_as_float);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params, xnn_float16_from_float(1.0f / float(pooling_elements())), output_min_as_half, output_max_as_half);

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), zero.data(),
        buffer.data(), output.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(xnn_float16_to_float(output[x * output_stride() + c]), output_min_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(xnn_float16_to_float(output[x * output_stride() + c]), output_max_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              xnn_float16_to_float(output[x * output_stride() + c]),
              output_ref[x * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 3.0e-3f))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_avgpool_minmax_unipass_ukernel_fn avgpool_minmax, xnn_init_f32_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc / float(pooling_elements());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_scaleminmax_params params;
      init_params(&params, 1.0f / float(pooling_elements()), output_min, output_max);

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), zero.data(),
        output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-6f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_avgpool_minmax_multipass_ukernel_fn avgpool_minmax, xnn_init_f32_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<float, AlignedAllocator<float, 64>> buffer(XNN_EXTRA_BYTES / sizeof(float) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc / float(pooling_elements());
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_scaleminmax_params params;
      init_params(&params, 1.0f / float(pooling_elements()), output_min, output_max);

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), zero.data(),
        buffer.data(), output.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-6f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(
      xnn_qu8_avgpool_minmax_unipass_ukernel_fn avgpool_minmax,
      xnn_init_qu8_avgpool_minmax_params_fn init_params,
      xnn_qu8_requantize_fn requantize) const
  {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<const uint8_t*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      input_offset() + indirect_input.size() * channels());
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_ref(output_pixels() * channels());
    std::vector<float> output_real(output_pixels() * channels());
    std::vector<int32_t> accumulator(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      std::fill(input.begin(), input.begin() + input_offset(), UINT8_C(0xA5));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(uint8_t), input.end(), UINT8_C(0xA5));
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Prepare parameters.
      xnn_qu8_avgpool_minmax_params params;
      init_params(
        &params,
        -int32_t(input_zero_point()) * int32_t(pooling_elements()),
        input_scale() / (output_scale() * float(pooling_elements())),
        output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const uint8_t* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += int32_t(row[c + input_offset()]);
            }
            acc -= int32_t(input_zero_point());
          }
          accumulator[x * channels() + c] = acc;
          output_ref[x * channels() + c] = requantize(
            acc, input_scale() / (output_scale() * float(pooling_elements())), output_zero_point(), qmin(), qmax());
          const float scaled_acc =
            float(acc) * input_scale() / (output_scale() * float(pooling_elements())) + float(output_zero_point());
          output_real[x * channels() + c] = std::min(std::max(scaled_acc, float(qmin())), float(qmax()));
        }
      }

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(uint8_t), zero.data(),
        output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint8_t),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(float(int32_t(output[x * output_stride() + c])), output_real[x * channels() + c], 0.5f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
          EXPECT_EQ(uint32_t(output_ref[x * channels() + c]), uint32_t(output[x * output_stride() + c]))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
        }
      }
    }
  }

  void Test(
      xnn_qu8_avgpool_minmax_multipass_ukernel_fn avgpool_minmax,
      xnn_init_qu8_avgpool_minmax_params_fn init_params,
      xnn_qu8_requantize_fn requantize) const
  {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<const uint8_t*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      input_offset() + indirect_input.size() * channels());
    std::vector<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_ref(output_pixels() * channels());
    std::vector<float> output_real(output_pixels() * channels());
    std::vector<int32_t> accumulator(output_pixels() * channels());
    std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(XNN_EXTRA_BYTES / sizeof(uint8_t) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      std::fill(input.begin(), input.begin() + input_offset(), UINT8_C(0xA5));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(uint8_t), input.end(), UINT8_C(0xA5));
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Prepare parameters.
      xnn_qu8_avgpool_minmax_params params;
      init_params(
        &params,
        -int32_t(input_zero_point()) * int32_t(pooling_elements()),
        input_scale() / (output_scale() * float(pooling_elements())),
        output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const uint8_t* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += int32_t(row[c + input_offset()]);
            }
            acc -= int32_t(input_zero_point());
          }
          accumulator[x * channels() + c] = acc;
          output_ref[x * channels() + c] = requantize(
            acc, input_scale() / (output_scale() * float(pooling_elements())), output_zero_point(), qmin(), qmax());
          const float scaled_acc =
            float(acc) * input_scale() / (output_scale() * float(pooling_elements())) + float(output_zero_point());
          output_real[x * channels() + c] = std::min(std::max(scaled_acc, float(qmin())), float(qmax()));
        }
      }

      // Call optimized micro-kernel.
      avgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(uint8_t), zero.data(),
        buffer.data(), output.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint8_t),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(float(int32_t(output[x * output_stride() + c])), output_real[x * channels() + c], 0.5f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
          EXPECT_EQ(uint32_t(output_ref[x * channels() + c]), uint32_t(output[x * output_stride() + c]))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
        }
      }
    }
  }

  void Test(xnn_f16_pavgpool_minmax_unipass_ukernel_fn pavgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> m32dist(0.1f, 0.5f);

    std::vector<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      input_offset() + indirect_input.size() * channels());
    std::vector<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16> multiplier(output_pixels());
    std::vector<xnn_float16> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return xnn_float16_from_float(f32dist(rng)); });
      std::fill(input.begin(), input.begin() + input_offset(), UINT16_C(0x7E00) /* NaN */);
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(xnn_float16), input.end(), UINT16_C(0x7E00) /* NaN */);
      std::generate(multiplier.begin(), multiplier.end(), [&]() { return xnn_float16_from_float(m32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const xnn_float16* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += xnn_float16_to_float(row[c + input_offset()]);
            }
          }
          output_ref[x * channels() + c] = acc * xnn_float16_to_float(multiplier[x]);
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min_as_float = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      float output_max_as_float = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;
      const xnn_float16 output_min_as_half = xnn_float16_from_float(output_min_as_float);
      const xnn_float16 output_max_as_half = xnn_float16_from_float(output_max_as_float);
      output_min_as_float = xnn_float16_to_float(output_min_as_half);
      output_max_as_float = xnn_float16_to_float(output_max_as_half);

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max_as_float), output_min_as_float);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params, 0, output_min_as_half, output_max_as_half);

      // Call optimized micro-kernel.
      pavgpool_minmax(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), zero.data(),
        multiplier.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(xnn_float16_to_float(output[x * output_stride() + c]), output_min_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(xnn_float16_to_float(output[x * output_stride() + c]), output_max_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              xnn_float16_to_float(output[x * output_stride() + c]),
              output_ref[x * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 3.0e-3f))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f16_pavgpool_minmax_multipass_ukernel_fn pavgpool_minmax, xnn_init_f16_scaleminmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> m32dist(0.1f, 0.5f);

    std::vector<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      input_offset() + indirect_input.size() * channels());
    std::vector<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16> multiplier(output_pixels());
    std::vector<xnn_float16> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<xnn_float16, AlignedAllocator<xnn_float16, 64>> buffer(XNN_EXTRA_BYTES / sizeof(xnn_float16) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return xnn_float16_from_float(f32dist(rng)); });
      std::fill(input.begin(), input.begin() + input_offset(), UINT16_C(0x7E00) /* NaN */);
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(xnn_float16), input.end(), UINT16_C(0x7E00) /* NaN */);
      std::generate(multiplier.begin(), multiplier.end(), [&]() { return xnn_float16_from_float(m32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const xnn_float16* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += xnn_float16_to_float(row[c + input_offset()]);
            }
          }
          output_ref[x * channels() + c] = acc * xnn_float16_to_float(multiplier[x]);
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min_as_float = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      float output_max_as_float = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;
      const xnn_float16 output_min_as_half = xnn_float16_from_float(output_min_as_float);
      const xnn_float16 output_max_as_half = xnn_float16_from_float(output_max_as_float);
      output_min_as_float = xnn_float16_to_float(output_min_as_half);
      output_max_as_float = xnn_float16_to_float(output_max_as_half);

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max_as_float), output_min_as_float);
      }

      // Prepare parameters.
      xnn_f16_scaleminmax_params params;
      init_params(&params, 0, output_min_as_half, output_max_as_half);

      // Call optimized micro-kernel.
      pavgpool_minmax(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), zero.data(),
        multiplier.data(), buffer.data(), output.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(xnn_float16_to_float(output[x * output_stride() + c]), output_min_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(xnn_float16_to_float(output[x * output_stride() + c]), output_max_as_float)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              xnn_float16_to_float(output[x * output_stride() + c]),
              output_ref[x * channels() + c],
              std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 3.0e-3f))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_pavgpool_minmax_unipass_ukernel_fn pavgpool_minmax, xnn_init_f32_minmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> m32dist(0.1f, 0.5f);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> multiplier(output_pixels());
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::generate(multiplier.begin(), multiplier.end(), [&]() { return m32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc * multiplier[x];
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      // Call optimized micro-kernel.
      pavgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), zero.data(),
        multiplier.data(), output.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-6f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_pavgpool_minmax_multipass_ukernel_fn pavgpool_minmax, xnn_init_f32_minmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;
    std::uniform_real_distribution<float> m32dist(0.1f, 0.5f);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> multiplier(output_pixels());
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<float, AlignedAllocator<float, 64>> buffer(XNN_EXTRA_BYTES / sizeof(float) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::generate(multiplier.begin(), multiplier.end(), [&]() { return m32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index_mod2() != SIZE_MAX) {
        for (size_t i = zero_index_mod2(); i < indirect_input.size(); i += 2) {
          indirect_input[i] = zero.data();
        }
      }

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float acc = 0.0f;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += row[c + input_offset()];
            }
          }
          output_ref[x * channels() + c] = acc * multiplier[x];
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float output_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      // Call optimized micro-kernel.
      pavgpool_minmax(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), zero.data(),
        multiplier.data(), buffer.data(), output.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_NEAR(
              output[x * output_stride() + c],
              output_ref[x * channels() + c],
              std::abs(output_ref[x * channels() + c]) * 1.0e-6f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  struct TestF16AvgPoolFns {
    xnn_f16_avgpool_minmax_unipass_ukernel_fn uni;
    xnn_f16_avgpool_minmax_multipass_ukernel_fn multi;
    xnn_init_f16_scaleminmax_params_fn init;
  };

  void Test(const TestF16AvgPoolFns& fns) const {
    assert((fns.uni != nullptr) != (fns.multi != nullptr));
    if (fns.uni) Test(fns.uni, fns.init);
    else Test(fns.multi, fns.init);
  }

  struct TestF32AvgPoolFns {
    xnn_f32_avgpool_minmax_unipass_ukernel_fn uni;
    xnn_f32_avgpool_minmax_multipass_ukernel_fn multi;
    xnn_init_f32_scaleminmax_params_fn init;
  };

  void Test(const TestF32AvgPoolFns& fns) const {
    assert((fns.uni != nullptr) != (fns.multi != nullptr));
    if (fns.uni) Test(fns.uni, fns.init);
    else Test(fns.multi, fns.init);
  }

  struct TestQU8AvgPoolFns {
    xnn_qu8_avgpool_minmax_unipass_ukernel_fn uni;
    xnn_qu8_avgpool_minmax_multipass_ukernel_fn multi;
    xnn_init_qu8_avgpool_minmax_params_fn init;
    xnn_qu8_requantize_fn requantize;
  };

  void Test(const TestQU8AvgPoolFns& fns) const {
    assert((fns.uni != nullptr) != (fns.multi != nullptr));
    if (fns.uni) Test(fns.uni, fns.init, fns.requantize);
    else Test(fns.multi, fns.init, fns.requantize);
  }

  struct TestF16PAvgPoolFns {
    xnn_f16_pavgpool_minmax_unipass_ukernel_fn uni;
    xnn_f16_pavgpool_minmax_multipass_ukernel_fn multi;
    xnn_init_f16_scaleminmax_params_fn init;
  };

  void Test(const TestF16PAvgPoolFns& fns) const {
    assert((fns.uni != nullptr) != (fns.multi != nullptr));
    if (fns.uni) Test(fns.uni, fns.init);
    else Test(fns.multi, fns.init);
  }

  struct TestF32PAvgPoolFns {
    xnn_f32_pavgpool_minmax_unipass_ukernel_fn uni;
    xnn_f32_pavgpool_minmax_multipass_ukernel_fn multi;
    xnn_init_f32_minmax_params_fn init;
  };

  void Test(const TestF32PAvgPoolFns& fns) const {
    assert((fns.uni != nullptr) != (fns.multi != nullptr));
    if (fns.uni) Test(fns.uni, fns.init);
    else Test(fns.multi, fns.init);
  }

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t zero_index_mod2_{SIZE_MAX};
  size_t step_{1};
  size_t primary_pooling_tile_{1};
  size_t incremental_pooling_tile_{1};
  size_t output_stride_{0};
  float input_scale_{1.25f};
  float output_scale_{0.75f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{3};
};

template<typename Fns>
struct XnnAvgPoolTestParam {
  const char *name;
  Fns fns;
  uint64_t arch_flags;
  size_t channel_tile, channel_scaled_tile, primary_tile, incremental_tile;
};

template<typename Fns>
class XnnAvgPoolTest : public testing::TestWithParam<XnnAvgPoolTestParam<Fns>> {
protected:
  const XnnAvgPoolTestParam<Fns>& TestParam() const {
    return this->GetParam();
  }

  void channels_eq_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile)
      .channels(channel_tile)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile)
      .channels(channel_tile)
      .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                    xnnpack::NextPrime(TestParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_unipass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .zero_index_mod2(zero_index_mod2)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile)
      .channels(channel_tile)
      .qmin(128)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile)
      .channels(channel_tile)
      .qmax(128)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile)
        .channels(channel_tile)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_unipass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile)
          .channels(channel_tile)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile + 1) :
                        channel_tile + 1)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .input_offset(channel_tile*8)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .qmin(128)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .qmax(128)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 8))
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_unipass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .input_offset(channel_tile*8)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_lt_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile) :
                      channel_tile)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile) :
                        channel_tile)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .qmin(128)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile)
        .channels(channels)
        .qmax(128)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile) :
                        channel_tile)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_unipass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                          xnnpack::NextPrime(TestParam().channel_tile) :
                          channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(TestParam().primary_tile)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(TestParam().primary_tile)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .qmin(128)
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .qmin(128)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .qmax(128)
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile)
          .channels(channels)
          .qmax(128)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_unipass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
              .zero_index_mod2(zero_index_mod2)
              .Test(TestParam().fns);
          }
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile)
              .channels(channels)
              .input_offset(channel_tile*2)
              .zero_index_mod2(zero_index_mod2)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void few_output_pixels_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void few_output_pixels_with_input_offset_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void few_output_pixels_with_zero_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .channels(channels)
                .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
                .zero_index_mod2(zero_index_mod2)
                .Test(TestParam().fns);
            }
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .channels(channels)
                .input_offset(channel_tile*5+1)
                .zero_index_mod2(zero_index_mod2)
                .Test(TestParam().fns);
            }
          }
        }
      }
    }
  }

  void few_output_pixels_with_qmin_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmin(128)
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmin(128)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void few_output_pixels_with_qmax_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmax(128)
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmax(128)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void few_output_pixels_with_output_stride_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .output_stride(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void few_output_pixels_with_step_0() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile != 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile - 1, TestParam().primary_tile}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            for (size_t step = 2; step <= pooling_elements; step++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .step(step)
                .channels(channels)
                .output_stride(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
                .Test(TestParam().fns);
            }
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= 3 * channel_tile; channels += channel_tile-1) {
            for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .step(step)
                .channels(channels)
                .output_stride(channel_tile*5+1)
                .Test(TestParam().fns);
            }
          }
        }
      }
    }
  }

  void channels_eq_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                    xnnpack::NextPrime(TestParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_twopass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .zero_index_mod2(zero_index_mod2)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .qmin(128)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    AvgPoolMicrokernelTester()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .qmax(128)
      .Test(TestParam().fns);
  }

  void channels_eq_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_twopass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channel_tile)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile + 1) :
                        channel_tile + 1)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*5)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*5)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_twopass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*8)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_lt_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile) :
                      channel_tile)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile) :
                        channel_tile)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(TestParam().fns);
    }
  }

  void channels_lt_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile) :
                        channel_tile)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_twopass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                          xnnpack::NextPrime(TestParam().channel_tile) :
                          channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(TestParam().fns);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_gt_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_gt_channel_tile_twopass_subtile_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
              .zero_index_mod2(zero_index_mod2)
              .Test(TestParam().fns);
          }
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*2)
              .zero_index_mod2(zero_index_mod2)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }

  void channels_eq_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_multipass_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channel_tile)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile + 1) :
                        channel_tile + 1)
          .zero_index_mod2(zero_index_mod2)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_eq_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .qmin(128)
        .Test(TestParam().fns);
    }
  }

  void channels_eq_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .qmax(128)
        .Test(TestParam().fns);
    }
  }

  void channels_div_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_multipass_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*8)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }

  void channels_div_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_div_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().fns);
      }
    }
  }

  void channels_lt_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile)
          .Test(TestParam().fns);
      }
    }
  }


  void channels_lt_channel_tile_multipass_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(TestParam().fns);
        }
      }
    }
  }



  void channels_lt_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(TestParam().fns);
      }
    }
  }


  void channels_lt_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    if (TestParam().channel_tile <= 1 || TestParam().channel_scaled_tile == TestParam().channel_tile) {
      GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(TestParam().fns);
      }
    }
  }


  void channels_gt_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().fns);
        }
      }
    }
  }


  void channels_gt_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(TestParam().fns);
        }
      }
    }
  }


  void channels_gt_channel_tile_multipass_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
              .zero_index_mod2(zero_index_mod2)
              .Test(TestParam().fns);
          }
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*2)
              .zero_index_mod2(zero_index_mod2)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }



  void channels_gt_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmin(128)
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmin(128)
            .Test(TestParam().fns);
        }
      }
    }
  }


  void channels_gt_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile + 1; pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmax(128)
            .Test(TestParam().fns);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmax(128)
            .Test(TestParam().fns);
        }
      }
    }
  }


  void few_output_pixels() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }


  void few_output_pixels_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }


  void few_output_pixels_with_zero() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .channels(channels)
                .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
                .zero_index_mod2(zero_index_mod2)
                .Test(TestParam().fns);
            }
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .channels(channels)
                .input_offset(channel_tile*5+1)
                .zero_index_mod2(zero_index_mod2)
                .Test(TestParam().fns);
            }
          }
        }
      }
    }
  }



  void few_output_pixels_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmin(128)
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmin(128)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }


  void few_output_pixels_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmax(128)
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmax(128)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }


  void few_output_pixels_with_output_stride() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .output_stride(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
              .Test(TestParam().fns);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(TestParam().fns);
          }
        }
      }
    }
  }


  void few_output_pixels_with_step() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().incremental_tile == 0) {
      GTEST_SKIP();
    }
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{TestParam().primary_tile + 1, TestParam().primary_tile + TestParam().incremental_tile - 1, TestParam().primary_tile + TestParam().incremental_tile + 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            for (size_t step = 2; step <= pooling_elements; step++) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .step(step)
                .channels(channels)
                .output_stride(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
                .Test(TestParam().fns);
            }
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= 3 * channel_tile; channels += channel_tile-1) {
            for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
              AvgPoolMicrokernelTester()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .step(step)
                .channels(channels)
                .output_stride(channel_tile*5+1)
                .Test(TestParam().fns);
            }
          }
        }
      }
    }
  }
};
