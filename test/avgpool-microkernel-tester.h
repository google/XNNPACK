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


class AvgPoolMicrokernelTester {
 public:
  enum class Variant {
    Native,
    Scalar,
  };

  inline AvgPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  inline size_t output_pixels() const {
    return this->output_pixels_;
  }

  inline AvgPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  inline size_t step() const {
    return this->step_;
  }

  inline AvgPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  inline size_t input_offset() const {
    return this->input_offset_;
  }

  inline AvgPoolMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  inline size_t zero_index() const {
    return this->zero_index_;
  }

  inline AvgPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  inline size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  inline size_t packed_pooling_elements() const {
    if (pooling_elements() <= primary_pooling_tile()) {
      return primary_pooling_tile();
    } else {
      return (pooling_elements() - primary_pooling_tile()) % incremental_pooling_tile() == 0 ? pooling_elements() : ((pooling_elements() - primary_pooling_tile()) / incremental_pooling_tile() + 1) * incremental_pooling_tile() + primary_pooling_tile();
    }
  }

  inline AvgPoolMicrokernelTester& pooling_tile(size_t primary_tile, size_t incremental_tile = 0) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = incremental_tile;
    return *this;
  }

  inline AvgPoolMicrokernelTester& primary_pooling_tile(size_t primary_pooling_tile) {
    assert(primary_pooling_tile != 0);
    this->primary_pooling_tile_ = primary_pooling_tile;
    return *this;
  }

  inline size_t primary_pooling_tile() const {
    return this->primary_pooling_tile_;
  }

  inline AvgPoolMicrokernelTester& incremental_pooling_tile(size_t incremental_pooling_tile) {
    assert(incremental_pooling_tile != 0);
    this->incremental_pooling_tile_ = incremental_pooling_tile;
    return *this;
  }

  inline size_t incremental_pooling_tile() const {
    return this->incremental_pooling_tile_;
  }

  inline AvgPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AvgPoolMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  inline AvgPoolMicrokernelTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline AvgPoolMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline AvgPoolMicrokernelTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline AvgPoolMicrokernelTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline AvgPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline AvgPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_q8_avgpool_minmax_unipass_ukernel_function avgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

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
        std::generate(input.begin(), input.end(), std::ref(u8rng));
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      std::fill(input.begin(), input.begin() + input_offset(), 0xA5);
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(uint8_t), input.end(), 0xA5);
      std::fill(output.begin(), output.end(), 0xA5);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index() != SIZE_MAX) {
        indirect_input[zero_index()] = zero.data();
      }

      // Prepare quantization parameters.
      xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(pooling_elements()),
            input_scale() / (output_scale() * float(pooling_elements())),
            output_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(pooling_elements()),
            input_scale() / (output_scale() * float(pooling_elements())),
            output_zero_point(), qmin(), qmax());
          break;
      }
      const xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(input_zero_point()) * int32_t(pooling_elements()),
          input_scale() / (output_scale() * float(pooling_elements())),
          output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = scalar_quantization_params.scalar.bias;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const uint8_t* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += int32_t(row[c + input_offset()]);
            }
          }
          accumulator[x * channels() + c] = acc;
          output_ref[x * channels() + c] = xnn_avgpool_quantize(acc, scalar_quantization_params);
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
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(float(int32_t(output[x * output_stride() + c])), output_real[x * channels() + c], 0.5f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
          ASSERT_EQ(uint32_t(output_ref[x * channels() + c]), uint32_t(output[x * output_stride() + c]))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
        }
      }
    }
  }

  void Test(xnn_q8_avgpool_minmax_multipass_ukernel_function avgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

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
        std::generate(input.begin(), input.end(), std::ref(u8rng));
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      std::fill(input.begin(), input.begin() + input_offset(), 0xA5);
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(uint8_t), input.end(), 0xA5);
      std::fill(output.begin(), output.end(), 0xA5);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index() != SIZE_MAX) {
        indirect_input[zero_index()] = zero.data();
      }

      // Prepare quantization parameters.
      xnn_q8_avgpool_params quantization_params = { };
      switch (variant) {
        case Variant::Native:
          quantization_params = xnn_init_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(pooling_elements()),
            input_scale() / (output_scale() * float(pooling_elements())),
            output_zero_point(), qmin(), qmax());
          break;
        case Variant::Scalar:
          quantization_params = xnn_init_scalar_q8_avgpool_params(
            -int32_t(input_zero_point()) * int32_t(pooling_elements()),
            input_scale() / (output_scale() * float(pooling_elements())),
            output_zero_point(), qmin(), qmax());
          break;
      }
      const xnn_q8_avgpool_params scalar_quantization_params =
        xnn_init_scalar_q8_avgpool_params(
          -int32_t(input_zero_point()) * int32_t(pooling_elements()),
          input_scale() / (output_scale() * float(pooling_elements())),
          output_zero_point(), qmin(), qmax());

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = scalar_quantization_params.scalar.bias;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const uint8_t* row = indirect_input[x * step() + p];
            if (row != zero.data()) {
              acc += int32_t(row[c + input_offset()]);
            }
          }
          accumulator[x * channels() + c] = acc;
          output_ref[x * channels() + c] = xnn_avgpool_quantize(acc, scalar_quantization_params);
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
        &quantization_params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(uint32_t(output[x * output_stride() + c]), uint32_t(qmin()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(uint32_t(output[x * output_stride() + c]), uint32_t(qmax()))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(float(int32_t(output[x * output_stride() + c])), output_real[x * channels() + c], 0.5f)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
          ASSERT_EQ(uint32_t(output_ref[x * channels() + c]), uint32_t(output[x * output_stride() + c]))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset() << ", accumulator = " << accumulator[x * channels() + c];
        }
      }
    }
  }

  void Test(xnn_f32_avgpool_minmax_unipass_ukernel_function avgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index() != SIZE_MAX) {
        indirect_input[zero_index()] = zero.data();
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

      // Prepare output parameters.
      xnn_f32_scaleminmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_scaleminmax_params(
            1.0f / float(pooling_elements()), output_min, output_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_scaleminmax_params(
            1.0f / float(pooling_elements()), output_min, output_max);
          break;
      }

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
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
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

  void Test(xnn_f32_avgpool_minmax_multipass_ukernel_function avgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<float, AlignedAllocator<float, 64>> buffer(XNN_EXTRA_BYTES / sizeof(float) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index() != SIZE_MAX) {
        indirect_input[zero_index()] = zero.data();
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

      // Prepare output parameters.
      xnn_f32_scaleminmax_params params = { };
      switch (variant) {
        case Variant::Native:
          params = xnn_init_f32_scaleminmax_params(
            1.0f / float(pooling_elements()), output_min, output_max);
          break;
        case Variant::Scalar:
          params = xnn_init_scalar_f32_scaleminmax_params(
            1.0f / float(pooling_elements()), output_min, output_max);
          break;
      }

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
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
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

  void Test(xnn_f32_pavgpool_minmax_unipass_ukernel_function pavgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f32mrng = std::bind(std::uniform_real_distribution<float>(0.1f, 0.5f), rng);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> multiplier(output_pixels());
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32irng));
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::generate(multiplier.begin(), multiplier.end(), std::ref(f32mrng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index() != SIZE_MAX) {
        indirect_input[zero_index()] = zero.data();
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
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
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

  void Test(xnn_f32_pavgpool_minmax_multipass_ukernel_function pavgpool_minmax, Variant variant = Variant::Native) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32irng = std::bind(std::uniform_real_distribution<float>(), rng);
    auto f32mrng = std::bind(std::uniform_real_distribution<float>(0.1f, 0.5f), rng);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      input_offset() + indirect_input.size() * channels());
    std::vector<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> multiplier(output_pixels());
    std::vector<float> output((output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    std::vector<float, AlignedAllocator<float, 64>> buffer(XNN_EXTRA_BYTES / sizeof(float) + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32irng));
      std::fill(input.begin(), input.begin() + input_offset(), std::nanf(""));
      std::fill(input.end() - XNN_EXTRA_BYTES / sizeof(float), input.end(), std::nanf(""));
      std::generate(multiplier.begin(), multiplier.end(), std::ref(f32mrng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);
      if (zero_index() != SIZE_MAX) {
        indirect_input[zero_index()] = zero.data();
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
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_NEAR(
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

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t zero_index_{SIZE_MAX};
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
