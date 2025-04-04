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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack.h"
#include "test/next_prime.h"
#include "test/replicable_random_device.h"

class VMulCAddCMicrokernelTester {
 public:
  VMulCAddCMicrokernelTester& channel_tile(size_t channel_tile) {
    this->channel_tile_ = channel_tile;
    return *this;
  }

  size_t channel_tile() const { return this->channel_tile_; }

  VMulCAddCMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const { return this->channels_; }

  size_t packed_channels() const {
    return channels() % channel_tile() == 0
               ? channels()
               : (channels() / channel_tile() + 1) * channel_tile();
  }

  VMulCAddCMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const { return this->rows_; }

  VMulCAddCMicrokernelTester& input_stride(size_t input_stride) {
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    return this->input_stride_ == 0 ? channels() : this->input_stride_;
  }

  VMulCAddCMicrokernelTester& output_stride(size_t output_stride) {
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    return this->output_stride_ == 0 ? channels() : this->output_stride_;
  }

  VMulCAddCMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const { return this->inplace_; }

  VMulCAddCMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const { return this->qmin_; }

  VMulCAddCMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const { return this->qmax_; }

  VMulCAddCMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_f16_vmulcaddc_ukernel_fn vmulcaddc,
            xnn_init_f16_minmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    if (inplace()) {
      ASSERT_EQ(input_stride(), output_stride());
    }

    xnnpack::Buffer<xnn_float16> x((rows() - 1) * input_stride() + channels() +
                                   XNN_EXTRA_BYTES / sizeof(xnn_float16));
    xnnpack::Buffer<xnn_float16> scale(channels());
    xnnpack::Buffer<xnn_float16> bias(channels());
    xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_channels() * 2);
    xnnpack::Buffer<xnn_float16> y(
        (rows() - 1) * output_stride() + channels() +
        (inplace() ? XNN_EXTRA_BYTES / sizeof(xnn_float16) : 0));
    xnnpack::Buffer<float> y_ref(rows() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      }
      const xnn_float16* x_data = inplace() ? y.data() : x.data();

      xnn_pack_f16_vmulcaddc_w(channels(), channel_tile(),
                               reinterpret_cast<const uint16_t*>(scale.data()),
                               reinterpret_cast<const uint16_t*>(bias.data()),
                               reinterpret_cast<uint16_t*>(packed_w.data()),
                               nullptr);

      // Compute reference results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          y_ref[i * channels() + j] =
              x_data[i * input_stride() + j] * scale[j] + bias[j];
        }
      }
      const float accumulated_min =
          *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max =
          *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = xnn_float16(
          accumulated_max - accumulated_range / 255.0f * float(255 - qmax()));
      const float y_min = xnn_float16(
          accumulated_min + accumulated_range / 255.0f * float(qmin()));

      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params, static_cast<xnn_float16>(y_min),
                  static_cast<xnn_float16>(y_max));

      // Call optimized micro-kernel.
      vmulcaddc(rows(), channels() * sizeof(xnn_float16), x_data,
                input_stride() * sizeof(xnn_float16), packed_w.data(), y.data(),
                output_stride() * sizeof(xnn_float16), &params);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          ASSERT_NEAR(
              y[i * output_stride() + j], y_ref[i * channels() + j],
              std::max(1.0e-4f, std::abs(y_ref[i * channels() + j]) * 1.0e-2f))
              << "at pixel " << i << " / " << rows() << ", channel = " << j
              << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_f32_vmulcaddc_ukernel_fn vmulcaddc,
            xnn_init_f32_minmax_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    if (inplace()) {
      ASSERT_EQ(input_stride(), output_stride());
    }

    xnnpack::Buffer<float> x((rows() - 1) * input_stride() + channels() +
                             XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float> scale(channels());
    xnnpack::Buffer<float> bias(channels());
    xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_w(
        packed_channels() * 2);
    xnnpack::Buffer<float> y((rows() - 1) * output_stride() + channels() +
                             (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    xnnpack::Buffer<float> y_ref(rows() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      }
      const float* x_data = inplace() ? y.data() : x.data();

      xnn_pack_f32_vmulcaddc_w(channels(), channel_tile(), scale.data(),
                               bias.data(), packed_w.data(), nullptr);

      // Compute reference results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          y_ref[i * channels() + j] =
              x_data[i * input_stride() + j] * scale[j] + bias[j];
        }
      }
      const float accumulated_min =
          *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max =
          *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max =
          accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      const float y_min =
          accumulated_min + accumulated_range / 255.0f * float(qmin());
      for (float& y_value : y_ref) {
        y_value = std::max<float>(std::min<float>(y_value, y_max), y_min);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, y_min, y_max);

      // Call optimized micro-kernel.
      vmulcaddc(rows(), channels() * sizeof(float), x_data,
                input_stride() * sizeof(float), packed_w.data(), y.data(),
                output_stride() * sizeof(float), &params);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          ASSERT_NEAR(y[i * output_stride() + j], y_ref[i * channels() + j],
                      std::abs(y_ref[i * channels() + j]) * 1.0e-6f)
              << "at pixel " << i << " / " << rows() << ", channel = " << j
              << " / " << channels();
        }
      }
    }
  }

 private:
  size_t channel_tile_{1};
  size_t channels_{1};
  size_t rows_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};

#define XNN_TEST_VMULCADDC_ROW_DIV(ukernel, arch_flags, row_tile_,       \
                                   channel_tile_, datatype, params_type, \
                                   init_params)                          \
  TEST(ukernel, ROW_div) {                                               \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                \
    for (size_t rows_ = row_tile_ * 2; rows_ <= row_tile_ * 4;           \
         rows_ += row_tile_) {                                           \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;         \
           channels_ += std::max(1, channel_tile_ - 1)) {                \
        VMulCAddCMicrokernelTester()                                     \
            .channel_tile(channel_tile_)                                 \
            .channels(channels_)                                         \
            .rows(rows_)                                                 \
            .Test(ukernel, init_params);                                 \
      }                                                                  \
    }                                                                    \
  }

#define XNN_TEST_VMULCADDC_ROW_LT(ukernel, arch_flags, row_tile_,       \
                                  channel_tile_, datatype, params_type, \
                                  init_params)                          \
  TEST(ukernel, ROW_lt) {                                               \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                               \
    for (size_t rows_ = 1; rows_ < row_tile_; rows_++) {                \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;        \
           channels_ += std::max(1, channel_tile_ - 1)) {               \
        VMulCAddCMicrokernelTester()                                    \
            .channel_tile(channel_tile_)                                \
            .channels(channels_)                                        \
            .rows(rows_)                                                \
            .Test(ukernel, init_params);                                \
      }                                                                 \
    }                                                                   \
  }
#define XNN_TEST_VMULCADDC_ROW_GT(ukernel, arch_flags, row_tile_,        \
                                  channel_tile_, datatype, params_type,  \
                                  init_params)                           \
  TEST(ukernel, ROW_gt) {                                                \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                \
    for (size_t rows_ = row_tile_ + 1; rows_ < row_tile_ * 2; rows_++) { \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;         \
           channels_ += std::max(1, channel_tile_ - 1)) {                \
        VMulCAddCMicrokernelTester()                                     \
            .channel_tile(channel_tile_)                                 \
            .channels(channels_)                                         \
            .rows(rows_)                                                 \
            .Test(ukernel, init_params);                                 \
      }                                                                  \
    }                                                                    \
  }
#define XNN_TEST_VMULCADDC_CHANNEL_GT(ukernel, arch_flags, row_tile_,       \
                                      channel_tile_, datatype, params_type, \
                                      init_params)                          \
  TEST(ukernel, channels_gt_) {                                             \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    for (size_t channels_ = channel_tile_ + 1;                              \
         channels_ < (channel_tile_ == 1 ? 10 : channel_tile_ * 2);         \
         channels_++) {                                                     \
      VMulCAddCMicrokernelTester()                                          \
          .channel_tile(channel_tile_)                                      \
          .channels(channels_)                                              \
          .rows(row_tile_)                                                  \
          .Test(ukernel, init_params);                                      \
    }                                                                       \
  }
#define XNN_TEST_VMULCADDC_CHANNEL_EQ(ukernel, arch_flags, row_tile_,       \
                                      channel_tile_, datatype, params_type, \
                                      init_params)                          \
  TEST(ukernel, channels_eq_) {                                             \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    VMulCAddCMicrokernelTester()                                            \
        .channel_tile(channel_tile_)                                        \
        .channels(channel_tile_)                                            \
        .rows(row_tile_)                                                    \
        .Test(ukernel, init_params);                                        \
  }
#define XNN_TEST_VMULCADDC_CHANNEL_DIV(ukernel, arch_flags, row_tile_,         \
                                       channel_tile_, datatype, params_type,   \
                                       init_params)                            \
  TEST(ukernel, channels_div_) {                                               \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    for (size_t channels_ = channel_tile_ * 2; channels_ < channel_tile_ * 10; \
         channels_ += channel_tile_) {                                         \
      VMulCAddCMicrokernelTester()                                             \
          .channel_tile(channel_tile_)                                         \
          .channels(channels_)                                                 \
          .rows(row_tile_)                                                     \
          .Test(ukernel, init_params);                                         \
    }                                                                          \
  }
#define XNN_TEST_VMULCADDC_CHANNEL_LT(ukernel, arch_flags, row_tile_,       \
                                      channel_tile_, datatype, params_type, \
                                      init_params)                          \
  TEST(ukernel, channels_lt_) {                                             \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    for (size_t channels_ = 1; channels_ < channel_tile_; channels_++) {    \
      VMulCAddCMicrokernelTester()                                          \
          .channel_tile(channel_tile_)                                      \
          .channels(channels_)                                              \
          .rows(row_tile_)                                                  \
          .Test(ukernel, init_params);                                      \
    }                                                                       \
  }
#define XNN_TEST_VMULCADDC_INPUT_STRIDE(ukernel, arch_flags, row_tile_,       \
                                        channel_tile_, datatype, params_type, \
                                        init_params)                          \
  TEST(ukernel, input_stride) {                                               \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                     \
    for (size_t rows_ = 1; rows_ <= row_tile_ * 3;                            \
         rows_ += std::max(1, row_tile_ - 1)) {                               \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;              \
           channels_ += std::max(1, channel_tile_ - 1)) {                     \
        VMulCAddCMicrokernelTester()                                          \
            .channel_tile(channel_tile_)                                      \
            .channels(channels_)                                              \
            .rows(rows_)                                                      \
            .input_stride(xnnpack::NextPrime(channel_tile_ * 5 + 1))          \
            .Test(ukernel, init_params);                                      \
      }                                                                       \
    }                                                                         \
  }
#define XNN_TEST_VMULCADDC_OUTPUT_STRIDE(ukernel, arch_flags, row_tile_,       \
                                         channel_tile_, datatype, params_type, \
                                         init_params)                          \
  TEST(ukernel, output_stride) {                                               \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    for (size_t rows_ = 1; rows_ <= row_tile_ * 3;                             \
         rows_ += std::max(1, row_tile_ - 1)) {                                \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;               \
           channels_ += std::max(1, channel_tile_ - 1)) {                      \
        VMulCAddCMicrokernelTester()                                           \
            .channel_tile(channel_tile_)                                       \
            .channels(channels_)                                               \
            .rows(rows_)                                                       \
            .output_stride(xnnpack::NextPrime(channel_tile_ * 5 + 1))          \
            .Test(ukernel, init_params);                                       \
      }                                                                        \
    }                                                                          \
  }
#define XNN_TEST_VMULCADDC_INPLACE(ukernel, arch_flags, row_tile_,       \
                                   channel_tile_, datatype, params_type, \
                                   init_params)                          \
  TEST(ukernel, inplace) {                                               \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                \
    for (size_t rows_ = 1; rows_ <= row_tile_ * 3;                       \
         rows_ += std::max(1, row_tile_ - 1)) {                          \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;         \
           channels_ += std::max(1, channel_tile_ - 1)) {                \
        VMulCAddCMicrokernelTester()                                     \
            .channel_tile(channel_tile_)                                 \
            .channels(channels_)                                         \
            .rows(rows_)                                                 \
            .inplace(true)                                               \
            .Test(ukernel, init_params);                                 \
      }                                                                  \
    }                                                                    \
  }
#define XNN_TEST_VMULCADDC_QMIN(ukernel, arch_flags, row_tile_, channel_tile_, \
                                datatype, params_type, init_params)            \
  TEST(ukernel, qmin) {                                                        \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    for (size_t rows_ = 1; rows_ <= row_tile_ * 3;                             \
         rows_ += std::max(1, row_tile_ - 1)) {                                \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;               \
           channels_ += std::max(1, channel_tile_ - 1)) {                      \
        VMulCAddCMicrokernelTester()                                           \
            .channel_tile(channel_tile_)                                       \
            .channels(channels_)                                               \
            .rows(rows_)                                                       \
            .qmin(128)                                                         \
            .Test(ukernel, init_params);                                       \
      }                                                                        \
    }                                                                          \
  }
#define XNN_TEST_VMULCADDC_QMAX(ukernel, arch_flags, row_tile_, channel_tile_, \
                                datatype, params_type, init_params)            \
  TEST(ukernel, qmax) {                                                        \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    for (size_t rows_ = 1; rows_ <= row_tile_ * 3;                             \
         rows_ += std::max(1, row_tile_ - 1)) {                                \
      for (size_t channels_ = 1; channels_ <= channel_tile_ * 5;               \
           channels_ += std::max(1, channel_tile_ - 1)) {                      \
        VMulCAddCMicrokernelTester()                                           \
            .channel_tile(channel_tile_)                                       \
            .channels(channels_)                                               \
            .rows(rows_)                                                       \
            .qmax(128)                                                         \
            .Test(ukernel, init_params);                                       \
      }                                                                        \
    }                                                                          \
  }
