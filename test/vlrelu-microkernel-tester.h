// Copyright 2022 Google LLC
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
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class VLReLUMicrokernelTester {
 public:
  VLReLUMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  VLReLUMicrokernelTester& positive_scale(float positive_scale) {
    assert(positive_scale > 0.0f);
    assert(std::isnormal(positive_scale));
    this->positive_scale_ = positive_scale;
    return *this;
  }

  float positive_scale() const {
    return this->positive_scale_;
  }

  VLReLUMicrokernelTester& negative_scale(float negative_scale) {
    assert(std::isnormal(negative_scale));
    this->negative_scale_ = negative_scale;
    return *this;
  }

  float negative_scale() const {
    return this->negative_scale_;
  }

  VLReLUMicrokernelTester& input_zero_point(int16_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  int16_t input_zero_point() const {
    return this->input_zero_point_;
  }

  VLReLUMicrokernelTester& output_zero_point(int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  int16_t output_zero_point() const {
    return this->output_zero_point_;
  }

  VLReLUMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_qs8_vlrelu_ukernel_fn vlrelu, xnn_init_qs8_lrelu_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(batch_size());
    std::vector<int8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      struct xnn_qs8_lrelu_params params;
      init_params(&params, positive_scale(), negative_scale(), input_zero_point(), output_zero_point());

      // Call optimized micro-kernel.
      vlrelu(batch_size() * sizeof(int8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t positive_multiplier = (int32_t) lrintf(-256.0f * positive_scale());
      const int32_t negative_multiplier = (int32_t) lrintf(-256.0f * negative_scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = (input_zero_point() - input[i]) * 128;
        const int32_t multiplier = input_value <= 0 ? positive_multiplier : negative_multiplier;
        int32_t output_value = math_asr_s32(input_value * multiplier + INT32_C(0x4000), 15) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<int8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<int8_t>::min());
        output_ref[i] = static_cast<int8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

  void Test(xnn_qu8_vlrelu_ukernel_fn vlrelu, xnn_init_qu8_lrelu_params_fn init_params) const {
    ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(batch_size());
    std::vector<uint8_t> output_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      struct xnn_qu8_lrelu_params params;
      init_params(&params, positive_scale(), negative_scale(), input_zero_point(), output_zero_point());

      // Call optimized micro-kernel.
      vlrelu(batch_size() * sizeof(uint8_t), input.data(), output.data(), &params);

      // Compute reference results
      const int32_t positive_multiplier = (int32_t) lrintf(-256.0f * positive_scale());
      const int32_t negative_multiplier = (int32_t) lrintf(-256.0f * negative_scale());
      for (size_t i = 0; i < batch_size(); i++) {
        const int32_t input_value = (input_zero_point() - input[i]) * 128;
        const int32_t multiplier = input_value <= 0 ? positive_multiplier : negative_multiplier;
        int32_t output_value = math_asr_s32(input_value * multiplier + INT32_C(0x4000), 15) + output_zero_point();
        output_value = std::min<int32_t>(output_value, std::numeric_limits<uint8_t>::max());
        output_value = std::max<int32_t>(output_value, std::numeric_limits<uint8_t>::min());
        output_ref[i] = static_cast<uint8_t>(output_value);
      }

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(output[i]), int32_t(output_ref[i]))
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "] = " << int32_t(input[i]);
      }
    }
  }

 private:
  float positive_scale_ = 1.75f;
  float negative_scale_ = 0.75f;
  int16_t input_zero_point_ = 1;
  int16_t output_zero_point_ = 5;
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};

// TODO(b/361780131): This could probably be rewritten as some kind of GTest
// instantiate thing instead of macros.
#define XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, \
                                ...)                                       \
  TEST(ukernel, batch_eq) {                                                \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                  \
    const size_t batch_scale = get_batch_scale<datatype>();                \
    VLReLUMicrokernelTester()                                              \
        .batch_size(batch_tile* batch_scale)                               \
        .Test(__VA_ARGS__);                                                \
  }

#define XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, \
                                 ...)                                       \
  TEST(ukernel, batch_div) {                                                \
    if (batch_tile == 1) return;                                            \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    const size_t batch_step = batch_tile * batch_scale;                     \
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step;  \
         batch_size += batch_step) {                                        \
      VLReLUMicrokernelTester().batch_size(batch_size).Test(__VA_ARGS__);   \
    }                                                                       \
  }

#define XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, \
                                ...)                                       \
  TEST(ukernel, batch_lt) {                                                \
    if (batch_tile == 1) return;                                           \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                  \
    const size_t batch_scale = get_batch_scale<datatype>();                \
    const size_t batch_end = batch_tile * batch_scale;                     \
    for (size_t batch_size = 1; batch_size < batch_end; batch_size++) {    \
      VLReLUMicrokernelTester().batch_size(batch_size).Test(__VA_ARGS__);  \
    }                                                                      \
  }

#define XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, \
                                ...)                                       \
  TEST(ukernel, batch_gt) {                                                \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                  \
    const size_t batch_scale = get_batch_scale<datatype>();                \
    const size_t batch_step = batch_tile * batch_scale;                    \
    const size_t batch_end = batch_tile == 1 ? 10 : 2 * batch_step;        \
    for (size_t batch_size = batch_step + 1; batch_size < batch_end;       \
         batch_size++) {                                                   \
      VLReLUMicrokernelTester().batch_size(batch_size).Test(__VA_ARGS__);  \
    }                                                                      \
  }

#define XNN_TEST_UNARY_INPLACE(ukernel, arch_flags, batch_tile, datatype, ...)

#define XNN_TEST_UNARY_QMIN(ukernel, arch_flags, batch_tile, datatype, ...) \
  TEST(ukernel, qmin) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    const size_t batch_end = batch_tile * batch_scale;                      \
    const size_t batch_step =                                               \
        batch_scale == 1 ? std::max(1, batch_tile - 1) : batch_end - 1;     \
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {    \
      for (size_t batch_size = 1; batch_size <= 5 * batch_end;              \
           batch_size += batch_step) {                                      \
        VLReLUMicrokernelTester()                                           \
            .batch_size(batch_size)                                         \
            .qmin(qmin)                                                     \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }

#define XNN_TEST_UNARY_QMAX(ukernel, arch_flags, batch_tile, datatype, ...) \
  TEST(ukernel, qmax) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    const size_t batch_end = batch_tile * batch_scale;                      \
    const size_t batch_step =                                               \
        batch_scale == 1 ? std::max(1, batch_tile - 1) : batch_end - 1;     \
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {    \
      for (size_t batch_size = 1; batch_size <= 5 * batch_end;              \
           batch_size += batch_step) {                                      \
        VLReLUMicrokernelTester()                                           \
            .batch_size(batch_size)                                         \
            .qmax(qmax)                                                     \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }
