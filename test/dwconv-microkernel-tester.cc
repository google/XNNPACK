// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "dwconv-microkernel-tester.h"

#include <stdint.h>

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
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"
#include "xnnpack/requantization.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

TEST_P(DWConvTest, Test) {
  const DWConvTestParams& params = GetParam();
  DWConvMicrokernelTester tester = params.tester;

  // Make sure that we can execute this test.
  if (params.isa_check) {
    params.isa_check();
    if (IsSkipped()) {
      return;
    }
  }

  // Loop over the kernel size and channels, if required.
  for (size_t ks = params.loop_kernel_size_.from;
       ks <= params.loop_kernel_size_.to; ks += params.loop_kernel_size_.step) {
    if (params.loop_kernel_size_.is_set) {
      tester.kernel_size(ks);
    }
    for (size_t c = params.loop_channels_.from; c <= params.loop_channels_.to;
         c += params.loop_channels_.step) {
      if (params.loop_channels_.is_set) {
        tester.channels(c);
      }
      for (size_t s = params.loop_step_.from; s <= params.loop_step_.to;
           s += params.loop_step_.step) {
        if (params.loop_step_.is_set) {
          tester.step(s);
        }
        for (size_t zi = params.loop_zi_.from; zi <= params.loop_zi_.to;
             zi += params.loop_zi_.step) {
          if (params.loop_zi_.is_set) {
            tester.zero_index(zi);
          }

          // Call the test function.
          params.test_func(tester);
        }
      }
    }
  }
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DWConvTest);

void DWConvMicrokernelTester::Test(
    xnn_qu8_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_qu8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

  xnnpack::Buffer<const uint8_t*> indirection((width() - 1) * step() +
                                          kernel_tile());
  xnnpack::Buffer<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
                             indirection.size() * channels());
  xnnpack::Buffer<uint8_t> kernel(channels() * kernel_tile());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + sizeof(int32_t) / sizeof(uint8_t)) * packed_channels());
  xnnpack::Buffer<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t), input_zero_point());
  xnnpack::Buffer<uint8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<uint8_t> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
    } while (input.size() > 1 &&
             *std::max_element(input.cbegin(), input.cend()) ==
                 *std::min_element(input.cbegin(), input.cend()));
    do {
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return u8dist(rng); });
    } while (kernel.size() > 1 &&
             *std::max_element(kernel.cbegin(), kernel.cend()) ==
                 *std::min_element(kernel.cbegin(), kernel.cend()));
    std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(),
              kernel_zero_point());
    const xnn_qu8_packing_params packing_params = {input_zero_point(),
                                                   kernel_zero_point()};
    xnn_pack_qu8_dwconv_ghw_w(kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
                              channel_tile(), channel_tile(), channel_tile(),
                              kernel.data(), bias.data(), /*scale=*/nullptr,
                              packed_weights.data(),
                              /*per_tile_extra_bytes=*/0,
                              /*per_subtile_extra_bytes=*/0, &packing_params);
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
            acc += (static_cast<int32_t>(
                        indirection[x * step() + k][c + input_offset()]) -
                    static_cast<int32_t>(input_zero_point())) *
                   (static_cast<int32_t>(kernel[c * kernel_tile() + k]) -
                    static_cast<int32_t>(kernel_zero_point()));
          }
        }
        accumulators[x * channels() + c] = acc;
      }
    }

    // Compute renormalization parameters.
    const int32_t accumulated_min =
        *std::min_element(accumulators.cbegin(), accumulators.cend());
    const int32_t accumulated_max =
        *std::max_element(accumulators.cbegin(), accumulators.cend());
    const uint32_t accumulated_range = static_cast<uint32_t>(accumulated_max) -
                                       static_cast<uint32_t>(accumulated_min);
    const double output_scale =
        accumulated_range >= 256
            ? static_cast<double>(accumulated_range) / 255.0
            : 1.00001;
    const uint8_t output_zero_point = static_cast<uint8_t>(std::max(
        std::min(
            lrint(127.5 -
                  0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                      output_scale),
            static_cast<long>(std::numeric_limits<uint8_t>::max())),
        static_cast<long>(std::numeric_limits<uint8_t>::min())));

    // Prepare parameters.
    const float requantization_scale = 1.0f / static_cast<float>(output_scale);
    union xnn_qu8_conv_minmax_params quantization_params;
    init_params(&quantization_params, kernel_zero_point(), requantization_scale,
                output_zero_point, qmin(), qmax());

    // Renormalize reference results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        output_ref[x * channels() + c] =
            requantize(accumulators[x * channels() + c], requantization_scale,
                       output_zero_point, qmin(), qmax());
      }
    }

    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(), step() * sizeof(void*),
                  (output_stride() - channels()) * sizeof(uint8_t),
                  input_offset() * sizeof(uint8_t), zero.data(),
                  &quantization_params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(static_cast<uint32_t>(output[x * output_stride() + c]),
                  static_cast<uint32_t>(qmin()))
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(static_cast<uint32_t>(output[x * output_stride() + c]),
                  static_cast<uint32_t>(qmax()))
            << "x = " << x << ", channel = " << c;
        EXPECT_EQ(static_cast<uint32_t>(output[x * output_stride() + c]),
                  static_cast<uint32_t>(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c
            << ", accumulator = " << accumulators[x * channels() + c];
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qu8_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_qu8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
  xnnpack::Buffer<const uint8_t*> indirection((width() - 1) * step() + tile_size);
  xnnpack::Buffer<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
                             indirection.size() * channels());
  xnnpack::Buffer<int32_t, XNN_ALLOCATION_ALIGNMENT> buffer(
      XNN_MULTIPASS_EXTRA_BYTES / sizeof(uint8_t) + channels());
  xnnpack::Buffer<uint8_t> kernel(channels() * kernel_size());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      xnn_dwconv_multipass_weights_size(tile_size, channels(), channel_tile(),
                                        channel_subtile(), channel_round(),
                                        /*bias_element_size=*/4,
                                        /*log2_filter_element_size=*/0,
                                        /*extra_weights_byte=*/0));
  xnnpack::Buffer<uint8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(uint8_t), input_zero_point());
  xnnpack::Buffer<uint8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<uint8_t> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
    } while (input.size() > 1 &&
             *std::max_element(input.cbegin(), input.cend()) ==
                 *std::min_element(input.cbegin(), input.cend()));
    do {
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return u8dist(rng); });
    } while (kernel.size() > 1 &&
             *std::max_element(kernel.cbegin(), kernel.cend()) ==
                 *std::min_element(kernel.cbegin(), kernel.cend()));
    std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(),
              kernel_zero_point());
    const xnn_qu8_packing_params packing_params = {input_zero_point(),
                                                   kernel_zero_point()};
    xnn_pack_qu8_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(),
        1, channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
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
            acc += (static_cast<int32_t>(
                        indirection[x * step() + k][c + input_offset()]) -
                    static_cast<int32_t>(input_zero_point())) *
                   (static_cast<int32_t>(kernel[c * kernel_size() + k]) -
                    static_cast<int32_t>(kernel_zero_point()));
          }
        }
        accumulators[x * channels() + c] = acc;
      }
    }

    // Compute renormalization parameters.
    const int32_t accumulated_min =
        *std::min_element(accumulators.cbegin(), accumulators.cend());
    const int32_t accumulated_max =
        *std::max_element(accumulators.cbegin(), accumulators.cend());
    const uint32_t accumulated_range = static_cast<uint32_t>(accumulated_max) -
                                       static_cast<uint32_t>(accumulated_min);
    const double output_scale =
        accumulated_range >= 256
            ? static_cast<double>(accumulated_range) / 255.0
            : 1.00001;
    const uint8_t output_zero_point = static_cast<uint8_t>(std::max(
        std::min(
            lrint(127.5 -
                  0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                      output_scale),
            static_cast<long>(std::numeric_limits<uint8_t>::max())),
        static_cast<long>(std::numeric_limits<uint8_t>::min())));

    // Prepare parameters.
    const float requantization_scale = 1.0f / static_cast<float>(output_scale);
    union xnn_qu8_conv_minmax_params quantization_params;
    init_params(&quantization_params, kernel_zero_point(), requantization_scale,
                output_zero_point, qmin(), qmax());

    // Renormalize reference results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        output_ref[x * channels() + c] =
            requantize(accumulators[x * channels() + c], requantization_scale,
                       output_zero_point, qmin(), qmax());
      }
    }

    // input_stride is step() - first and middle pass
    size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()),
                        middle_pass_tile());
    const int input_advanced =
        first_pass_tile() + num_middle_pass * middle_pass_tile();
    int input_stride_elements = step() - input_advanced;
    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(),
                  input_stride_elements * sizeof(void*),
                  (output_stride() - channels()) * sizeof(uint8_t),
                  input_offset() * sizeof(uint8_t), zero.data(), kernel_size(),
                  buffer.data(), &quantization_params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(static_cast<uint32_t>(output[x * output_stride() + c]),
                  static_cast<uint32_t>(qmin()))
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(static_cast<uint32_t>(output[x * output_stride() + c]),
                  static_cast<uint32_t>(qmax()))
            << "x = " << x << ", channel = " << c;
        EXPECT_EQ(static_cast<uint32_t>(output[x * output_stride() + c]),
                  static_cast<uint32_t>(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c
            << ", accumulator = " << accumulators[x * channels() + c];
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qs8_qc8w_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

  xnnpack::Buffer<const int8_t*> indirection((width() - 1) * step() +
                                         kernel_tile());
  xnnpack::Buffer<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
                            indirection.size() * channels());
  xnnpack::Buffer<int8_t> kernel(channels() * kernel_tile());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t)) *
      packed_channels());
  xnnpack::Buffer<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t), static_cast<int8_t>(input_zero_point() - 0x80));
  xnnpack::Buffer<int8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<float> scale(channels());
  xnnpack::Buffer<int8_t> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
    } while (input.size() > 1 &&
             *std::max_element(input.cbegin(), input.cend()) ==
                 *std::min_element(input.cbegin(), input.cend()));
    do {
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return w8dist(rng); });
    } while (kernel.size() > 1 &&
             *std::max_element(kernel.cbegin(), kernel.cend()) ==
                 *std::min_element(kernel.cbegin(), kernel.cend()));
    std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        static_cast<int8_t>(input_zero_point() - 0x80)};
    xnn_pack_qs8_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(), channel_tile(),
        channel_tile(), channel_tile(), kernel.data(), bias.data(),
        /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/channel_tile() * sizeof(float),
        /*per_subtile_extra_bytes=*/channel_tile() * sizeof(float),
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
            acc += (static_cast<int32_t>(
                        indirection[x * step() + k][c + input_offset()]) -
                    static_cast<int32_t>(input_zero_point() - 0x80)) *
                   static_cast<int32_t>(kernel[c * kernel_tile() + k]);
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
        accumulated_min =
            std::min(accumulated_min, accumulators[x * channels() + c]);
        accumulated_max =
            std::max(accumulated_max, accumulators[x * channels() + c]);
      }
      const uint32_t accumulated_range =
          static_cast<uint32_t>(accumulated_max - accumulated_min);
      const float output_scale =
          accumulated_range >= 256
              ? static_cast<double>(accumulated_range) / 255.0
              : 1.00001;
      scale[c] = 1.0f / output_scale;
    }
    xnn_init_qs8_qc8w_scale_fp32_params(
        channels(), channel_tile(), channel_tile(),
        channel_tile() *
            (kernel_tile() * sizeof(int8_t) + sizeof(int32_t) + sizeof(float)),
        channel_tile() *
            (kernel_tile() * sizeof(int8_t) + sizeof(int32_t) + sizeof(float)),
        0, scale.data(),
        (void*)((uintptr_t)packed_weights.data() +
                channel_tile() *
                    (kernel_tile() * sizeof(int8_t) + sizeof(int32_t))));

    // Prepare parameters.
    union xnn_qs8_qc8w_conv_minmax_params minmax_params;
    init_params(&minmax_params, output_zero_point,
                static_cast<int8_t>(qmin() - 0x80),
                static_cast<int8_t>(qmax() - 0x80));

    // Renormalize reference results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        output_ref[x * channels() + c] =
            requantize(accumulators[x * channels() + c], scale[c],
                       output_zero_point, static_cast<int8_t>(qmin() - 0x80),
                       static_cast<int8_t>(qmax() - 0x80));
      }
    }

    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(), step() * sizeof(void*),
                  (output_stride() - channels()) * sizeof(int8_t),
                  input_offset() * sizeof(int8_t), zero.data(), &minmax_params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_EQ(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c
            << ", accumulator = " << accumulators[x * channels() + c];
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qs8_qc8w_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
  xnnpack::Buffer<const int8_t*> indirection((width() - 1) * step() + tile_size);
  xnnpack::Buffer<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
                            indirection.size() * channels());
  xnnpack::Buffer<int32_t, XNN_ALLOCATION_ALIGNMENT> buffer(
      XNN_MULTIPASS_EXTRA_BYTES / sizeof(int8_t) + channels());
  xnnpack::Buffer<int8_t> kernel(channels() * kernel_size());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      xnn_dwconv_multipass_weights_size(tile_size, channels(), channel_tile(),
                                        channel_subtile(), channel_round(),
                                        /*bias_element_size=*/4,
                                        /*log2_filter_element_size=*/0,
                                        /*extra_weights_byte=*/sizeof(float)));
  xnnpack::Buffer<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t),
              static_cast<int8_t>(input_zero_point() - 0x80));
  xnnpack::Buffer<int8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<float> scale(channels());
  xnnpack::Buffer<int8_t> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
    } while (input.size() > 1 &&
             *std::max_element(input.cbegin(), input.cend()) ==
                 *std::min_element(input.cbegin(), input.cend()));
    do {
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return w8dist(rng); });
    } while (kernel.size() > 1 &&
             *std::max_element(kernel.cbegin(), kernel.cend()) ==
                 *std::min_element(kernel.cbegin(), kernel.cend()));
    std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        static_cast<int8_t>(input_zero_point() - 0x80)};
    xnn_pack_qs8_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(),
        1, channels(), channel_tile(), channel_subtile(), channel_round(),
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
            acc += (static_cast<int32_t>(
                        indirection[x * step() + k][c + input_offset()]) -
                    static_cast<int32_t>(input_zero_point() - 0x80)) *
                   static_cast<int32_t>(kernel[c * kernel_size() + k]);
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
        accumulated_min =
            std::min(accumulated_min, accumulators[x * channels() + c]);
        accumulated_max =
            std::max(accumulated_max, accumulators[x * channels() + c]);
      }
      const uint32_t accumulated_range =
          static_cast<uint32_t>(accumulated_max - accumulated_min);
      const float output_scale =
          accumulated_range >= 256
              ? static_cast<double>(accumulated_range) / 255.0
              : 1.00001;
      scale[c] = 1.0f / output_scale;
    }

    size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()),
                        middle_pass_tile());
    const size_t rounded_c = round_up_po2(channels(), channel_subtile());
    const size_t packed_weights_offset_to_last_tile =
        first_pass_tile() * rounded_c * sizeof(int8_t) +
        rounded_c * sizeof(int32_t) +
        num_middle_pass * middle_pass_tile() * rounded_c * sizeof(int8_t) +
        last_pass_tile() * channel_tile();

    xnn_init_qs8_qc8w_scale_fp32_params(
        channels(), channel_tile(), channel_subtile(),
        channel_tile() * (last_pass_tile() * sizeof(int8_t) + sizeof(int32_t)),
        channel_subtile() *
            (last_pass_tile() * sizeof(int8_t) + sizeof(int32_t)),
        (channel_tile() - channel_subtile()) * last_pass_tile() *
            sizeof(int8_t),
        scale.data(),
        (void*)((uintptr_t)packed_weights.data() +
                packed_weights_offset_to_last_tile));

    // Prepare parameters.
    union xnn_qs8_qc8w_conv_minmax_params minmax_params;
    init_params(&minmax_params, output_zero_point,
                static_cast<int8_t>(qmin() - 0x80),
                static_cast<int8_t>(qmax() - 0x80));

    // Renormalize reference results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        output_ref[x * channels() + c] =
            requantize(accumulators[x * channels() + c], scale[c],
                       output_zero_point, static_cast<int8_t>(qmin() - 0x80),
                       static_cast<int8_t>(qmax() - 0x80));
      }
    }

    // input_stride is step() - first and middle pass
    const int input_advanced =
        first_pass_tile() + num_middle_pass * middle_pass_tile();
    int input_stride_elements = step() - input_advanced;
    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(),
                  input_stride_elements * sizeof(void*),
                  (output_stride() - channels()) * sizeof(int8_t),
                  input_offset() * sizeof(int8_t), zero.data(), kernel_size(),
                  buffer.data(), &minmax_params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_EQ(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c
            << ", accumulator = " << accumulators[x * channels() + c]
            << ", channels = " << channels();
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qs8_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

  xnnpack::Buffer<const int8_t*> indirection((width() - 1) * step() +
                                         kernel_tile());
  xnnpack::Buffer<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
                            indirection.size() * channels());
  xnnpack::Buffer<int8_t> kernel(channels() * kernel_tile());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + sizeof(int32_t) / sizeof(int8_t)) * packed_channels());
  xnnpack::Buffer<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t),
              static_cast<int8_t>(input_zero_point() - 0x80));
  xnnpack::Buffer<int8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<int8_t> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
    } while (input.size() > 1 &&
             *std::max_element(input.cbegin(), input.cend()) ==
                 *std::min_element(input.cbegin(), input.cend()));
    do {
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return w8dist(rng); });
    } while (kernel.size() > 1 &&
             *std::max_element(kernel.cbegin(), kernel.cend()) ==
                 *std::min_element(kernel.cbegin(), kernel.cend()));
    std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        static_cast<int8_t>(input_zero_point() - 0x80)};
    xnn_pack_qs8_dwconv_ghw_w(kernel_tile(), 0, 0, kernel_tile(), 1, channels(),
                              channel_tile(), channel_tile(), channel_tile(),
                              kernel.data(), bias.data(), /*scale=*/nullptr,
                              packed_weights.data(),
                              /*per_tile_extra_bytes=*/0,
                              /*per_subtile_extra_bytes=*/0, &packing_params);
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
            acc += (static_cast<int32_t>(
                        indirection[x * step() + k][c + input_offset()]) -
                    static_cast<int32_t>(input_zero_point() - 0x80)) *
                   static_cast<int32_t>(kernel[c * kernel_tile() + k]);
          }
        }
        accumulators[x * channels() + c] = acc;
      }
    }

    // Compute renormalization parameters.
    const int32_t accumulated_min =
        *std::min_element(accumulators.cbegin(), accumulators.cend());
    const int32_t accumulated_max =
        *std::max_element(accumulators.cbegin(), accumulators.cend());
    const uint32_t accumulated_range = static_cast<uint32_t>(accumulated_max) -
                                       static_cast<uint32_t>(accumulated_min);
    const double output_scale =
        accumulated_range >= 256
            ? static_cast<double>(accumulated_range) / 255.0
            : 1.00001;
    const int8_t output_zero_point = static_cast<int8_t>(std::max(
        std::min(
            lrint(-0.5 -
                  0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                      output_scale),
            static_cast<long>(std::numeric_limits<int8_t>::max())),
        static_cast<long>(std::numeric_limits<int8_t>::min())));

    // Prepare parameters.
    const float requantization_scale = 1.0f / static_cast<float>(output_scale);
    union xnn_qs8_conv_minmax_params quantization_params;
    init_params(&quantization_params, requantization_scale, output_zero_point,
                static_cast<int8_t>(qmin() - 0x80),
                static_cast<int8_t>(qmax() - 0x80));

    // Renormalize reference results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        output_ref[x * channels() + c] =
            requantize(accumulators[x * channels() + c], requantization_scale,
                       output_zero_point, static_cast<int8_t>(qmin() - 0x80),
                       static_cast<int8_t>(qmax() - 0x80));
      }
    }

    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(), step() * sizeof(void*),
                  (output_stride() - channels()) * sizeof(int8_t),
                  input_offset() * sizeof(int8_t), zero.data(),
                  &quantization_params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_EQ(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c
            << ", accumulator = " << accumulators[x * channels() + c];
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qs8_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  std::uniform_int_distribution<int32_t> w8dist(
      -std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
  xnnpack::Buffer<const int8_t*> indirection((width() - 1) * step() + tile_size);
  xnnpack::Buffer<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
                            indirection.size() * channels());
  xnnpack::Buffer<int32_t, XNN_ALLOCATION_ALIGNMENT> buffer(
      XNN_MULTIPASS_EXTRA_BYTES / sizeof(int8_t) + channels());
  xnnpack::Buffer<int8_t> kernel(channels() * kernel_size());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      xnn_dwconv_multipass_weights_size(tile_size, channels(), channel_tile(),
                                        channel_subtile(), channel_round(),
                                        /*bias_element_size=*/4,
                                        /*log2_filter_element_size=*/0,
                                        /*extra_weights_byte=*/0));
  xnnpack::Buffer<int8_t> zero(channels() + XNN_EXTRA_BYTES / sizeof(int8_t),
              static_cast<int8_t>(input_zero_point() - 0x80));
  xnnpack::Buffer<int8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<int8_t> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    do {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
    } while (input.size() > 1 &&
             *std::max_element(input.cbegin(), input.cend()) ==
                 *std::min_element(input.cbegin(), input.cend()));
    do {
      std::generate(kernel.begin(), kernel.end(),
                    [&]() { return w8dist(rng); });
    } while (kernel.size() > 1 &&
             *std::max_element(kernel.cbegin(), kernel.cend()) ==
                 *std::min_element(kernel.cbegin(), kernel.cend()));
    std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0);
    const xnn_qs8_packing_params packing_params = {
        static_cast<int8_t>(input_zero_point() - 0x80)};
    xnn_pack_qs8_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(),
        1, channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
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
            acc += (static_cast<int32_t>(
                        indirection[x * step() + k][c + input_offset()]) -
                    static_cast<int32_t>(input_zero_point() - 0x80)) *
                   static_cast<int32_t>(kernel[c * kernel_size() + k]);
          }
        }
        accumulators[x * channels() + c] = acc;
      }
    }

    // Compute renormalization parameters.
    const int32_t accumulated_min =
        *std::min_element(accumulators.cbegin(), accumulators.cend());
    const int32_t accumulated_max =
        *std::max_element(accumulators.cbegin(), accumulators.cend());
    const uint32_t accumulated_range = static_cast<uint32_t>(accumulated_max) -
                                       static_cast<uint32_t>(accumulated_min);
    const double output_scale =
        accumulated_range >= 256
            ? static_cast<double>(accumulated_range) / 255.0
            : 1.00001;
    const int8_t output_zero_point = static_cast<int8_t>(std::max(
        std::min(
            lrint(-0.5 -
                  0.5 * static_cast<double>(accumulated_min + accumulated_max) /
                      output_scale),
            static_cast<long>(std::numeric_limits<int8_t>::max())),
        static_cast<long>(std::numeric_limits<int8_t>::min())));

    // Prepare parameters.
    const float requantization_scale = 1.0f / static_cast<float>(output_scale);
    union xnn_qs8_conv_minmax_params quantization_params;
    init_params(&quantization_params, requantization_scale, output_zero_point,
                static_cast<int8_t>(qmin() - 0x80),
                static_cast<int8_t>(qmax() - 0x80));

    // Renormalize reference results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        output_ref[x * channels() + c] =
            requantize(accumulators[x * channels() + c], requantization_scale,
                       output_zero_point, static_cast<int8_t>(qmin() - 0x80),
                       static_cast<int8_t>(qmax() - 0x80));
      }
    }

    // input_stride is step() - first and middle pass
    size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()),
                        middle_pass_tile());
    const int input_advanced =
        first_pass_tile() + num_middle_pass * middle_pass_tile();
    int input_stride_elements = step() - input_advanced;
    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(),
                  input_stride_elements * sizeof(void*),
                  (output_stride() - channels()) * sizeof(int8_t),
                  input_offset() * sizeof(int8_t), zero.data(), kernel_size(),
                  buffer.data(), &quantization_params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmin()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(qmax()) - 0x80)
            << "x = " << x << ", channel = " << c;
        EXPECT_EQ(static_cast<int32_t>(output[x * output_stride() + c]),
                  static_cast<int32_t>(output_ref[x * channels() + c]))
            << "x = " << x << ", channel = " << c
            << ", accumulator = " << accumulators[x * channels() + c];
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_f16_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<const xnn_float16*> indirection((width() - 1) * step() +
                                           kernel_tile());
  xnnpack::Buffer<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
                              indirection.size() * channels());
  xnnpack::Buffer<xnn_float16> kernel(channels() * kernel_tile());
  xnnpack::Buffer<xnn_float16> bias(channels());
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + 1) * packed_channels());
  xnnpack::Buffer<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16), 0.0f);
  xnnpack::Buffer<xnn_float16> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(),
                  [&]() { return f32dist(rng); });
    std::generate(kernel.begin(), kernel.end(),
                  [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(),
                  [&]() { return f32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0);
    xnn_pack_f16_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(), channel_tile(),
        channel_tile(), channel_tile(), 
        reinterpret_cast<const uint16_t*>(kernel.data()), 
        reinterpret_cast<const uint16_t*>(bias.data()),
        /*scale=*/nullptr, 
        reinterpret_cast<uint16_t*>(packed_weights.data()),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
        /*params=*/nullptr);
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
            acc += indirection[x * step() + k][c + input_offset()] *
                   kernel[c * kernel_tile() + k];
          }
        }
        output_ref[x * channels() + c] = acc;
      }
    }

    // Compute clamping parameters.
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float output_min = xnn_float16(
        accumulated_min +
        accumulated_range / 255.0f * static_cast<float>(qmin()));
    const float output_max = xnn_float16(
        accumulated_max -
        accumulated_range / 255.0f * static_cast<float>(255 - qmax()));

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, output_min,
                output_max);

    // Clamp reference results.
    for (float& output_val : output_ref) {
      output_val = std::max(std::min(output_val, output_max), output_min);
    }

    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(),
                  reinterpret_cast<const xnn_float16**>(indirection.data()),
                  packed_weights.data(), output.data(), step() * sizeof(void*),
                  (output_stride() - channels()) * sizeof(xnn_float16),
                  input_offset() * sizeof(xnn_float16), zero.data(), &params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(output[x * output_stride() + c],
                  output_min)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(output[x * output_stride() + c],
                  output_max)
            << "x = " << x << ", channel = " << c;
        EXPECT_NEAR(output_ref[x * channels() + c],
                    output[x * output_stride() + c],
                    std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) *
                                          1.0e-2f))
            << "x = " << x << ", channel = " << c;
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f16_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_f16_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
  xnnpack::Buffer<const xnn_float16*> indirection((width() - 1) * step() + tile_size);
  xnnpack::Buffer<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
                              indirection.size() * channels());
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> buffer(
      XNN_MULTIPASS_EXTRA_BYTES / sizeof(xnn_float16) + channels());
  xnnpack::Buffer<xnn_float16> kernel(channels() * kernel_size());
  xnnpack::Buffer<xnn_float16> bias(channels());
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      xnn_dwconv_multipass_weights_size(
          tile_size, channels(), channel_tile(), channel_subtile(),
          channel_round(),
          /*bias_element_size=*/sizeof(xnn_float16),
          /*log2_filter_element_size=*/1,
          /*extra_weights_byte=*/0) /
      sizeof(xnn_float16));
  xnnpack::Buffer<xnn_float16> zero(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16), 0.0f);
  xnnpack::Buffer<xnn_float16> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(),
                  [&]() { return f32dist(rng); });
    std::generate(kernel.begin(), kernel.end(),
                  [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(),
                  [&]() { return f32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0);
    xnn_pack_f16_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(),
        1, channels(), channel_tile(), channel_subtile(), channel_round(),
        reinterpret_cast<const uint16_t*>(kernel.data()), 
        reinterpret_cast<const uint16_t*>(bias.data()), /*scale=*/nullptr, 
        reinterpret_cast<uint16_t*>(packed_weights.data()),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
        /*params=*/nullptr);
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
            acc += indirection[x * step() + k][c + input_offset()] *
                   kernel[c * kernel_size() + k];
          }
        }
        output_ref[x * channels() + c] = acc;
      }
    }

    // Compute clamping parameters.
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float output_min = xnn_float16(
        accumulated_min +
        accumulated_range / 255.0f * static_cast<float>(qmin()));
    const float output_max = xnn_float16(
        accumulated_max -
        accumulated_range / 255.0f * static_cast<float>(255 - qmax()));

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, output_min,
                output_max);

    // Clamp reference results.
    for (float& output_val : output_ref) {
      output_val = std::max(std::min(output_val, output_max), output_min);
    }

    // input_stride is step() - first and middle pass
    size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()),
                        middle_pass_tile());
    const int input_advanced =
        first_pass_tile() + num_middle_pass * middle_pass_tile();
    int input_stride_elements = step() - input_advanced;
    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(),
                  reinterpret_cast<const xnn_float16**>(indirection.data()),
                  packed_weights.data(), output.data(),
                  input_stride_elements * sizeof(void*),
                  (output_stride() - channels()) * sizeof(xnn_float16),
                  input_offset() * sizeof(xnn_float16), zero.data(), kernel_size(),
                  buffer.data(), &params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(output[x * output_stride() + c],
                  output_min)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(output[x * output_stride() + c],
                  output_max)
            << "x = " << x << ", channel = " << c;
        EXPECT_NEAR(output_ref[x * channels() + c],
                    output[x * output_stride() + c],
                    std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) *
                                          1.0e-2f))
            << "x = " << x << ", channel = " << c
            << ", channels = " << channels()
            << ", kernel_size = " << kernel_size()
            << ", first_pass_tile = " << first_pass_tile()
            << ", middle_pass_tile = " << middle_pass_tile()
            << ", last_pass_tile = " << last_pass_tile();
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f32_dwconv_unipass_ukernel_fn dwconv, const void*) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<const float*> indirection((width() - 1) * step() + kernel_tile());
  xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
                           indirection.size() * channels());
  xnnpack::Buffer<float> kernel(channels() * kernel_tile());
  xnnpack::Buffer<float> bias(channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights((kernel_tile() + 1) *
                                                          packed_channels());
  xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
  xnnpack::Buffer<float> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
    std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
    xnn_pack_f32_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(), channel_tile(),
        channel_tile(), channel_tile(), kernel.data(), bias.data(),
        /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
        /*params=*/nullptr);
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
            acc += indirection[x * step() + k][c + input_offset()] *
                   kernel[c * kernel_tile() + k];
          }
        }
        output_ref[x * channels() + c] = acc;
      }
    }

    // Call optimized micro-kernel.
    dwconv(channels(), width(), indirection.data(), packed_weights.data(),
           output.data(), step() * sizeof(void*),
           (output_stride() - channels()) * sizeof(float),
           input_offset() * sizeof(float), zero.data(), nullptr);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_NEAR(output_ref[x * channels() + c],
                    output[x * output_stride() + c],
                    std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c;
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f32_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
    xnn_init_f32_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<const float*> indirection((width() - 1) * step() + kernel_tile());
  xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
                           indirection.size() * channels());
  xnnpack::Buffer<float> kernel(channels() * kernel_tile());
  xnnpack::Buffer<float> bias(channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights((kernel_tile() + 1) *
                                                          packed_channels());
  xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
  xnnpack::Buffer<float> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
    std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
    xnn_pack_f32_dwconv_ghw_w(
        kernel_tile(), 0, 0, kernel_tile(), 1, channels(), channel_tile(),
        channel_tile(), channel_tile(), kernel.data(), bias.data(),
        /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
        /*params=*/nullptr);
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
            acc += indirection[x * step() + k][c + input_offset()] *
                   kernel[c * kernel_tile() + k];
          }
        }
        output_ref[x * channels() + c] = acc;
      }
    }

    // Compute clamping parameters.
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float output_min = accumulated_min + accumulated_range / 255.0f *
                                                   static_cast<float>(qmin());
    const float output_max =
        accumulated_max -
        accumulated_range / 255.0f * static_cast<float>(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, output_min, output_max);

    // Clamp reference results.
    for (float& output_val : output_ref) {
      output_val = std::max(std::min(output_val, output_max), output_min);
    }

    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(), step() * sizeof(void*),
                  (output_stride() - channels()) * sizeof(float),
                  input_offset() * sizeof(float), zero.data(), &params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(output[x * output_stride() + c], output_min)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(output[x * output_stride() + c], output_max)
            << "x = " << x << ", channel = " << c;
        EXPECT_NEAR(output_ref[x * channels() + c],
                    output[x * output_stride() + c],
                    std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c;
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f32_dwconv_multipass_ukernel_fn dwconv, const void*) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
  xnnpack::Buffer<const float*> indirection((width() - 1) * step() + tile_size);
  xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
                           indirection.size() * channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> buffer(
      XNN_MULTIPASS_EXTRA_BYTES / sizeof(float) + channels());
  xnnpack::Buffer<float> kernel(channels() * kernel_size());
  xnnpack::Buffer<float> bias(channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      xnn_dwconv_multipass_weights_size(tile_size, channels(), channel_tile(),
                                        channel_subtile(), channel_round(),
                                        /*bias_element_size=*/sizeof(float),
                                        /*log2_filter_element_size=*/2,
                                        /*extra_weights_byte=*/0) /
      sizeof(float));
  xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
  xnnpack::Buffer<float> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
    std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
    xnn_pack_f32_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(),
        1, channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
        /*params=*/nullptr);
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
            acc += indirection[x * step() + k][c + input_offset()] *
                   kernel[c * kernel_size() + k];
          }
        }
        output_ref[x * channels() + c] = acc;
      }
    }

    // input_stride is step() - first and middle pass
    size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()),
                        middle_pass_tile());
    const int input_advanced =
        first_pass_tile() + num_middle_pass * middle_pass_tile();
    int input_stride_elements = step() - input_advanced;
    // Call optimized micro-kernel.
    dwconv(channels(), width(), indirection.data(), packed_weights.data(),
           output.data(), input_stride_elements * sizeof(void*),
           (output_stride() - channels()) * sizeof(float),
           input_offset() * sizeof(float), zero.data(), kernel_size(),
           buffer.data(), nullptr);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_NEAR(output_ref[x * channels() + c],
                    output[x * output_stride() + c],
                    std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c
            << " channels = " << channels()
            << " kernel_size = " << kernel_size()
            << " first_pass_tile = " << first_pass_tile()
            << " middle_pass_tile = " << middle_pass_tile()
            << " last_pass_tile = " << last_pass_tile();
      }
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f32_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
    xnn_init_f32_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
      kernel_size(), first_pass_tile(), middle_pass_tile(), last_pass_tile());
  xnnpack::Buffer<const float*> indirection((width() - 1) * step() + tile_size);
  xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
                           indirection.size() * channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> buffer(
      XNN_MULTIPASS_EXTRA_BYTES / sizeof(float) + channels());
  xnnpack::Buffer<float> kernel(channels() * kernel_size());
  xnnpack::Buffer<float> bias(channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      xnn_dwconv_multipass_weights_size(tile_size, channels(), channel_tile(),
                                        channel_subtile(), channel_round(),
                                        /*bias_element_size=*/sizeof(float),
                                        /*log2_filter_element_size=*/2,
                                        /*extra_weights_byte=*/0) /
      sizeof(float));
  xnnpack::Buffer<float> zero(channels() + XNN_EXTRA_BYTES / sizeof(float), 0.0f);
  xnnpack::Buffer<float> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
    std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
    std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

    std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
    xnn_pack_f32_dwconv_ghw_w(
        first_pass_tile(), middle_pass_tile(), last_pass_tile(), kernel_size(),
        1, channels(), channel_tile(), channel_subtile(), channel_round(),
        kernel.data(), bias.data(), /*scale=*/nullptr, packed_weights.data(),
        /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0,
        /*params=*/nullptr);
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
            acc += indirection[x * step() + k][c + input_offset()] *
                   kernel[c * kernel_size() + k];
          }
        }
        output_ref[x * channels() + c] = acc;
      }
    }

    // Compute clamping parameters.
    const float accumulated_min =
        *std::min_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_max =
        *std::max_element(output_ref.cbegin(), output_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float output_min = accumulated_min + accumulated_range / 255.0f *
                                                   static_cast<float>(qmin());
    const float output_max =
        accumulated_max -
        accumulated_range / 255.0f * static_cast<float>(255 - qmax());

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, output_min, output_max);

    // Clamp reference results.
    for (float& output_val : output_ref) {
      output_val = std::max(std::min(output_val, output_max), output_min);
    }

    // input_stride is step() - first and middle pass
    size_t num_middle_pass =
        divide_round_up(doz(tile_size, first_pass_tile() + last_pass_tile()),
                        middle_pass_tile());
    const int input_advanced =
        first_pass_tile() + num_middle_pass * middle_pass_tile();
    int input_stride_elements = step() - input_advanced;
    // Call optimized micro-kernel.
    dwconv_minmax(channels(), width(), indirection.data(),
                  packed_weights.data(), output.data(),
                  input_stride_elements * sizeof(void*),
                  (output_stride() - channels()) * sizeof(float),
                  input_offset() * sizeof(float), zero.data(), kernel_size(),
                  buffer.data(), &params);

    // Verify results.
    for (size_t x = 0; x < width(); x++) {
      for (size_t c = 0; c < channels(); c++) {
        EXPECT_GE(output[x * output_stride() + c], output_min)
            << "x = " << x << ", channel = " << c;
        EXPECT_LE(output[x * output_stride() + c], output_max)
            << "x = " << x << ", channel = " << c;
        EXPECT_NEAR(output_ref[x * channels() + c],
                    output[x * output_stride() + c],
                    std::abs(output_ref[x * channels() + c]) * 1.0e-5)
            << "x = " << x << ", channel = " << c
            << " channels = " << channels()
            << " kernel_size = " << kernel_size()
            << " first_pass_tile = " << first_pass_tile()
            << " middle_pass_tile = " << middle_pass_tile()
            << " last_pass_tile = " << last_pass_tile();
      }
    }
  }
}
