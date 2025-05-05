// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test/dwconv-microkernel-tester.h"

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/requantization.h"
#include "test/replicable_random_device.h"

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
    xnn_qu8_dwconv_minmax_ukernel_fn dwconv_minmax,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_qu8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> i32dist(-10000, 10000);
  std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

  xnnpack::Buffer<const uint8_t*> indirection((width() - 1) * step() +
                                              kernel_tile());
  xnnpack::Buffer<uint8_t> input(indirection.size() * channels(),
                                 xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t> kernel(channels() * kernel_tile());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + sizeof(int32_t) / sizeof(uint8_t)) * packed_channels());
  xnnpack::Buffer<uint8_t> zero(channels(), input_zero_point(),
                                xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<uint8_t> output_ref(width() * channels());

  // Use the same packed kernel and indirection buffers for all iterations.
  do {
    std::generate(kernel.begin(), kernel.end(), [&]() { return u8dist(rng); });
  } while (kernel.size() > 1 &&
           *std::max_element(kernel.cbegin(), kernel.cend()) ==
               *std::min_element(kernel.cbegin(), kernel.cend()));
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });

  std::fill(packed_weights.begin(), packed_weights.end(), kernel_zero_point());
  const xnn_qu8_packing_params packing_params = {input_zero_point(),
                                                 kernel_zero_point()};
  xnn_pack_qu8_dwconv_ghw_w(kernel_tile(), kernel_tile(), 1, channels(),
                            channel_tile(), kernel.data(), bias.data(),
                            /*scale=*/nullptr, packed_weights.data(),
                            /*per_tile_extra_bytes=*/0, &packing_params);
  for (size_t i = 0; i < indirection.size(); i++) {
    indirection[i] = input.data() + i * channels() - input_offset();
  }
  std::shuffle(indirection.begin(), indirection.end(), rng);
  if (zero_index() != SIZE_MAX) {
    for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
      indirection[i + zero_index()] = zero.data();
    }
  }

  do {
    std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  } while (input.size() > 1 &&
           *std::max_element(input.cbegin(), input.cend()) ==
               *std::min_element(input.cbegin(), input.cend()));

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
      accumulated_range >= 256 ? static_cast<double>(accumulated_range) / 255.0
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
  dwconv_minmax(channels(), width(), indirection.data(), packed_weights.data(),
                output.data(), step() * sizeof(void*),
                (output_stride() - channels()) * sizeof(uint8_t),
                input_offset() * sizeof(uint8_t), /*input_pixel_stride=*/0,
                zero.data(), &quantization_params);

  // Verify results.
  for (size_t x = 0; x < width(); x++) {
    for (size_t c = 0; c < channels(); c++) {
      ASSERT_GE(static_cast<uint32_t>(output[x * output_stride() + c]),
                static_cast<uint32_t>(qmin()))
          << "x = " << x << ", channel = " << c;
      ASSERT_LE(static_cast<uint32_t>(output[x * output_stride() + c]),
                static_cast<uint32_t>(qmax()))
          << "x = " << x << ", channel = " << c;
      ASSERT_EQ(static_cast<uint32_t>(output[x * output_stride() + c]),
                static_cast<uint32_t>(output_ref[x * channels() + c]))
          << "x = " << x << ", channel = " << c
          << ", accumulator = " << accumulators[x * channels() + c];
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qs8_qc8w_dwconv_minmax_ukernel_fn dwconv_minmax,
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
  xnnpack::Buffer<int8_t> input(indirection.size() * channels(),
                                xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> kernel(channels() * kernel_tile());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + (sizeof(int32_t) + sizeof(float)) / sizeof(int8_t)) *
      packed_channels());
  xnnpack::Buffer<int8_t> zero(channels(),
                               static_cast<int8_t>(input_zero_point() - 0x80),
                               xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<float> scale(channels());
  xnnpack::Buffer<int8_t> output_ref(width() * channels());

  // Use the same packed kernel and indirection buffers for all iterations.
  do {
    std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  } while (kernel.size() > 1 &&
           *std::max_element(kernel.cbegin(), kernel.cend()) ==
               *std::min_element(kernel.cbegin(), kernel.cend()));
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(packed_weights.begin(), packed_weights.end(), 0);
  const xnn_qs8_packing_params packing_params = {
      static_cast<int8_t>(input_zero_point() - 0x80)};
  xnn_pack_qs8_dwconv_ghw_w(
      kernel_tile(), kernel_tile(), 1, channels(), channel_tile(),
      kernel.data(), bias.data(),
      /*scale=*/nullptr, packed_weights.data(),
      /*per_tile_extra_bytes=*/channel_tile() * sizeof(float), &packing_params);
  for (size_t i = 0; i < indirection.size(); i++) {
    indirection[i] = input.data() + i * channels() - input_offset();
  }
  std::shuffle(indirection.begin(), indirection.end(), rng);
  if (zero_index() != SIZE_MAX) {
    for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
      indirection[i + zero_index()] = zero.data();
    }
  }

  do {
    std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  } while (input.size() > 1 &&
           *std::max_element(input.cbegin(), input.cend()) ==
               *std::min_element(input.cbegin(), input.cend()));

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
      channels(), channel_tile(),
      channel_tile() *
          (kernel_tile() * sizeof(int8_t) + sizeof(int32_t) + sizeof(float)),
      scale.data(),
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
  dwconv_minmax(channels(), width(), indirection.data(), packed_weights.data(),
                output.data(), step() * sizeof(void*),
                (output_stride() - channels()) * sizeof(int8_t),
                input_offset() * sizeof(int8_t), /*input_pixel_stride=*/0,
                zero.data(), &minmax_params);

  // Verify results.
  for (size_t x = 0; x < width(); x++) {
    for (size_t c = 0; c < channels(); c++) {
      ASSERT_GE(static_cast<int32_t>(output[x * output_stride() + c]),
                static_cast<int32_t>(qmin()) - 0x80)
          << "x = " << x << ", channel = " << c;
      ASSERT_LE(static_cast<int32_t>(output[x * output_stride() + c]),
                static_cast<int32_t>(qmax()) - 0x80)
          << "x = " << x << ", channel = " << c;
      ASSERT_EQ(static_cast<int32_t>(output[x * output_stride() + c]),
                static_cast<int32_t>(output_ref[x * channels() + c]))
          << "x = " << x << ", channel = " << c
          << ", accumulator = " << accumulators[x * channels() + c];
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_qs8_dwconv_minmax_ukernel_fn dwconv_minmax,
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
  xnnpack::Buffer<int8_t> input(indirection.size() * channels(),
                                xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> kernel(channels() * kernel_tile());
  xnnpack::Buffer<int32_t> bias(channels());
  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + sizeof(int32_t) / sizeof(int8_t)) * packed_channels());
  xnnpack::Buffer<int8_t> zero(channels(),
                               static_cast<int8_t>(input_zero_point() - 0x80),
                               xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<int32_t> accumulators(width() * channels());
  xnnpack::Buffer<int8_t> output_ref(width() * channels());

  // Use the same packed kernel and indirection buffers for all iterations.
  do {
    std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  } while (kernel.size() > 1 &&
           *std::max_element(kernel.cbegin(), kernel.cend()) ==
               *std::min_element(kernel.cbegin(), kernel.cend()));
  std::generate(bias.begin(), bias.end(), [&]() { return i32dist(rng); });
  std::fill(packed_weights.begin(), packed_weights.end(), 0);
  const xnn_qs8_packing_params packing_params = {
      static_cast<int8_t>(input_zero_point() - 0x80)};
  xnn_pack_qs8_dwconv_ghw_w(kernel_tile(), kernel_tile(), 1, channels(),
                            channel_tile(), kernel.data(), bias.data(),
                            /*scale=*/nullptr, packed_weights.data(),
                            /*per_tile_extra_bytes=*/0, &packing_params);
  for (size_t i = 0; i < indirection.size(); i++) {
    indirection[i] = input.data() + i * channels() - input_offset();
  }
  std::shuffle(indirection.begin(), indirection.end(), rng);
  if (zero_index() != SIZE_MAX) {
    for (size_t i = 0; i < indirection.size(); i += kernel_tile()) {
      indirection[i + zero_index()] = zero.data();
    }
  }

  do {
    std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  } while (input.size() > 1 &&
           *std::max_element(input.cbegin(), input.cend()) ==
               *std::min_element(input.cbegin(), input.cend()));

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
      accumulated_range >= 256 ? static_cast<double>(accumulated_range) / 255.0
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
  dwconv_minmax(channels(), width(), indirection.data(), packed_weights.data(),
                output.data(), step() * sizeof(void*),
                (output_stride() - channels()) * sizeof(int8_t),
                input_offset() * sizeof(int8_t), /*input_pixel_stride=*/0,
                zero.data(), &quantization_params);

  // Verify results.
  for (size_t x = 0; x < width(); x++) {
    for (size_t c = 0; c < channels(); c++) {
      ASSERT_GE(static_cast<int32_t>(output[x * output_stride() + c]),
                static_cast<int32_t>(qmin()) - 0x80)
          << "x = " << x << ", channel = " << c;
      ASSERT_LE(static_cast<int32_t>(output[x * output_stride() + c]),
                static_cast<int32_t>(qmax()) - 0x80)
          << "x = " << x << ", channel = " << c;
      ASSERT_EQ(static_cast<int32_t>(output[x * output_stride() + c]),
                static_cast<int32_t>(output_ref[x * channels() + c]))
          << "x = " << x << ", channel = " << c
          << ", accumulator = " << accumulators[x * channels() + c];
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f16_dwconv_minmax_ukernel_fn dwconv_minmax,
    xnn_init_f16_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<const xnn_float16*> indirection((width() - 1) * step() +
                                                  kernel_tile());
  xnnpack::Buffer<xnn_float16> input(indirection.size() * channels(),
                                     xnnpack::XnnExtraBytes);
  xnnpack::Buffer<xnn_float16> kernel(channels() * kernel_tile());
  xnnpack::Buffer<xnn_float16> bias(channels());
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + 1) * packed_channels());
  xnnpack::Buffer<xnn_float16> zero(channels(), 0.0f, xnnpack::XnnExtraBytes);
  xnnpack::Buffer<xnn_float16> output((width() - 1) * output_stride() +
                                      channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  // Use the same packed kernel and indirection buffers for all iterations.
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
  xnn_pack_f16_dwconv_ghw_w(
      kernel_tile(), kernel_tile(), 1, channels(), channel_tile(),
      reinterpret_cast<const uint16_t*>(kernel.data()),
      reinterpret_cast<const uint16_t*>(bias.data()),
      /*scale=*/nullptr, reinterpret_cast<uint16_t*>(packed_weights.data()),
      /*per_tile_extra_bytes=*/0,
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

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

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
  const float output_min =
      xnn_float16(accumulated_min +
                  accumulated_range / 255.0f * static_cast<float>(qmin()));
  const float output_max =
      xnn_float16(accumulated_max - accumulated_range / 255.0f *
                                        static_cast<float>(255 - qmax()));

  // Prepare parameters.
  xnn_f16_minmax_params params;
  init_params(&params, static_cast<xnn_float16>(output_min),
              static_cast<xnn_float16>(output_max));

  // Clamp reference results.
  for (float& output_val : output_ref) {
    output_val = std::max(std::min(output_val, output_max), output_min);
  }

  // Call optimized micro-kernel.
  dwconv_minmax(channels(), width(),
                reinterpret_cast<const xnn_float16**>(indirection.data()),
                packed_weights.data(), output.data(), step() * sizeof(void*),
                (output_stride() - channels()) * sizeof(xnn_float16),
                input_offset() * sizeof(xnn_float16), /*input_pixel_stride=*/0,
                zero.data(), &params);

  // Verify results.
  for (size_t x = 0; x < width(); x++) {
    for (size_t c = 0; c < channels(); c++) {
      ASSERT_GE(output[x * output_stride() + c], output_min)
          << "x = " << x << ", channel = " << c;
      ASSERT_LE(output[x * output_stride() + c], output_max)
          << "x = " << x << ", channel = " << c;
      ASSERT_NEAR(
          output_ref[x * channels() + c], output[x * output_stride() + c],
          std::max(1.0e-4f, std::abs(output_ref[x * channels() + c]) * 1.0e-2f))
          << "x = " << x << ", channel = " << c;
    }
  }
}

void DWConvMicrokernelTester::Test(xnn_f32_dwconv_unipass_ukernel_fn dwconv,
                                   const void*) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<const float*> indirection((width() - 1) * step() +
                                            kernel_tile());
  xnnpack::Buffer<float> input(indirection.size() * channels(),
                               xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> kernel(channels() * kernel_tile());
  xnnpack::Buffer<float> bias(channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + 1) * packed_channels());
  xnnpack::Buffer<float> zero(channels(), 0.0f, xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  // Use the same packed kernel and indirection buffers for all iterations.
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
  xnn_pack_f32_dwconv_ghw_w(kernel_tile(), kernel_tile(), 1, channels(),
                            channel_tile(), kernel.data(), bias.data(),
                            /*scale=*/nullptr, packed_weights.data(),
                            /*per_tile_extra_bytes=*/0,
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

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

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
         input_offset() * sizeof(float), /*input_pixel_stride=*/0, zero.data(),
         nullptr);

  // Verify results.
  for (size_t x = 0; x < width(); x++) {
    for (size_t c = 0; c < channels(); c++) {
      ASSERT_NEAR(output_ref[x * channels() + c],
                  output[x * output_stride() + c],
                  std::abs(output_ref[x * channels() + c]) * 1.0e-5)
          << "x = " << x << ", channel = " << c;
    }
  }
}

void DWConvMicrokernelTester::Test(
    xnn_f32_dwconv_minmax_ukernel_fn dwconv_minmax,
    xnn_init_f32_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  xnnpack::Buffer<const float*> indirection((width() - 1) * step() +
                                            kernel_tile());
  xnnpack::Buffer<float> input(indirection.size() * channels(),
                               xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> kernel(channels() * kernel_tile());
  xnnpack::Buffer<float> bias(channels());
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      (kernel_tile() + 1) * packed_channels());
  xnnpack::Buffer<float> zero(channels(), 0.0f, xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> output((width() - 1) * output_stride() + channels());
  xnnpack::Buffer<float> output_ref(width() * channels());

  // Use the same packed kernel and indirection buffers for all iterations.
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
  xnn_pack_f32_dwconv_ghw_w(kernel_tile(), kernel_tile(), 1, channels(),
                            channel_tile(), kernel.data(), bias.data(),
                            /*scale=*/nullptr, packed_weights.data(),
                            /*per_tile_extra_bytes=*/0,
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

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

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
  const float output_min =
      accumulated_min + accumulated_range / 255.0f * static_cast<float>(qmin());
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
  dwconv_minmax(channels(), width(), indirection.data(), packed_weights.data(),
                output.data(), step() * sizeof(void*),
                (output_stride() - channels()) * sizeof(float),
                input_offset() * sizeof(float), /*input_pixel_stride=*/0,
                zero.data(), &params);

  // Verify results.
  for (size_t x = 0; x < width(); x++) {
    for (size_t c = 0; c < channels(); c++) {
      ASSERT_GE(output[x * output_stride() + c], output_min)
          << "x = " << x << ", channel = " << c;
      ASSERT_LE(output[x * output_stride() + c], output_max)
          << "x = " << x << ", channel = " << c;
      ASSERT_NEAR(output_ref[x * channels() + c],
                  output[x * output_stride() + c],
                  std::abs(output_ref[x * channels() + c]) * 1.0e-5)
          << "x = " << x << ", channel = " << c;
    }
  }
}
