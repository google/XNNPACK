// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pavgpool.h"
#include "avgpool-microkernel-tester.h"
#include "next_prime.h"

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_qu8_avgpool_minmax_unipass_ukernel_fn uni_fn;
  xnn_qu8_avgpool_minmax_multipass_ukernel_fn multi_fn;
  xnn_init_qu8_avgpool_minmax_params_fn params_fn;
  xnn_qu8_requantize_fn requantize_fn;
  size_t channel_tile, channel_scaled_tile, primary_tile, incremental_tile;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  { "QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; },
    nullptr, xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32,
    /*channel_tile=*/8, /*channel_scaled_tile=*/8,
    /*primary_tile=*/9, /*incremental_tile=*/8 },
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8", []() { return TEST_REQUIRES_X86_SSE2_VALUE; },
    nullptr, xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32,
    /*channel_tile=*/8, /*channel_scaled_tile=*/8,
    /*primary_tile=*/9, /*incremental_tile=*/8 },
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1", []() { return true; },
    nullptr, xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32,
    /*channel_tile=*/1, /*channel_scaled_tile=*/1,
    /*primary_tile=*/9, /*incremental_tile=*/8 },
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  { "QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; },
    xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, nullptr, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32,
    /*channel_tile=*/8, /*channel_scaled_tile=*/8,
    /*primary_tile=*/9, /*incremental_tile=*/0 },
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8", []() { return TEST_REQUIRES_X86_SSE2_VALUE; },
    xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, nullptr, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32,
    /*channel_tile=*/8, /*channel_scaled_tile=*/8,
    /*primary_tile=*/9, /*incremental_tile=*/0 },
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1", []() { return true; },
    xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, nullptr, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32,
    /*channel_tile=*/1, /*channel_scaled_tile=*/1,
    /*primary_tile=*/9, /*incremental_tile=*/0 },
};

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                  xnnpack::NextPrime(GetParam().channel_tile + 1) :
                  channel_tile + 1)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .zero_index_mod2(zero_index_mod2)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .input_scale(scale)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .input_zero_point(zero_point)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .output_scale(scale)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .output_zero_point(zero_point)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .qmin(128)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .qmax(128)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile)
        .channels(channel_tile)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile + 1) :
                      channel_tile + 1)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .input_offset(channel_tile*8)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(channel_tile*8)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .output_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .output_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .qmin(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .qmax(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 8))
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile) :
                    channel_tile)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile) :
                      channel_tile)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .output_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .output_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .qmin(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile)
      .channels(channels)
      .qmax(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile) :
                      channel_tile)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                        xnnpack::NextPrime(GetParam().channel_tile) :
                        channel_tile)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(channel_tile*2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .output_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .output_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
            .zero_index_mod2(zero_index_mod2)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_offset_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*5+1)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_zero_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
              .zero_index_mod2(zero_index_mod2)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .zero_index_mod2(zero_index_mod2)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_scale_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .input_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .input_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_zero_point_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_scale_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .output_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .output_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_zero_point_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmin_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmin(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmin(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmax_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmax(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmax(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_stride_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_stride(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_stride(channel_tile*5+1)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_step_0) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile - 1, GetParam().primary_tile}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .step(step)
              .channels(channels)
              .output_stride(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= 3 * channel_tile; channels += channel_tile-1) {
          for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .step(step)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                  xnnpack::NextPrime(GetParam().channel_tile + 1) :
                  channel_tile + 1)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .zero_index_mod2(zero_index_mod2)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_scale(scale)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_zero_point(zero_point)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .output_scale(scale)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .output_zero_point(zero_point)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmin(128)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmax(128)
    .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile + 1) :
                      channel_tile + 1)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .input_offset(channel_tile*5)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*5)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .output_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .output_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmin(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmax(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*8)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile) :
                    channel_tile)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile) :
                      channel_tile)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .output_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .output_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmin(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmax(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile) :
                      channel_tile)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                        xnnpack::NextPrime(GetParam().channel_tile) :
                        channel_tile)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
            .zero_index_mod2(zero_index_mod2)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile + 1) :
                      channel_tile + 1)
        .zero_index_mod2(zero_index_mod2)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channel_tile)
        .input_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channel_tile)
        .input_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channel_tile)
        .output_scale(scale)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channel_tile)
        .output_zero_point(zero_point)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .qmin(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .qmax(128)
      .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*8)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile)
          .zero_index_mod2(zero_index_mod2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_scale(scale)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_scaled_tile == GetParam().channel_tile) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
            .zero_index_mod2(zero_index_mod2)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .zero_index_mod2(zero_index_mod2)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_scale(scale)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_scale(scale)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_scale(scale)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_scale(scale)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*5+1)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_zero) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
              .zero_index_mod2(zero_index_mod2)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .zero_index_mod2(zero_index_mod2)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_scale) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .output_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .output_scale(scale)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_zero_point) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmin(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmin(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmax(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmax(128)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_stride) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_stride(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_stride(channel_tile*5+1)
            .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_step) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{GetParam().primary_tile + 1, GetParam().primary_tile + GetParam().incremental_tile - 1, GetParam().primary_tile + GetParam().incremental_tile + 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .step(step)
              .channels(channels)
              .output_stride(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= 3 * channel_tile; channels += channel_tile-1) {
          for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .step(step)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(GetParam().uni_fn, GetParam().multi_fn, GetParam().params_fn, GetParam().requantize_fn);
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(qu8_avgpool_minmax,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

