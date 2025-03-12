// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/maxpool.h"
#include "src/xnnpack/microparams-init.h"
#include "test/maxpool-microkernel-tester.h"
#include "test/next_prime.h"

namespace {

struct XnnTestParam {
  const char* name;
  MaxPoolMicrokernelTester::Kernel kernel;
  uint64_t arch_flags;
  size_t channel_tile, primary_tile;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(
    const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile,   \
                                primary_tile, datatype, params_type, \
                                init_params)                         \
  {#ukernel, MaxPoolMicrokernelTester::Kernel{ukernel, init_params}, \
   arch_flags, channel_tile, primary_tile},

const XnnTestParam xnn_test_params[] = {
#include "src/f16-maxpool/f16-maxpool-minmax.h"
#include "src/f32-maxpool/f32-maxpool-minmax.h"
#include "src/s8-maxpool/s8-maxpool-minmax.h"
#include "src/u8-maxpool/u8-maxpool-minmax.h"
};

#undef XNN_UKERNEL_WITH_PARAMS

}  // namespace

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .channels(channel_tile)
      .input_offset(xnnpack::NextPrime(GetParam().channel_tile + 1))
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .channels(channel_tile)
      .qmin(GetParam().kernel.qmin)
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .channels(channel_tile)
      .qmax(GetParam().kernel.qmax)
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile;
       pooling_elements++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile;
       pooling_elements++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile + 1))
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .channels(channels)
        .qmin(GetParam().kernel.qmin)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .channels(channels)
        .qmax(GetParam().kernel.qmax)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile;
       pooling_elements++) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile;
       pooling_elements++) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(2 * GetParam().primary_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(2 * GetParam().primary_tile)
      .channels(channel_tile)
      .input_offset(xnnpack::NextPrime(GetParam().channel_tile + 1))
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .pooling_elements(2 * GetParam().primary_tile)
      .channels(channel_tile)
      .qmin(GetParam().kernel.qmin)
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  MaxPoolMicrokernelTester()
      .output_pixels(1)
      .pooling_elements(2 * GetParam().primary_tile)
      .channels(channel_tile)
      .qmax(GetParam().kernel.qmax)
      .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1;
       pooling_elements < 2 * GetParam().primary_tile; pooling_elements++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1;
       pooling_elements < 2 * GetParam().primary_tile; pooling_elements++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile + 1))
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(2 * GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(2 * GetParam().primary_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(2 * GetParam().primary_tile)
        .channels(channels)
        .qmin(GetParam().kernel.qmin)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t channels = GetParam().channel_tile + 1;
       channels <
       (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
       channels++) {
    MaxPoolMicrokernelTester()
        .pooling_elements(2 * GetParam().primary_tile)
        .channels(channels)
        .qmax(GetParam().kernel.qmax)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = GetParam().primary_tile + 1;
       pooling_elements < 2 * GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = GetParam().primary_tile + 1;
       pooling_elements < 2 * GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile + 1))
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .qmin(GetParam().kernel.qmin)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t channel_tile = GetParam().channel_tile;
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .channels(channel_tile)
        .qmax(GetParam().kernel.qmax)
        .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .qmin(GetParam().kernel.qmin)
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t pooling_elements = 2 * GetParam().primary_tile;
       pooling_elements <= 2 * GetParam().primary_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = GetParam().channel_tile + 1;
         channels <
         (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
         channels++) {
      MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .channels(channels)
          .qmax(GetParam().kernel.qmax)
          .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, few_output_pixels) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{
             {2, GetParam().primary_tile, 2 * GetParam().primary_tile - 1}}) {
      for (size_t channels = 1; channels <= GetParam().channel_tile * 5;
           channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
        MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .channels(channels)
            .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{
             {2, GetParam().primary_tile, 2 * GetParam().primary_tile - 1}}) {
      for (size_t channels = 1; channels <= GetParam().channel_tile * 5;
           channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
        MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{
             {2, GetParam().primary_tile, 2 * GetParam().primary_tile - 1}}) {
      for (size_t channels = 1; channels <= GetParam().channel_tile * 5;
           channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
        MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .channels(channels)
            .qmin(GetParam().kernel.qmin)
            .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{
             {2, GetParam().primary_tile, 2 * GetParam().primary_tile - 1}}) {
      for (size_t channels = 1; channels <= GetParam().channel_tile * 5;
           channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
        MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .channels(channels)
            .qmax(GetParam().kernel.qmax)
            .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{
             {2, GetParam().primary_tile, 2 * GetParam().primary_tile - 1}}) {
      for (size_t channels = 1; channels <= GetParam().channel_tile * 5;
           channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
        MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_step) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{
             {2, GetParam().primary_tile, 2 * GetParam().primary_tile - 1}}) {
      for (size_t channels = 1; channels <= GetParam().channel_tile * 5;
           channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .step(step)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
              .Test(GetParam().kernel);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(maxpool_minmax, XnnTest,
                         testing::ValuesIn(xnn_test_params), GetTestName);
