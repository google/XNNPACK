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

namespace {


struct XnnTestParam {
  const char *name;
  AvgPoolMicrokernelTester::Kernel kernel;
  uint64_t arch_flags;
  size_t channel_tile, channel_scaled_tile, primary_tile, incremental_tile;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {

#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, AvgPoolMicrokernelTester::Kernel{ukernel, init_params}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, AvgPoolMicrokernelTester::Kernel{ukernel, init_params}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

#include "src/f16-avgpool/f16-avgpool-minmax.h"
#include "src/f16-pavgpool/f16-pavgpool-minmax.h"
#include "src/f32-avgpool/f32-avgpool-minmax.h"
#include "src/f32-pavgpool/f32-pavgpool-minmax.h"

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS

#define XNN_UKERNEL_MULTIPASS(arch_flags, ukernel, requantize, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, AvgPoolMicrokernelTester::Kernel{ukernel, init_params, requantize}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

#define XNN_UKERNEL_UNIPASS(arch_flags, ukernel, requantize, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, init_params) \
  { #ukernel, AvgPoolMicrokernelTester::Kernel{ukernel, init_params, requantize}, arch_flags, channel_tile, channel_scaled_tile, primary_tile, incremental_tile },

#include "src/qu8-avgpool/qu8-avgpool-minmax.h"

#undef XNN_UKERNEL_MULTIPASS
#undef XNN_UKERNEL_UNIPASS

};


}  // namespace

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .qmin(128)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile)
    .channels(channel_tile)
    .qmax(128)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile != 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .input_offset(channel_tile*2)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_offset_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_zero_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
              .Test(GetParam().kernel);
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
              .Test(GetParam().kernel);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmin_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmax_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_stride_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_step_0) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
              .Test(GetParam().kernel);
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
              .Test(GetParam().kernel);
          }
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmin(128)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  AvgPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmax(128)
    .Test(GetParam().kernel);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*2)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(128)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(128)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().incremental_tile == 0) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile + 1; pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}


TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    }
  }
}



TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}


TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
        .Test(GetParam().kernel);
    }
  }
}


TEST_P(XnnTest, channels_gt_channel_tile_multipass) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel);
      }
    }
  }
}


TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().kernel);
      }
    }
  }
}


TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}



TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(128)
          .Test(GetParam().kernel);
      }
    }
  }
}


TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
          .Test(GetParam().kernel);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels = xnnpack::NextPrime(channels)) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(128)
          .Test(GetParam().kernel);
      }
    }
  }
}


TEST_P(XnnTest, few_output_pixels) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().kernel);
        }
      }
    }
  }
}


TEST_P(XnnTest, few_output_pixels_with_input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}


TEST_P(XnnTest, few_output_pixels_with_zero) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
              .Test(GetParam().kernel);
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
              .Test(GetParam().kernel);
          }
        }
      }
    }
  }
}



TEST_P(XnnTest, few_output_pixels_with_qmin) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}


TEST_P(XnnTest, few_output_pixels_with_qmax) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}


TEST_P(XnnTest, few_output_pixels_with_output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
            .Test(GetParam().kernel);
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
            .Test(GetParam().kernel);
        }
      }
    }
  }
}


TEST_P(XnnTest, few_output_pixels_with_step) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
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
              .Test(GetParam().kernel);
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
              .Test(GetParam().kernel);
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(f16_avgpool_minmax,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);
