// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams-init.h"
#include "maxpool-microkernel-tester.h"
#include "next_prime.h"

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_f32_maxpool_ukernel_fn kernel_fn;
  xnn_init_f32_minmax_params_fn params_fn;
  size_t channel_tile, channel_scaled_tile, primary_tile, incremental_tile;
  int16_t qmin, qmax;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "F32_MAXPOOL_MINMAX_9P8X__SSE_C4", []() { return TEST_REQUIRES_X86_SSE_VALUE; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params,
    /*channel_tile=*/4, /*channel_scaled_tile=*/4,
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  { "F32_MAXPOOL_MINMAX_9P8X__NEON_C4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params,
    /*channel_tile=*/4, /*channel_scaled_tile=*/4,
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4", []() { return true; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params,
    /*channel_tile=*/4, /*channel_scaled_tile=*/4,
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4", []() { return true; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params,
    /*channel_tile=*/4, /*channel_scaled_tile=*/4,
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "F32_MAXPOOL_MINMAX_9P8X__WASM_C1", []() { return true; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params,
    /*channel_tile=*/1, /*channel_scaled_tile=*/1,
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  { "F32_MAXPOOL_MINMAX_9P8X__RVV_C1V", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c1v, xnn_init_f32_minmax_scalar_params,
    /*channel_tile=*/1, /*channel_scaled_tile=*/(1*xnn_init_hardware_config()->vlenb/sizeof(float)),
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  { "F32_MAXPOOL_MINMAX_9P8X__RVV_C2V", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c2v, xnn_init_f32_minmax_scalar_params,
    /*channel_tile=*/2, /*channel_scaled_tile=*/(2*xnn_init_hardware_config()->vlenb/sizeof(float)),
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  { "F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1", []() { return true; },
    xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params,
    /*channel_tile=*/1, /*channel_scaled_tile=*/1,
    /*primary_tile=*/9, /*incremental_tile=*/8,
    /*qmin=*/-16384, /*qmax=*/16384 },
};

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                  xnnpack::NextPrime(GetParam().channel_tile + 1) :
                  channel_tile + 1)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmin(GetParam().qmin)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmax(GetParam().qmax)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 8))
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*8)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 8))
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile) :
                    channel_tile)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmin(GetParam().qmin)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmax(GetParam().qmax)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile) :
                      channel_tile)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*2)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_unipass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = 2; pooling_elements < GetParam().primary_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                  xnnpack::NextPrime(GetParam().channel_tile + 1) :
                  channel_tile + 1)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmin(GetParam().qmin)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  MaxPoolMicrokernelTester()
    .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
    .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
    .channels(channel_tile)
    .qmax(GetParam().qmax)
    .Test(GetParam().kernel_fn, GetParam().params_fn);
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5))
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*5)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 8))
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile) :
                    channel_tile)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmin(GetParam().qmin)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channels)
      .qmax(GetParam().qmax)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                      xnnpack::NextPrime(GetParam().channel_tile) :
                      channel_tile)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile*2)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_fulltile_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
    for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  } else {
    const size_t channel_tile = GetParam().channel_scaled_tile;
    for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(GetParam().primary_tile + GetParam().incremental_tile)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_twopass_subtile_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + 1; pooling_elements < GetParam().primary_tile + GetParam().incremental_tile; pooling_elements++) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(GetParam().channel_scaled_tile == GetParam().channel_tile ?
                    xnnpack::NextPrime(GetParam().channel_tile + 1) :
                    channel_tile+1)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .qmin(GetParam().qmin)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_eq_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
      .channels(channel_tile)
      .qmax(GetParam().qmax)
      .Test(GetParam().kernel_fn, GetParam().params_fn);
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 8))
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(GetParam().qmin)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(GetParam().qmin)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_div_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile * 2; channels < GetParam().channel_tile * 8; channels += GetParam().channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(GetParam().qmax)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(GetParam().qmax)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .input_offset(channel_tile)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmin(GetParam().qmin)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_lt_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  if (GetParam().channel_tile <= 1 || GetParam().channel_tile == GetParam().channel_scaled_tile) {
      GTEST_SKIP();
  }
  const size_t channel_tile = GetParam().channel_scaled_tile;
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
        .channels(channels)
        .qmax(GetParam().qmax)
        .Test(GetParam().kernel_fn, GetParam().params_fn);
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 2))
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(GetParam().qmin)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmin(GetParam().qmin)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, channels_gt_channel_tile_multipass_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t pooling_elements = GetParam().primary_tile + GetParam().incremental_tile;
       pooling_elements <= GetParam().primary_tile + GetParam().incremental_tile * 3;
       pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
      for (size_t channels = GetParam().channel_tile + 1; channels < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(GetParam().qmax)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    } else {
      const size_t channel_tile = GetParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
          .channels(channels)
          .qmax(GetParam().qmax)
          .Test(GetParam().kernel_fn, GetParam().params_fn);
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile, GetParam().primary_tile + GetParam().incremental_tile - 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_input_offset) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile, GetParam().primary_tile + GetParam().incremental_tile - 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*5+1)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmin) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile, GetParam().primary_tile + GetParam().incremental_tile - 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmin(GetParam().qmin)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmin(GetParam().qmin)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_qmax) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile, GetParam().primary_tile + GetParam().incremental_tile - 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmax(GetParam().qmax)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .qmax(GetParam().qmax)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_output_stride) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile, GetParam().primary_tile + GetParam().incremental_tile - 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
            .channels(channels)
            .output_stride(channel_tile*5+1)
            .Test(GetParam().kernel_fn, GetParam().params_fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, few_output_pixels_with_step) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, GetParam().primary_tile, GetParam().primary_tile + GetParam().incremental_tile - 1}}) {
      if (GetParam().channel_scaled_tile == GetParam().channel_tile) {
        for (size_t channels = 1; channels <= GetParam().channel_tile * 5; channels += std::max<size_t>(1, GetParam().channel_tile - 1)) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .step(step)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(GetParam().channel_tile * 5 + 1))
              .Test(GetParam().kernel_fn, GetParam().params_fn);
          }
        }
      } else {
        const size_t channel_tile = GetParam().channel_scaled_tile;
        for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
          for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(GetParam().primary_tile, GetParam().incremental_tile)
              .step(step)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(GetParam().kernel_fn, GetParam().params_fn);
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(f32_maxpool_minmax,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

