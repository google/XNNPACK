// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rdsum.yaml
//   Generator: tools/generate-reduce-discontiguous-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/rdsum-microkernel-tester.h"


TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows(14)
    .channels(channel_tile)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows(14)
    .channels(channel_tile)
    .input_stride(7)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows({1, 14})
    .channels(channel_tile)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows({1, 14})
    .channels(channel_tile)
    .input_stride(7)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows({1, 36, 7})
    .channels(channel_tile)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows({1, 36, 7})
    .channels(channel_tile)
    .input_stride(7)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_2pass_fulltile) {
  RDSumMicrokernelTester()
    .rows(14)
    .channels({8, 32, 4})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_2pass_subtile) {
  RDSumMicrokernelTester()
    .channels({8, 32, 4})
    .rows({1, 14})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile) {
  RDSumMicrokernelTester()
    .channels({8, 32, 4})
    .rows({1, 36, 7})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  RDSumMicrokernelTester()
    .channels({8, 32, 4})
    .rows({1, 36, 7})
    .input_stride(67)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_2pass_fulltile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows(14)
    .channels({1, channel_tile})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_2pass_subtile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 14})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 36, 7})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 36, 7})
    .input_stride(7)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_2pass_fulltile) {
  RDSumMicrokernelTester()
    .rows(14)
    .channels({5, 8})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_2pass_subtile) {
  RDSumMicrokernelTester()
    .channels({5, 8})
    .rows({1, 14})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile) {
  RDSumMicrokernelTester()
    .channels({5, 8})
    .rows({1, 35, 14})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  RDSumMicrokernelTester()
    .channels({5, 8})
    .rows({1, 35, 14})
    .input_stride(23)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, overflow_accumulator) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows(264)
    .channels({1, channel_tile*2})
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scale_scalar_params);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_eq_1v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_eq_1v_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_eq_1v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_eq_1v_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_eq_1v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_eq_1v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_div_1v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_div_1v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_div_1v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_div_1v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 36, 7})
      .input_stride(channel_tile*16+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_lt_1v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_lt_1v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_lt_1v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_lt_1v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_gt_1v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({channel_tile+1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_gt_1v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_gt_1v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, channels_gt_1v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));

    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 35, 14})
      .input_stride(channel_tile*2+11)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U1V, overflow_accumulator) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u1v, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_eq_2v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_eq_2v_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_eq_2v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_eq_2v_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_eq_2v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_eq_2v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_div_2v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_div_2v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_div_2v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_div_2v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 36, 7})
      .input_stride(channel_tile*16+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_lt_2v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_lt_2v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_lt_2v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_lt_2v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_gt_2v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({channel_tile+1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_gt_2v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_gt_2v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, channels_gt_2v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));

    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 35, 14})
      .input_stride(channel_tile*2+11)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U2V, overflow_accumulator) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u2v, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_eq_4v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_eq_4v_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_eq_4v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_eq_4v_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_eq_4v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_eq_4v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_div_4v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_div_4v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_div_4v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_div_4v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile*2, channel_tile*8, channel_tile})
      .rows({1, 36, 7})
      .input_stride(channel_tile*16+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_lt_4v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_lt_4v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_lt_4v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_lt_4v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(channel_tile+1)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_gt_4v_2pass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(14)
      .channels({channel_tile+1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_gt_4v_2pass_subtile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_gt_4v_multipass_fulltile) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, channels_gt_4v_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));

    RDSumMicrokernelTester()
      .channels({channel_tile+1, channel_tile*2})
      .rows({1, 35, 14})
      .input_stride(channel_tile*2+11)
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__RVV_U4V, overflow_accumulator) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t channel_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__rvv_u4v, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_subtile) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_multipass_fulltile) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_2pass_subtile) {
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_multipass_fulltile) {
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_2pass_fulltile) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_2pass_subtile) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_multipass_fulltile) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_2pass_subtile) {
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_multipass_fulltile) {
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, overflow_accumulator) {
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_subtile) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_multipass_fulltile) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_2pass_subtile) {
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_multipass_fulltile) {
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_2pass_fulltile) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_2pass_subtile) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_multipass_fulltile) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_2pass_subtile) {
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_multipass_fulltile) {
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, overflow_accumulator) {
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_fulltile) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_subtile) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_multipass_fulltile) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_2pass_subtile) {
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_multipass_fulltile) {
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_2pass_fulltile) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_2pass_subtile) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_multipass_fulltile) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_2pass_subtile) {
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_multipass_fulltile) {
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, overflow_accumulator) {
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scale_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
