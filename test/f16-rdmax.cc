// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-rdmax.yaml
//   Generator: tools/generate-reduce-discontiguous-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/reduce-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RDMAX_2P2X__NEONFP16ARITH_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f16_rdmax_ukernel_2p2x__neonfp16arith_c32, ReduceMicrokernelTester::OpType::Max);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_eq_2_2pass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows(4)
    .channels(channel_tile)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_eq_2_2pass_fulltile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows(4)
    .channels(channel_tile)
    .input_stride(5)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_eq_2_2pass_subtile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 4})
    .channels(channel_tile)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_eq_2_2pass_subtile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 4})
    .channels(channel_tile)
    .input_stride(5)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_eq_2_multipass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 11, 2})
    .channels(channel_tile)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_eq_2_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 11, 2})
    .channels(channel_tile)
    .input_stride(5)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_div_2_2pass_fulltile) {
  ReduceMicrokernelTester()
    .rows(4)
    .channels({4, 16, 2})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_div_2_2pass_subtile) {
  ReduceMicrokernelTester()
    .channels({4, 16, 2})
    .rows({1, 4})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_div_2_multipass_fulltile) {
  ReduceMicrokernelTester()
    .channels({4, 16, 2})
    .rows({1, 11, 2})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_div_2_multipass_fulltile_with_input_stride) {
  ReduceMicrokernelTester()
    .channels({4, 16, 2})
    .rows({1, 11, 2})
    .input_stride(37)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_lt_2_2pass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows(4)
    .channels({1, channel_tile})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_lt_2_2pass_subtile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 4})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_lt_2_multipass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 11, 2})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_lt_2_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 11, 2})
    .input_stride(5)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_gt_2_2pass_fulltile) {
  ReduceMicrokernelTester()
    .rows(4)
    .channels({3, 4})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_gt_2_2pass_subtile) {
  ReduceMicrokernelTester()
    .channels({3, 4})
    .rows({1, 4})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_gt_2_multipass_fulltile) {
  ReduceMicrokernelTester()
    .channels({3, 4})
    .rows({1, 10, 4})
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}

TEST(F16_RDMAX_2P2X__SCALAR_C2, channels_gt_2_multipass_fulltile_with_input_stride) {
  ReduceMicrokernelTester()
    .channels({3, 4})
    .rows({1, 10, 4})
    .input_stride(17)
    .Test(xnn_f16_rdmax_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Max);
}
