// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u8-rminmax.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/reduce-microkernel-tester.h"


TEST(U8_RMINMAX__SCALAR_U1, batch_eq_1) {
  ReduceMicrokernelTester()
    .batch_size(1)
    .Test(xnn_u8_rminmax_ukernel__scalar_u1, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(U8_RMINMAX__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u1, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(U8_RMINMAX__SCALAR_U2_ACC2, batch_eq_2) {
  ReduceMicrokernelTester()
    .batch_size(2)
    .Test(xnn_u8_rminmax_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(U8_RMINMAX__SCALAR_U2_ACC2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U2_ACC2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U2_ACC2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(U8_RMINMAX__SCALAR_U3_ACC3, batch_eq_3) {
  ReduceMicrokernelTester()
    .batch_size(3)
    .Test(xnn_u8_rminmax_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(U8_RMINMAX__SCALAR_U3_ACC3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U3_ACC3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U3_ACC3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(U8_RMINMAX__SCALAR_U4_ACC2, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(U8_RMINMAX__SCALAR_U4_ACC2, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U4_ACC2, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U4_ACC2, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(U8_RMINMAX__SCALAR_U4_ACC4, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(U8_RMINMAX__SCALAR_U4_ACC4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U4_ACC4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(U8_RMINMAX__SCALAR_U4_ACC4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rminmax_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_RMINMAX__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_u8_rminmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_RMINMAX__NEON_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_u8_rminmax_ukernel__neon_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__NEON_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_RMINMAX__NEON_U48_ACC3, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(48)
      .Test(xnn_u8_rminmax_ukernel__neon_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__NEON_U48_ACC3, batch_div_48) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U48_ACC3, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U48_ACC3, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_RMINMAX__NEON_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_u8_rminmax_ukernel__neon_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__NEON_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_RMINMAX__NEON_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_u8_rminmax_ukernel__neon_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__NEON_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__NEON_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__neon_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_RMINMAX__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_u8_rminmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_RMINMAX__SSE2_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_u8_rminmax_ukernel__sse2_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__SSE2_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_RMINMAX__SSE2_U48_ACC3, batch_eq_48) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .batch_size(48)
      .Test(xnn_u8_rminmax_ukernel__sse2_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__SSE2_U48_ACC3, batch_div_48) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U48_ACC3, batch_lt_48) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U48_ACC3, batch_gt_48) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u48_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_RMINMAX__SSE2_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__SSE2_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_RMINMAX__SSE2_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__SSE2_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__SSE2_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__sse2_u64_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(U8_RMINMAX__WASMSIMD_U32_ACC2, batch_eq_32) {
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_u8_rminmax_ukernel__wasmsimd_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(U8_RMINMAX__WASMSIMD_U32_ACC2, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__wasmsimd_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__WASMSIMD_U32_ACC2, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__wasmsimd_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(U8_RMINMAX__WASMSIMD_U32_ACC2, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rminmax_ukernel__wasmsimd_u32_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
