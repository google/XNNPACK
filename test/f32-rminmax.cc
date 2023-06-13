// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rminmax.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "reduce-microkernel-tester.h"


TEST(F32_RMINMAX__SCALAR_X1, batch_eq_1) {
  ReduceMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_rminmax_ukernel__scalar_x1, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(F32_RMINMAX__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x1, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(F32_RMINMAX__SCALAR_X2_ACC2, batch_eq_2) {
  ReduceMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_rminmax_ukernel__scalar_x2_acc2, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(F32_RMINMAX__SCALAR_X2_ACC2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x2_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X2_ACC2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x2_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X2_ACC2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x2_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(F32_RMINMAX__SCALAR_X3_ACC3, batch_eq_3) {
  ReduceMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f32_rminmax_ukernel__scalar_x3_acc3, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(F32_RMINMAX__SCALAR_X3_ACC3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x3_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X3_ACC3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x3_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X3_ACC3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x3_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(F32_RMINMAX__SCALAR_X4_ACC2, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc2, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(F32_RMINMAX__SCALAR_X4_ACC2, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X4_ACC2, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X4_ACC2, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }
}


TEST(F32_RMINMAX__SCALAR_X4_ACC4, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc4, ReduceMicrokernelTester::OpType::MinMax);
}

TEST(F32_RMINMAX__SCALAR_X4_ACC4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X4_ACC4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }
}

TEST(F32_RMINMAX__SCALAR_X4_ACC4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rminmax_ukernel__scalar_x4_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMINMAX__NEON_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rminmax_ukernel__neon_x4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__NEON_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMINMAX__NEON_X8_ACC2, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rminmax_ukernel__neon_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__NEON_X8_ACC2, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X8_ACC2, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X8_ACC2, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMINMAX__NEON_X12_ACC3, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rminmax_ukernel__neon_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__NEON_X12_ACC3, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X12_ACC3, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X12_ACC3, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMINMAX__NEON_X16_ACC2, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rminmax_ukernel__neon_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__NEON_X16_ACC2, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X16_ACC2, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X16_ACC2, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMINMAX__NEON_X16_ACC4, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rminmax_ukernel__neon_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__NEON_X16_ACC4, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X16_ACC4, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__NEON_X16_ACC4, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__neon_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMINMAX__SSE_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rminmax_ukernel__sse_x4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__SSE_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMINMAX__SSE_X8_ACC2, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rminmax_ukernel__sse_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__SSE_X8_ACC2, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X8_ACC2, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X8_ACC2, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x8_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMINMAX__SSE_X12_ACC3, batch_eq_12) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rminmax_ukernel__sse_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__SSE_X12_ACC3, batch_div_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X12_ACC3, batch_lt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X12_ACC3, batch_gt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x12_acc3, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMINMAX__SSE_X16_ACC2, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rminmax_ukernel__sse_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__SSE_X16_ACC2, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X16_ACC2, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X16_ACC2, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x16_acc2, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMINMAX__SSE_X16_ACC4, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rminmax_ukernel__sse_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(F32_RMINMAX__SSE_X16_ACC4, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X16_ACC4, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(F32_RMINMAX__SSE_X16_ACC4, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rminmax_ukernel__sse_x16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
