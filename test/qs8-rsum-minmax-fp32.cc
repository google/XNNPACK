// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-rsum-minmax-fp32.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "rsum-microkernel-tester.h"


TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, batch_eq_1) {
  RSumMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(2)
      .scale(scale)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(128)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_eq_2) {
  RSumMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(3)
      .scale(scale)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(256)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_eq_4) {
  RSumMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(5)
      .scale(scale)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(512)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_RSUM_MINMAX_FP32__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U16, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U32_ACC2, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC2, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC4, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEON_U64_ACC4, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neon_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U16, scale) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U16, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32, scale) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U32_ACC2, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u32_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64, scale) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC2, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc2, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC4, scale) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_RSUM_MINMAX_FP32__NEONDOT_U64_ACC4, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_DOT;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__neondot_u64_acc4, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
