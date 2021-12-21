// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-lut.yaml
//   Generator: tools/generate-lut-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/lut.h>
#include "lut-microkernel-tester.h"


TEST(X8_LUT__SCALAR_X1, batch_eq_1) {
  LUTMicrokernelTester()
    .batch_size(1)
    .Test(xnn_x8_lut_ukernel__scalar_x1);
}

TEST(X8_LUT__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x1);
  }
}

TEST(X8_LUT__SCALAR_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_x8_lut_ukernel__scalar_x1);
  }
}

TEST(X8_LUT__SCALAR_X2, batch_eq_2) {
  LUTMicrokernelTester()
    .batch_size(2)
    .Test(xnn_x8_lut_ukernel__scalar_x2);
}

TEST(X8_LUT__SCALAR_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x2);
  }
}

TEST(X8_LUT__SCALAR_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x2);
  }
}

TEST(X8_LUT__SCALAR_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x2);
  }
}

TEST(X8_LUT__SCALAR_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_x8_lut_ukernel__scalar_x2);
  }
}

TEST(X8_LUT__SCALAR_X4, batch_eq_4) {
  LUTMicrokernelTester()
    .batch_size(4)
    .Test(xnn_x8_lut_ukernel__scalar_x4);
}

TEST(X8_LUT__SCALAR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x4);
  }
}

TEST(X8_LUT__SCALAR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x4);
  }
}

TEST(X8_LUT__SCALAR_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x4);
  }
}

TEST(X8_LUT__SCALAR_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_x8_lut_ukernel__scalar_x4);
  }
}

TEST(X8_LUT__SCALAR_X8, batch_eq_8) {
  LUTMicrokernelTester()
    .batch_size(8)
    .Test(xnn_x8_lut_ukernel__scalar_x8);
}

TEST(X8_LUT__SCALAR_X8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x8);
  }
}

TEST(X8_LUT__SCALAR_X8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x8);
  }
}

TEST(X8_LUT__SCALAR_X8, batch_gt_8) {
  for (size_t batch_size = 9; batch_size < 16; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x8);
  }
}

TEST(X8_LUT__SCALAR_X8, inplace) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_x8_lut_ukernel__scalar_x8);
  }
}

TEST(X8_LUT__SCALAR_X16, batch_eq_16) {
  LUTMicrokernelTester()
    .batch_size(16)
    .Test(xnn_x8_lut_ukernel__scalar_x16);
}

TEST(X8_LUT__SCALAR_X16, batch_div_16) {
  for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x16);
  }
}

TEST(X8_LUT__SCALAR_X16, batch_lt_16) {
  for (size_t batch_size = 1; batch_size < 16; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x16);
  }
}

TEST(X8_LUT__SCALAR_X16, batch_gt_16) {
  for (size_t batch_size = 17; batch_size < 32; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_x8_lut_ukernel__scalar_x16);
  }
}

TEST(X8_LUT__SCALAR_X16, inplace) {
  for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_x8_lut_ukernel__scalar_x16);
  }
}

#if XNN_ARCH_ARM64
  TEST(X8_LUT__NEON_TBX128X4_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    LUTMicrokernelTester()
      .batch_size(16)
      .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x16);
  }

  TEST(X8_LUT__NEON_TBX128X4_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x16);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x16);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x16);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x16);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(X8_LUT__NEON_TBX128X4_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    LUTMicrokernelTester()
      .batch_size(32)
      .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x32);
  }

  TEST(X8_LUT__NEON_TBX128X4_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x32);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x32);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x32);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x32);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(X8_LUT__NEON_TBX128X4_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON;
    LUTMicrokernelTester()
      .batch_size(48)
      .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x48);
  }

  TEST(X8_LUT__NEON_TBX128X4_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x48);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x48);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x48);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X48, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x48);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(X8_LUT__NEON_TBX128X4_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    LUTMicrokernelTester()
      .batch_size(64)
      .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x64);
  }

  TEST(X8_LUT__NEON_TBX128X4_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x64);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x64);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x64);
    }
  }

  TEST(X8_LUT__NEON_TBX128X4_X64, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__neon_tbx128x4_x64);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__SSSE3_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSSE3;
    LUTMicrokernelTester()
      .batch_size(16)
      .Test(xnn_x8_lut_ukernel__ssse3_x16);
  }

  TEST(X8_LUT__SSSE3_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__ssse3_x16);
    }
  }

  TEST(X8_LUT__SSSE3_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__ssse3_x16);
    }
  }

  TEST(X8_LUT__SSSE3_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__ssse3_x16);
    }
  }

  TEST(X8_LUT__SSSE3_X16, inplace) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__ssse3_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__SSSE3_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSSE3;
    LUTMicrokernelTester()
      .batch_size(32)
      .Test(xnn_x8_lut_ukernel__ssse3_x32);
  }

  TEST(X8_LUT__SSSE3_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__ssse3_x32);
    }
  }

  TEST(X8_LUT__SSSE3_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__ssse3_x32);
    }
  }

  TEST(X8_LUT__SSSE3_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__ssse3_x32);
    }
  }

  TEST(X8_LUT__SSSE3_X32, inplace) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__ssse3_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    LUTMicrokernelTester()
      .batch_size(16)
      .Test(xnn_x8_lut_ukernel__avx_x16);
  }

  TEST(X8_LUT__AVX_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x16);
    }
  }

  TEST(X8_LUT__AVX_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x16);
    }
  }

  TEST(X8_LUT__AVX_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x16);
    }
  }

  TEST(X8_LUT__AVX_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    LUTMicrokernelTester()
      .batch_size(32)
      .Test(xnn_x8_lut_ukernel__avx_x32);
  }

  TEST(X8_LUT__AVX_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x32);
    }
  }

  TEST(X8_LUT__AVX_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x32);
    }
  }

  TEST(X8_LUT__AVX_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x32);
    }
  }

  TEST(X8_LUT__AVX_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    LUTMicrokernelTester()
      .batch_size(48)
      .Test(xnn_x8_lut_ukernel__avx_x48);
  }

  TEST(X8_LUT__AVX_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x48);
    }
  }

  TEST(X8_LUT__AVX_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x48);
    }
  }

  TEST(X8_LUT__AVX_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x48);
    }
  }

  TEST(X8_LUT__AVX_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx_x48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    LUTMicrokernelTester()
      .batch_size(64)
      .Test(xnn_x8_lut_ukernel__avx_x64);
  }

  TEST(X8_LUT__AVX_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x64);
    }
  }

  TEST(X8_LUT__AVX_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x64);
    }
  }

  TEST(X8_LUT__AVX_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx_x64);
    }
  }

  TEST(X8_LUT__AVX_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx_x64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    LUTMicrokernelTester()
      .batch_size(32)
      .Test(xnn_x8_lut_ukernel__avx2_x32);
  }

  TEST(X8_LUT__AVX2_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x32);
    }
  }

  TEST(X8_LUT__AVX2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x32);
    }
  }

  TEST(X8_LUT__AVX2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x32);
    }
  }

  TEST(X8_LUT__AVX2_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx2_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    LUTMicrokernelTester()
      .batch_size(64)
      .Test(xnn_x8_lut_ukernel__avx2_x64);
  }

  TEST(X8_LUT__AVX2_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x64);
    }
  }

  TEST(X8_LUT__AVX2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x64);
    }
  }

  TEST(X8_LUT__AVX2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x64);
    }
  }

  TEST(X8_LUT__AVX2_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx2_x64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX2_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    LUTMicrokernelTester()
      .batch_size(96)
      .Test(xnn_x8_lut_ukernel__avx2_x96);
  }

  TEST(X8_LUT__AVX2_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x96);
    }
  }

  TEST(X8_LUT__AVX2_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x96);
    }
  }

  TEST(X8_LUT__AVX2_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x96);
    }
  }

  TEST(X8_LUT__AVX2_X96, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx2_x96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX2_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX2;
    LUTMicrokernelTester()
      .batch_size(128)
      .Test(xnn_x8_lut_ukernel__avx2_x128);
  }

  TEST(X8_LUT__AVX2_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x128);
    }
  }

  TEST(X8_LUT__AVX2_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x128);
    }
  }

  TEST(X8_LUT__AVX2_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx2_x128);
    }
  }

  TEST(X8_LUT__AVX2_X128, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx2_x128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX512SKX_VPSHUFB_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    LUTMicrokernelTester()
      .batch_size(64)
      .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x64);
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x64);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x64);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x64);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX512SKX_VPSHUFB_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    LUTMicrokernelTester()
      .batch_size(128)
      .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x128);
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x128);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x128);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x128);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX512SKX_VPSHUFB_X192, batch_eq_192) {
    TEST_REQUIRES_X86_AVX512SKX;
    LUTMicrokernelTester()
      .batch_size(192)
      .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x192);
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X192, batch_div_192) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 384; batch_size < 1920; batch_size += 192) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x192);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X192, batch_lt_192) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 192; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x192);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X192, batch_gt_192) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 193; batch_size < 384; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x192);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X192, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 960; batch_size += 191) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x192);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_LUT__AVX512SKX_VPSHUFB_X256, batch_eq_256) {
    TEST_REQUIRES_X86_AVX512SKX;
    LUTMicrokernelTester()
      .batch_size(256)
      .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x256);
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X256, batch_div_256) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 512; batch_size < 2560; batch_size += 256) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x256);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X256, batch_lt_256) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 256; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x256);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X256, batch_gt_256) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 257; batch_size < 512; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x256);
    }
  }

  TEST(X8_LUT__AVX512SKX_VPSHUFB_X256, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 1280; batch_size += 255) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__avx512skx_vpshufb_x256);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X8_LUT__WASMSIMD_X16, batch_eq_16) {
    LUTMicrokernelTester()
      .batch_size(16)
      .Test(xnn_x8_lut_ukernel__wasmsimd_x16);
  }

  TEST(X8_LUT__WASMSIMD_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x16);
    }
  }

  TEST(X8_LUT__WASMSIMD_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x16);
    }
  }

  TEST(X8_LUT__WASMSIMD_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x16);
    }
  }

  TEST(X8_LUT__WASMSIMD_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X8_LUT__WASMSIMD_X32, batch_eq_32) {
    LUTMicrokernelTester()
      .batch_size(32)
      .Test(xnn_x8_lut_ukernel__wasmsimd_x32);
  }

  TEST(X8_LUT__WASMSIMD_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x32);
    }
  }

  TEST(X8_LUT__WASMSIMD_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x32);
    }
  }

  TEST(X8_LUT__WASMSIMD_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x32);
    }
  }

  TEST(X8_LUT__WASMSIMD_X32, inplace) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X8_LUT__WASMSIMD_X48, batch_eq_48) {
    LUTMicrokernelTester()
      .batch_size(48)
      .Test(xnn_x8_lut_ukernel__wasmsimd_x48);
  }

  TEST(X8_LUT__WASMSIMD_X48, batch_div_48) {
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x48);
    }
  }

  TEST(X8_LUT__WASMSIMD_X48, batch_lt_48) {
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x48);
    }
  }

  TEST(X8_LUT__WASMSIMD_X48, batch_gt_48) {
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x48);
    }
  }

  TEST(X8_LUT__WASMSIMD_X48, inplace) {
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x48);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X8_LUT__WASMSIMD_X64, batch_eq_64) {
    LUTMicrokernelTester()
      .batch_size(64)
      .Test(xnn_x8_lut_ukernel__wasmsimd_x64);
  }

  TEST(X8_LUT__WASMSIMD_X64, batch_div_64) {
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x64);
    }
  }

  TEST(X8_LUT__WASMSIMD_X64, batch_lt_64) {
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x64);
    }
  }

  TEST(X8_LUT__WASMSIMD_X64, batch_gt_64) {
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x64);
    }
  }

  TEST(X8_LUT__WASMSIMD_X64, inplace) {
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_x8_lut_ukernel__wasmsimd_x64);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
