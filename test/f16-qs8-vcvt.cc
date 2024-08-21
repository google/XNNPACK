// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-qs8-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_QS8_VCVT__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, xnn_init_f16_qs8_cvt_neonfp16arith_params);
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U8, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u8, xnn_init_f16_qs8_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_QS8_VCVT__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, xnn_init_f16_qs8_cvt_neonfp16arith_params);
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U16, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u16, xnn_init_f16_qs8_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_QS8_VCVT__NEONFP16ARITH_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, xnn_init_f16_qs8_cvt_neonfp16arith_params);
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U24, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U24, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24, xnn_init_f16_qs8_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_QS8_VCVT__NEONFP16ARITH_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, xnn_init_f16_qs8_cvt_neonfp16arith_params);
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U32, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32, xnn_init_f16_qs8_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_QS8_VCVT__NEONFP16ARITH_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(64)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, xnn_init_f16_qs8_cvt_neonfp16arith_params);
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U64, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, xnn_init_f16_qs8_cvt_neonfp16arith_params);
    }
  }

  TEST(F16_QS8_VCVT__NEONFP16ARITH_U64, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64, xnn_init_f16_qs8_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U1, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U2, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U3, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_FMAGIC_U4, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U1, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U2, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U3, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}


TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
  }
}

TEST(F16_QS8_VCVT__SCALAR_IMAGIC_U4, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f16_qs8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f16_qs8_cvt_scalar_params);
    }
  }
}
