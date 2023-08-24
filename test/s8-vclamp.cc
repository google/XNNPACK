// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s8-vclamp.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S8_VCLAMP__NEON_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
  }

  TEST(S8_VCLAMP__NEON_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
    }
  }

  TEST(S8_VCLAMP__NEON_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
    }
  }

  TEST(S8_VCLAMP__NEON_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
    }
  }

  TEST(S8_VCLAMP__NEON_U64, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
    }
  }

  TEST(S8_VCLAMP__NEON_U64, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
      }
    }
  }

  TEST(S8_VCLAMP__NEON_U64, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_s8_vclamp_ukernel__neon_u64, xnn_init_s8_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S8_VCLAMP__SSE2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
  }

  TEST(S8_VCLAMP__SSE2_U64, batch_div_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
    }
  }

  TEST(S8_VCLAMP__SSE2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
    }
  }

  TEST(S8_VCLAMP__SSE2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
    }
  }

  TEST(S8_VCLAMP__SSE2_U64, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
    }
  }

  TEST(S8_VCLAMP__SSE2_U64, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
      }
    }
  }

  TEST(S8_VCLAMP__SSE2_U64, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_s8_vclamp_ukernel__sse2_u64, xnn_init_s8_minmax_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S8_VCLAMP__SSE41_U64, batch_eq_64) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
  }

  TEST(S8_VCLAMP__SSE41_U64, batch_div_64) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
    }
  }

  TEST(S8_VCLAMP__SSE41_U64, batch_lt_64) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
    }
  }

  TEST(S8_VCLAMP__SSE41_U64, batch_gt_64) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
    }
  }

  TEST(S8_VCLAMP__SSE41_U64, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
    }
  }

  TEST(S8_VCLAMP__SSE41_U64, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
      }
    }
  }

  TEST(S8_VCLAMP__SSE41_U64, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_s8_vclamp_ukernel__sse41_u64, xnn_init_s8_minmax_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S8_VCLAMP__WASMSIMD_U64, batch_eq_64) {
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
  }

  TEST(S8_VCLAMP__WASMSIMD_U64, batch_div_64) {
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
    }
  }

  TEST(S8_VCLAMP__WASMSIMD_U64, batch_lt_64) {
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
    }
  }

  TEST(S8_VCLAMP__WASMSIMD_U64, batch_gt_64) {
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
    }
  }

  TEST(S8_VCLAMP__WASMSIMD_U64, inplace) {
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
    }
  }

  TEST(S8_VCLAMP__WASMSIMD_U64, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
      }
    }
  }

  TEST(S8_VCLAMP__WASMSIMD_U64, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_s8_vclamp_ukernel__wasmsimd_u64, xnn_init_s8_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(S8_VCLAMP__SCALAR_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
}

TEST(S8_VCLAMP__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
  }
}

TEST(S8_VCLAMP__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
  }
}

TEST(S8_VCLAMP__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
  }
}

TEST(S8_VCLAMP__SCALAR_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
  }
}

TEST(S8_VCLAMP__SCALAR_U4, qmin) {
  for (uint8_t qmin = 1; qmin < 255; qmin++) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(qmin)
        .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
    }
  }
}

TEST(S8_VCLAMP__SCALAR_U4, qmax) {
  for (uint8_t qmax = 1; qmax < 255; qmax++) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(qmax)
        .Test(xnn_s8_vclamp_ukernel__scalar_u4, xnn_init_s8_minmax_scalar_params);
    }
  }
}
