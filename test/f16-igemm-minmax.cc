// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-igemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(13)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(13)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(23)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(23)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(67)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(67)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_CORTEX_A76, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_cortex_a76, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(67)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(67)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__AARCH64_NEONFP16ARITH_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__aarch64_neonfp16arith_ld32, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(23)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(23)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(23)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(23)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(83)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(83)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_6X16__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(8)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 8; mz++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(8)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X8__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(8)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 8; m++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m <= 8; m++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(8)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, n_div_16_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 8; m++) {
          GemmMicrokernelTester()
            .mr(8)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(8)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(8)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 8; mz++) {
        GemmMicrokernelTester()
          .mr(8)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(8)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_IGEMM_MINMAX_8X16__NEONFP16ARITH_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(8)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64, xnn_init_f16_minmax_neon_params);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(7)
        .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(7)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X8__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_1x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_gt_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, n_div_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(7)
        .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 1; mz++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(7)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_1X16__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_1x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_gt_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, n_div_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(17)
        .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 3; mz++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(17)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_3X16__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_3x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(23)
        .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(23)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X8__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_4x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_gt_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, n_div_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(23)
        .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(23)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_4X16__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_4x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(5)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(29)
        .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 5; mz++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(5)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(29)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X8__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(5)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_5x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(5)
      .n(16)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(5)
      .n(16)
      .k(1)
      .cn_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 5; m++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 5; m++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(5)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_gt_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_gt_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_div_16_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_div_16_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_gt_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, n_div_16_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 5; m++) {
          GemmMicrokernelTester()
            .mr(5)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(5)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(5)
        .n(16)
        .k(k)
        .ks(3)
        .a_offset(29)
        .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 5; mz++) {
        GemmMicrokernelTester()
          .mr(5)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(5)
          .n(16)
          .k(k)
          .ks(3)
          .a_offset(29)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(5)
      .n(16)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(5)
      .n(16)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_5X16__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(5)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(5)
      .n(16)
      .k(1)
      .cm_stride(19)
      .Test(xnn_f16_igemm_minmax_ukernel_5x16__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(37)
        .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(37)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_6X8__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_6x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, k_eq_1) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(7)
      .n(8)
      .k(1)
      .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(7)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, k_eq_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 7; m++) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(1)
          .iterations(1)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, k_eq_1_subtile_m) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t m = 1; m <= 7; m++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, k_eq_1_subtile_n) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(7)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, k_gt_1) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, k_gt_1_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 2; k < 10; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_gt_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_gt_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_div_8_strided_cn) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_div_8_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, small_kernel_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_gt_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, n_div_8_small_kernel) {
    TEST_REQUIRES_X86_AVX2;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, strided_cm_subtile) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 7; m++) {
          GemmMicrokernelTester()
            .mr(7)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
        }
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, a_offset) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(7)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(7)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(37)
        .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, zero) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t mz = 0; mz < 7; mz++) {
        GemmMicrokernelTester()
          .mr(7)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(7)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(37)
          .zero_index(mz)
          .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, qmin) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(7)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, qmax) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(7)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_IGEMM_MINMAX_7X8__AVX2_BROADCAST, strided_cm) {
    TEST_REQUIRES_X86_AVX2;
    GemmMicrokernelTester()
      .mr(7)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(7)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_f16_igemm_minmax_ukernel_7x8__avx2_broadcast, xnn_init_f16_minmax_avx_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
