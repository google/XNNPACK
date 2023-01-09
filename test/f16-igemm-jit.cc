// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-igemm-jit.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, strided_cn) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2_subtile) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2_subtile_m) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_eq_2_subtile_n) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_lt_2) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_lt_2_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_gt_2) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_gt_2_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_div_2) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, k_div_2_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, small_kernel) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, small_kernel_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_gt_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, n_div_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, strided_cm_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, a_offset) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, zero) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, qmin) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, qmax) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, strided_cm) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
