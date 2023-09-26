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
  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
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
      .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, relu) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, small_kernel) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, small_kernel_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, a_offset) {
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
        .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, zero) {
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
          .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, qmin) {
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
      .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, qmax) {
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
      .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
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
      .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(1)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
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
      .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, relu) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, small_kernel) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, small_kernel_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, a_offset) {
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
        .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, zero) {
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
          .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, qmin) {
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
      .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, qmax) {
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
      .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
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
      .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, relu) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .relu(true)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(2)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, strided_cn) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4_subtile) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_lt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_lt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_gt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_gt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_div_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, k_div_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, relu) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, small_kernel) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, small_kernel_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_gt_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, n_div_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, strided_cm_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, a_offset) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, zero) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, qmin) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, qmax) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, strided_cm) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, strided_cn) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_eq_2_subtile) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_eq_2_subtile_m) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_eq_2_subtile_n) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_lt_2) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_lt_2_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_gt_2) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_gt_2_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_div_2) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, k_div_2_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_gt_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 32; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, relu) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(2)
      .relu(true)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_gt_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_gt_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_div_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_div_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_div_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, small_kernel) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, small_kernel_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_gt_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, n_div_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, strided_cm_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, a_offset) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, zero) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, qmin) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, qmax) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, strided_cm) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, subtile_m_upto_mr) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(2)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, strided_cn) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_lt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_gt_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_div_4) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, k_div_4_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, relu) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GemmMicrokernelTester()
      .mr(6)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(6)
      .n(16)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_strided_cn) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, small_kernel) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, small_kernel_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_gt_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, n_div_16_small_kernel) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, strided_cm_subtile) {
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
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, a_offset) {
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
        .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, zero) {
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
          .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, qmin) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, qmax) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, strided_cm) {
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
      .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
  }

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(6)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
