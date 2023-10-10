// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-igemm-jit.yaml
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


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(13)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
          .a_offset(13)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(13)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
          .a_offset(13)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A7, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, a_offset) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, zero) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, a_offset) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, zero) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, a_offset) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, zero) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_eq_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 4; k <= 20; k += 2) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, relu) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 10; k += 3) {
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
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH32_NEON_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(43)
        .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(43)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_4X8__AARCH64_NEONFMA_LD128, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 8; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 9; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 12; k <= 40; k += 4) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(16)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 16; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 17; k < 32; k++) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 24; k <= 80; k += 8) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(251)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k <= 40; k += 9) {
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
          .a_offset(251)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, relu) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, a_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_IGEMM_6X8__AARCH64_NEONFMA_LD128, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            &xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMSIMD32_X86_SPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMSIMD32_X86_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, k_eq_1) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, k_eq_1_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, k_eq_1_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, k_eq_1_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, k_gt_1) {
    for (size_t k = 2; k < 10; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, k_gt_1_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_eq_4_subtile) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_eq_4_subtile_m) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_eq_4_subtile_n) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_lt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_gt_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, k_div_4_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_gt_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_gt_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_gt_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_div_8) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_div_8_strided_cn) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_div_8_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, small_kernel) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, small_kernel_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_gt_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, n_div_8_small_kernel) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, strided_cm_subtile) {
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
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, a_offset) {
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
        .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, zero) {
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
          .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_eq_4) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, strided_cn) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cn_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_eq_4_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_eq_4_subtile_m) {
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(m)
        .n(8)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_eq_4_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_lt_4_subtile) {
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_gt_4_subtile) {
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_div_4) {
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, k_div_4_subtile) {
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, unknown_nc_mod_nr) {
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, relu) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .relu(true)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, small_kernel) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, small_kernel_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_gt_8_small_kernel) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, n_div_8_small_kernel) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, strided_cm_subtile) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, a_offset) {
    for (size_t k = 1; k <= 20; k += 5) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(6)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(127)
        .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, zero) {
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t mz = 0; mz < 6; mz++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(4)
          .m(6)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(127)
          .zero_index(mz)
          .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, qmin) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmin(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, qmax) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .qmax(128)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, strided_cm) {
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .cm_stride(11)
      .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .kr(1)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
          fused_operators);
  }
  TEST(GENERATE_F32_IGEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .kr(1)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_conv_goki_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
