// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc8w-igemm-jit-fp32.yaml
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
  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, relu) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t k = 16; k <= 80; k += 8) {
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, relu) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, small_kernel) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
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
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, a_offset) {
    TEST_REQUIRES_ARM_NEON_V8;
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
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, zero) {
    TEST_REQUIRES_ARM_NEON_V8;
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
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8__AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON_V8;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_PLATFORM_JIT
  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, unknown_nc_mod_nr) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, relu) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .relu(true)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, small_kernel_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_gt_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, n_div_8_small_kernel) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .ks(3)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
        }
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, a_offset) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .ks(3)
        .a_offset(163)
        .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, zero) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t mz = 0; mz < 4; mz++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(8)
          .k(k)
          .ks(3)
          .a_offset(163)
          .zero_index(mz)
          .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }

  TEST(GENERATE_QC8_IGEMM_FP32_4X8C4__AARCH32_NEONDOT_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64, xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params, xnn_pack_qs8_conv_goki_w, xnn_qs8_requantize_fp32);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_PLATFORM_JIT
