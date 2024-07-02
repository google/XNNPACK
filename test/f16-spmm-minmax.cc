// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-spmm-minmax.yaml
//   Generator: tools/generate-spmm-test.py

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/spmm.h"
#include "spmm-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(8)
      .nr(1)
      .m(8)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(8)
        .nr(1)
        .m(8)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(8)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, m_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 8; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, m_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 16; m <= 24; m += 8) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, m_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 9; m < 16; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .output_stride(19)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(8)
      .nr(1)
      .m(8)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(8)
        .nr(1)
        .m(8)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(8)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, m_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 8; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, m_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 16; m <= 24; m += 8) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, m_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 9; m < 16; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .output_stride(19)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_PIPELINED, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(8)
      .nr(1)
      .m(8)
      .n(1)
      .k(2)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      SpMMMicrokernelTester()
        .mr(8)
        .nr(1)
        .m(8)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      SpMMMicrokernelTester()
        .mr(8)
        .nr(1)
        .m(8)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      SpMMMicrokernelTester()
        .mr(8)
        .nr(1)
        .m(8)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(8)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, m_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 8; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, m_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 16; m <= 24; m += 8) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, m_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 9; m < 16; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(8)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .output_stride(19)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_8X1__NEONFP16ARITH_X2, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(8)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(16)
      .nr(1)
      .m(16)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(16)
        .nr(1)
        .m(16)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, m_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 16; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, m_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 32; m <= 48; m += 16) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, m_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 17; m < 32; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .output_stride(37)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(16)
      .nr(1)
      .m(16)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(16)
        .nr(1)
        .m(16)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, m_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 16; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, m_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 32; m <= 48; m += 16) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, m_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 17; m < 32; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .output_stride(37)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_PIPELINED, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(16)
      .nr(1)
      .m(16)
      .n(1)
      .k(2)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      SpMMMicrokernelTester()
        .mr(16)
        .nr(1)
        .m(16)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      SpMMMicrokernelTester()
        .mr(16)
        .nr(1)
        .m(16)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      SpMMMicrokernelTester()
        .mr(16)
        .nr(1)
        .m(16)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(16)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, m_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 16; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, m_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 32; m <= 48; m += 16) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, m_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 17; m < 32; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(16)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .output_stride(37)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_16X1__NEONFP16ARITH_X2, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(16)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(24)
      .nr(1)
      .m(24)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(24)
        .nr(1)
        .m(24)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(24)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, m_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 24; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, m_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 48; m <= 72; m += 24) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, m_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 25; m < 48; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .output_stride(53)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(24)
      .nr(1)
      .m(24)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(24)
        .nr(1)
        .m(24)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(24)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, m_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 24; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, m_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 48; m <= 72; m += 24) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, m_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 25; m < 48; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .output_stride(53)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_PIPELINED, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(24)
      .nr(1)
      .m(24)
      .n(1)
      .k(2)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      SpMMMicrokernelTester()
        .mr(24)
        .nr(1)
        .m(24)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      SpMMMicrokernelTester()
        .mr(24)
        .nr(1)
        .m(24)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      SpMMMicrokernelTester()
        .mr(24)
        .nr(1)
        .m(24)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(24)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, m_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 24; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, m_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 48; m <= 72; m += 24) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, m_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 25; m < 48; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(24)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .output_stride(53)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_24X1__NEONFP16ARITH_X2, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(24)
          .nr(1)
          .m(48)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(32)
      .nr(1)
      .m(32)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(32)
        .nr(1)
        .m(32)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, m_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 32; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, m_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 64; m <= 96; m += 32) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, m_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 33; m < 64; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .output_stride(67)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, k_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(32)
      .nr(1)
      .m(32)
      .n(1)
      .k(1)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, k_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 2; k < 10; k++) {
      SpMMMicrokernelTester()
        .mr(32)
        .nr(1)
        .m(32)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, m_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 32; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, m_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 64; m <= 96; m += 32) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, m_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 33; m < 64; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 5; k += 2) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .output_stride(67)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_PIPELINED, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 5; k += 2) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    SpMMMicrokernelTester()
      .mr(32)
      .nr(1)
      .m(32)
      .n(1)
      .k(2)
      .sparsity(0.0f)
      .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 1; k < 2; k++) {
      SpMMMicrokernelTester()
        .mr(32)
        .nr(1)
        .m(32)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 3; k < 4; k++) {
      SpMMMicrokernelTester()
        .mr(32)
        .nr(1)
        .m(32)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t k = 4; k <= 20; k += 2) {
      SpMMMicrokernelTester()
        .mr(32)
        .nr(1)
        .m(32)
        .n(1)
        .k(k)
        .sparsity(0.0f)
        .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, n_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 2; n < 10; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(32)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, m_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 1; m < 32; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, m_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 64; m <= 96; m += 32) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, m_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t m = 33; m < 64; m++) {
      for (uint32_t n = 1; n < 10; n += 2) {
        for (size_t k = 1; k <= 10; k += 3) {
          SpMMMicrokernelTester()
            .mr(32)
            .nr(1)
            .m(m)
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .output_stride(67)
          .sparsity(0.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmin(128)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .qmax(128)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, half_sparse) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(0.5f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_SPMM_MINMAX_32X1__NEONFP16ARITH_X2, zero_weights) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t n = 1; n < 10; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        SpMMMicrokernelTester()
          .mr(32)
          .nr(1)
          .m(64)
          .n(n)
          .k(k)
          .sparsity(1.0f)
          .Test(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
