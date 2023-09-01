// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc4w-gemm-minmax.yaml
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


TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .cn_stride(3)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 1; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(m)
      .n(1)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 1; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(1)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(1)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 1; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(1)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(1)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 1; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(1)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(1)
      .kr(1)
      .sr(1)
      .m(1)
      .n(1)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 1; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_gt_1) {
  for (uint32_t n = 2; n < 2; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_gt_1_strided_cn) {
  for (uint32_t n = 2; n < 2; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(3)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_gt_1_strided_a) {
  for (uint32_t n = 2; n < 2; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_gt_1_subtile) {
  for (uint32_t n = 2; n < 2; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_div_1) {
  for (uint32_t n = 2; n <= 3; n += 1) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_div_1_strided_cn) {
  for (uint32_t n = 2; n <= 3; n += 1) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(3)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_div_1_strided_a) {
  for (uint32_t n = 2; n <= 3; n += 1) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(1)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, n_div_1_subtile) {
  for (uint32_t n = 2; n <= 3; n += 1) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 1; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(1)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(3)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(1)
    .kr(1)
    .sr(1)
    .m(1)
    .n(1)
    .k(2)
    .cm_stride(3)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .cn_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .cm_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .cn_stride(7)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(2)
    .cm_stride(7)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .cn_stride(11)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_eq_2_subtile) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_eq_2_subtile_m) {
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
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_eq_2_subtile_n) {
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
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_lt_2_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_gt_2_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, k_div_2_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_gt_8) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_gt_8_strided_cn) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_gt_8_strided_a) {
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
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_gt_8_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_div_8) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_div_8_strided_cn) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_div_8_strided_a) {
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
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, n_div_8_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, strided_cm_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(2)
    .cm_stride(11)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .cn_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .cm_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .cn_stride(7)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(2)
    .cm_stride(7)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .cn_stride(11)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 8; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(m)
      .n(8)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 8; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_gt_8) {
  for (uint32_t n = 9; n < 16; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_gt_8_strided_cn) {
  for (uint32_t n = 9; n < 16; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(11)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_gt_8_strided_a) {
  for (uint32_t n = 9; n < 16; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_gt_8_subtile) {
  for (uint32_t n = 9; n < 16; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_div_8) {
  for (uint32_t n = 16; n <= 24; n += 8) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_div_8_strided_cn) {
  for (uint32_t n = 16; n <= 24; n += 8) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(11)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_div_8_strided_a) {
  for (uint32_t n = 16; n <= 24; n += 8) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, n_div_8_subtile) {
  for (uint32_t n = 16; n <= 24; n += 8) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(11)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(2)
    .n(8)
    .k(2)
    .cm_stride(11)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, no_zero_point) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .b_zero_point(0)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .cn_stride(7)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_eq_2_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .a_stride(5)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(2)
      .iterations(1)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_lt_2_strided_a) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_gt_2_strided_a) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .a_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_div_2_strided_a) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .a_stride(23)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(13)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .qmin(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .qmax(128)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}

TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(2)
    .cm_stride(7)
    .b_zero_point(7)
    .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_qc4w_minmax_scalar_params);
}


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .cn_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_eq_2_subtile) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_eq_2_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_eq_2_subtile_n) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_lt_2_subtile) {
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_gt_2_subtile) {
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, k_div_2_subtile) {
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_gt_2) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_gt_2_strided_cn) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(5)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_gt_2_strided_a) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_gt_2_subtile) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_div_2) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_div_2_strided_cn) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(5)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_div_2_strided_a) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, n_div_2_subtile) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X2__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(2)
      .cm_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .cn_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_eq_2_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_eq_2_subtile_m) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_eq_2_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_lt_2_subtile) {
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_gt_2_subtile) {
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(4)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, k_div_2_subtile) {
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(7)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(2)
      .cm_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cn_stride(11)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_eq_2_subtile) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_eq_2_subtile_m) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_eq_2_subtile_n) {
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
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_lt_2_subtile) {
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
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_gt_2_subtile) {
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
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, k_div_2_subtile) {
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
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_gt_8) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_gt_8_strided_cn) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_gt_8_strided_a) {
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
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_gt_8_subtile) {
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
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_div_8) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_div_8_strided_cn) {
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
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_div_8_strided_a) {
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
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, n_div_8_subtile) {
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
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, strided_cm_subtile) {
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
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(1)
      .n(8)
      .k(2)
      .cm_stride(11)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .cn_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_eq_2_subtile) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_eq_2_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(2)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_eq_2_subtile_n) {
    for (uint32_t n = 1; n <= 2; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_lt_2_subtile) {
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_gt_2_subtile) {
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, k_div_2_subtile) {
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_gt_2) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_gt_2_strided_cn) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(5)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_gt_2_strided_a) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_gt_2_subtile) {
    for (uint32_t n = 3; n < 4; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_div_2) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_div_2_strided_cn) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(5)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_div_2_strided_a) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, n_div_2_subtile) {
    for (uint32_t n = 4; n <= 6; n += 2) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 2; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(2)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(5)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X2__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(2)
      .cm_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .cn_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_eq_2_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_eq_2_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_eq_2_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_lt_2_subtile) {
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_gt_2_subtile) {
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(4)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, k_div_2_subtile) {
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(7)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X4__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(2)
      .cm_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .cn_stride(11)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_eq_2_subtile) {
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_eq_2_subtile_m) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_eq_2_subtile_n) {
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_lt_2_subtile) {
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_gt_2_subtile) {
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, k_div_2_subtile) {
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_gt_8) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_gt_8_strided_cn) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_gt_8_strided_a) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_gt_8_subtile) {
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_div_8) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_div_8_strided_cn) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_div_8_strided_a) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, n_div_8_subtile) {
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_2X8__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(2)
      .n(8)
      .k(2)
      .cm_stride(11)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_eq_2) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, no_zero_point) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .b_zero_point(0)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, strided_cn) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .cn_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_eq_2_strided_a) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .a_stride(5)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_eq_2_subtile) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(2)
          .iterations(1)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_eq_2_subtile_m) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(4)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_eq_2_subtile_n) {
    for (uint32_t n = 1; n <= 4; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(2)
        .iterations(1)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_lt_2) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_lt_2_strided_a) {
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(5)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_lt_2_subtile) {
    for (size_t k = 1; k < 2; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_gt_2) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_gt_2_strided_a) {
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(7)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_gt_2_subtile) {
    for (size_t k = 3; k < 4; k++) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_div_2) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_div_2_strided_a) {
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(4)
        .k(k)
        .a_stride(23)
        .b_zero_point(7)
        .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, k_div_2_subtile) {
    for (size_t k = 4; k <= 20; k += 2) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_gt_4) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_gt_4_strided_cn) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_gt_4_strided_a) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_gt_4_subtile) {
    for (uint32_t n = 5; n < 8; n++) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_div_4) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_div_4_strided_cn) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(7)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_div_4_strided_a) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(13)
          .b_zero_point(7)
          .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, n_div_4_subtile) {
    for (uint32_t n = 8; n <= 12; n += 4) {
      for (size_t k = 1; k <= 10; k += 3) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, strided_cm_subtile) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t n = 1; n <= 4; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(4)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(7)
            .iterations(1)
            .b_zero_point(7)
            .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
        }
      }
    }
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, qmin) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .qmin(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, qmax) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .qmax(128)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }

  TEST(QD8_F32_QC4W_GEMM_MINMAX_4X4__WASM, strided_cm) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(2)
      .cm_stride(7)
      .b_zero_point(7)
      .Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_qc4w_minmax_scalar_params);
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
