// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-qc8w-gemm-minmax.yaml
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


TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 1; m++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(1)
      .n(4)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(1)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_1X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(1)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params);
}


TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(2)
      .n(4)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_2X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(2)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params);
}


TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_QC8W_GEMM_MINMAX_4X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params);
}


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cn_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .a_stride(5)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile) {
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
          .k(2)
          .iterations(1)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_m) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_eq_2_subtile_n) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 1; k < 2; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(5)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_lt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 3; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_gt_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t k = 4; k <= 20; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, k_div_2_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
          .a_stride(13)
          .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
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
            .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_FMA;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(2)
      .cm_stride(11)
      .Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
