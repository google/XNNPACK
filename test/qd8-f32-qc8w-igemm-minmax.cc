// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc8w-igemm-minmax.yaml
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


TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .cn_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_eq_2_subtile) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_eq_2_subtile_m) {
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
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_eq_2_subtile_n) {
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
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_lt_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_gt_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, k_div_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_gt_2) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_gt_2_strided_cn) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_gt_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_div_2) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_div_2_strided_cn) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_div_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, small_kernel) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .ks(3)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, small_kernel_subtile) {
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
          .ks(3)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_gt_2_small_kernel) {
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
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, n_div_2_small_kernel) {
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
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, strided_cm_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, a_offset) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(1)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(1)
      .n(2)
      .k(k)
      .ks(3)
      .a_offset(13)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, zero) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t mz = 0; mz < 1; mz++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(1)
        .n(2)
        .k(k)
        .ks(3)
        .a_offset(13)
        .zero_index(mz)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .qmin(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .qmax(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_1X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(1)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(1)
    .n(2)
    .k(2)
    .cm_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}


TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .cn_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_eq_2_subtile) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_eq_2_subtile_m) {
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
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_eq_2_subtile_n) {
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
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_lt_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_gt_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, k_div_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_gt_2) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_gt_2_strided_cn) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_gt_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_div_2) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_div_2_strided_cn) {
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
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_div_2_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, small_kernel) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .ks(3)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, small_kernel_subtile) {
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
          .ks(3)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_gt_2_small_kernel) {
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
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, n_div_2_small_kernel) {
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
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, strided_cm_subtile) {
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
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, a_offset) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .ks(3)
      .a_offset(23)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, zero) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t mz = 0; mz < 2; mz++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(2)
        .k(k)
        .ks(3)
        .a_offset(23)
        .zero_index(mz)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .qmin(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .qmax(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_2X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(2)
    .cm_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}


TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(2)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(2)
    .cn_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 3; m++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(2)
      .iterations(1)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(n)
      .k(2)
      .iterations(1)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, small_kernel) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .ks(3)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, small_kernel_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .ks(3)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_gt_2_small_kernel) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, n_div_2_small_kernel) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, a_offset) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .ks(3)
      .a_offset(37)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, zero) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t mz = 0; mz < 3; mz++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(2)
        .k(k)
        .ks(3)
        .a_offset(37)
        .zero_index(mz)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(2)
    .qmin(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(2)
    .qmax(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_3X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(2)
    .cm_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_3x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}


TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_eq_2) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(2)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(2)
    .cn_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_eq_2_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(2)
        .iterations(1)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_eq_2_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(2)
      .iterations(1)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_eq_2_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(2)
      .iterations(1)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_lt_2) {
  for (size_t k = 1; k < 2; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_lt_2_subtile) {
  for (size_t k = 1; k < 2; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_gt_2) {
  for (size_t k = 3; k < 4; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_gt_2_subtile) {
  for (size_t k = 3; k < 4; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_div_2) {
  for (size_t k = 4; k <= 20; k += 2) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, k_div_2_subtile) {
  for (size_t k = 4; k <= 20; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, small_kernel) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .ks(3)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, small_kernel_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .ks(3)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_gt_2_small_kernel) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, n_div_2_small_kernel) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 10; k += 3) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .ks(3)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
      }
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, a_offset) {
  for (size_t k = 1; k <= 10; k += 3) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .ks(3)
      .a_offset(43)
      .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, zero) {
  for (size_t k = 1; k <= 10; k += 3) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(2)
        .k(k)
        .ks(3)
        .a_offset(43)
        .zero_index(mz)
        .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
    }
  }
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(2)
    .qmin(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(2)
    .qmax(128)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}

TEST(QD8_F32_QC8W_IGEMM_MINMAX_4X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(2)
    .cm_stride(5)
    .Test(xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x2__scalar, xnn_init_f32_minmax_scalar_params, xnn_pack_qs8_conv_goki_w);
}
