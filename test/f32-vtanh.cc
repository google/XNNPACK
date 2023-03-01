// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vtanh.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}
