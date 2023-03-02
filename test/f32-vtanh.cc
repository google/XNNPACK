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


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
