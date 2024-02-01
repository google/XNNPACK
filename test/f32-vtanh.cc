// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vtanh.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5TS_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U1, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U2, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__WASM_EXPM1MINUS_RR1_P6H5TS_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasm_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5TS_NR2_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR1_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5TS_NR2_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2TS_PERM_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5TS_NR2_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5ts_nr2_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5TS_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5ts_nr1adj_u80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_DIV_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3TS_PERM_NR1ADJ_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_DIV_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_GATHER_NR1ADJ_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_gather_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_DIV_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3TS_PERM_NR1ADJ_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3ts_perm_nr1adj_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_DIV_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_div_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5TS_NR1_U160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5ts_nr1_u160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_MIN_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_ABS_PMIN_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_MAX_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_NABS_PMAX_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_lut8_p4h3ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_MIN_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_min_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_ABS_PMIN_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_abs_pmin_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_MAX_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_max_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u4, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u8, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u12, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }

  TEST(F32_VTANH__WASMSIMD_EXPM1MINUS_RR1_P6H5TS_DIV_NABS_PMAX_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__wasmsimd_expm1minus_rr1_p6h5ts_div_nabs_pmax_u16, xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_lut8_p4h3ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AARCH64_NEONFMA_EXPM1MINUS_RR1_P6H5TS_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__aarch64_neonfma_expm1minus_rr1_p6h5ts_div_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEON_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_LUT8_P4H3TS_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr1recps1fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2fma_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u4, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u8, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u12, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__NEONFMA_EXPM1MINUS_RR1_P6H5TS_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_p6h5ts_nr2recps_u16, xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
