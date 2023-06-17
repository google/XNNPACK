// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vtanh.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X24, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X32, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X40, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X48, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X56, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X64, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X72, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_X80, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X24, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X32, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_eq_40) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_div_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_lt_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_gt_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X40, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_eq_48) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_div_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_lt_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_gt_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X48, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_eq_56) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_div_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_lt_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_gt_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X56, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_eq_64) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_div_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_lt_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_gt_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X64, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_eq_72) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_div_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_lt_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_gt_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X72, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_eq_80) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_div_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_lt_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_gt_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_X80, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X24, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X24, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X32, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X40, batch_eq_40) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X40, batch_div_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X40, batch_lt_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X40, batch_gt_40) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X40, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X48, batch_eq_48) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X48, batch_div_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X48, batch_lt_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X48, batch_gt_48) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X48, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X56, batch_eq_56) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X56, batch_div_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X56, batch_lt_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X56, batch_gt_56) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X56, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X64, batch_div_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X64, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X72, batch_eq_72) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X72, batch_div_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X72, batch_lt_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X72, batch_gt_72) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X72, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X80, batch_eq_80) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X80, batch_div_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X80, batch_lt_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X80, batch_gt_80) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_X80, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_x80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_x80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x8);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x8);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x8);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x8);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x16);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x16);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x16);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x16);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x24);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x24);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x24);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x24);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x32);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x32);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x32);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x32);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x40);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x40);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x40);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x40);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x48);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x48);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x48);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x48);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x56);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x56);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x56);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x56);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x64);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x64);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x64);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x64);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x72);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x72);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x72);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x72);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x80);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x80);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x80);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x80);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_X80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_x80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x8);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x16);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x24);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x32);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x40);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x48);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x56);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x64);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x72);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x80);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_X80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_x80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x8);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x16);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x24);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x32);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x40);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x48);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x56);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x64);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x72);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x80);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_X80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_x80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x8);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x16);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x24);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x32);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x40);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x48);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x56);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x64);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x72);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))


#if XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x80);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_X80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_x80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && ((XNN_ARCH_ARM || XNN_ARCH_ARM64))
