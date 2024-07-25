// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vtanh.yaml
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U40, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U56, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U72, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U24, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U32, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_eq_40) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_div_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_lt_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_gt_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U40, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_eq_48) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_div_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_lt_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_gt_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U48, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_eq_56) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_div_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_lt_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_gt_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U56, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_eq_64) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_div_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_lt_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_gt_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U64, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_eq_72) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_div_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_lt_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_gt_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U72, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_eq_80) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_div_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_lt_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_gt_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__F16C_EXPM1MINUS_RR1_P3H2TS_RCP_U80, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U24, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U24, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U32, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U40, batch_eq_40) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U40, batch_div_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U40, batch_lt_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U40, batch_gt_40) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U40, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U48, batch_eq_48) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U48, batch_div_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U48, batch_lt_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U48, batch_gt_48) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U48, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U56, batch_eq_56) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U56, batch_div_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U56, batch_lt_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U56, batch_gt_56) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U56, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U64, batch_div_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U64, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U72, batch_eq_72) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U72, batch_div_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U72, batch_lt_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U72, batch_gt_72) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U72, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U80, batch_eq_80) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U80, batch_div_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U80, batch_lt_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U80, batch_gt_80) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__F16C_POLYNOMIAL_P19H9T2_U80, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__f16c_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__FMA3_EXPM1MINUS_RR1_P3H2TS_RCP_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u8, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u16, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u24, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u40, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u48, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u56, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u64, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u72, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }

  TEST(F16_VTANH__FMA3_POLYNOMIAL_P19H9T2_U80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u80, xnn_init_f16_tanh_avx_polynomial_p19h9t2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_div_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u8, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u16, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u24, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u32, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u40, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u48, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u56, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u64, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u72, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }

  TEST(F16_VTANH__AVX2_EXPM1MINUS_RR1_P3H2TS_RCP_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__avx2_expm1minus_rr1_p3h2ts_rcp_u80, xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80);
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80);
    }
  }

  TEST(F16_VTANH__AARCH64_NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_DIV_U80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__aarch64_neonfp16arith_expm1minus_rr1_p3h2ts_div_u80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1FMA_U80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1fma_u80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_NR1RECPS_U80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u40);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u48);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u56);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u64);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U72, batch_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U72, batch_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U72, batch_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U72, batch_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U72, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u72);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U80, batch_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80);
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U80, batch_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U80, batch_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U80, batch_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80);
    }
  }

  TEST(F16_VTANH__NEONFP16ARITH_EXPM1MINUS_RR1_P3H2TS_RECPEADJ_U80, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_recpeadj_u80);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
