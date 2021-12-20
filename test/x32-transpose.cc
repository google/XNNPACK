// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_SSE, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_AARCH64_NEON_TBL, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl);
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_TBL, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_tbl);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
  }

  TEST(X32_TRANSPOSE__4X4_NEON_ZIP, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_neon_zip);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
