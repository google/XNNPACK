// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x16-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSE__4X8_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_1_8_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 16; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x16_transpose_ukernel__4x8_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_4_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(4)
      .block_width(16)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_4_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_5_8_bw_16){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_5_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 9; j < 16; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x16_transpose_ukernel__4x8_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_4_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_4_bw_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSE__4X8_SSE2, bh_4_bw_8_is_16_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__4x8_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSE__8X8_SSE2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__8x8_sse2);
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 16; ++i){
      for(size_t j = 1; j <= 16; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x16_transpose_ukernel__8x8_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__8x8_sse2);
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__8x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__8x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__8x8_sse2);
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_9_16_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__8x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_9_16_bw_16){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x16_transpose_ukernel__8x8_sse2);
    }
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 9; i < 16; ++i){
      for(size_t j = 9; j < 16; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x16_transpose_ukernel__8x8_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__8x8_sse2);
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__8x8_sse2);
  }

  TEST(X16_TRANSPOSE__8X8_SSE2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x16_transpose_ukernel__8x8_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
