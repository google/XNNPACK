// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_1_32_bw_1_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 32; ++i){
      for(size_t j = 1; j <= 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_16_bw_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_16_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_32_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_32_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(32)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_17_32_bw_16){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_17_32_bw_32){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_17_32_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      for(size_t j = 17; j < 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_16_bw_16_is_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_16_bw_16_os_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_SSE2, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_1_32_bw_1_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 32; ++i){
      for(size_t j = 1; j <= 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_16_bw_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_16_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_32_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_32_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(32)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_17_32_bw_16){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_17_32_bw_32){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_17_32_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      for(size_t j = 17; j < 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_16_bw_16_is_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_16_bw_16_os_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_SSE2, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
