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
  TEST(X32_TRANSPOSE__4X4_SSE, offset_12_12) {
    TEST_REQUIRES_X86_SSE;
      TransposeMicrokernelTester()
        .height(12)
        .width(12)
        .h_start(2)
        .h_end(11)
        .w_start(1)
        .w_end(9)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, offset_17_13) {
    TEST_REQUIRES_X86_SSE;
      TransposeMicrokernelTester()
        .height(17)
        .width(13)
        .h_start(5)
        .h_end(13)
        .w_start(4)
        .w_end(12)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, offset_9_39) {
    TEST_REQUIRES_X86_SSE;
      TransposeMicrokernelTester()
        .height(9)
        .width(39)
        .h_start(1)
        .h_end(9)
        .w_start(30)
        .w_end(38)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, offset_16_16) {
    TEST_REQUIRES_X86_SSE;
      TransposeMicrokernelTester()
        .height(16)
        .width(16)
        .h_start(2)
        .h_end(10)
        .w_start(3)
        .w_end(11)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_sse);
  }

  TEST(X32_TRANSPOSE__4X4_SSE, offset_loop_32_32) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 8; i < 32; i += 4){
      for(size_t j = 8; j < 32; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(2)
          .h_end(i - 2)
          .w_start(3)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_32_32) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4; i < 32; i += 4){
      for(size_t j = 4; j < 32; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_33_32) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4 + 1; i < 33; i += 4){
      for(size_t j = 4; j < 32; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_64_32) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4; i < 64; i += 4){
      for(size_t j = 4; j < 32; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_32_64) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4; i < 32; i += 4){
      for(size_t j = 4; j < 64; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_17_34) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4; i < 17; i += 3){
      for(size_t j = 4; j < 34; j += 5){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_32_25) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4; i < 32; i += 7){
      for(size_t j = 4; j < 25; j += 5){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_19_30) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 5; i < 19; i += 4){
      for(size_t j = 6; j < 30; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_20_30) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 4; i < 20; i += 4){
      for(size_t j = 6; j < 30; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_13_19) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 7; i < 13; i += 4){
      for(size_t j = 7; j < 19; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_SSE, loop_1027_127) {
    TEST_REQUIRES_X86_SSE;
    for(size_t i = 7; i < 1027; i += 4){
      for(size_t j = 7; j < 127; j += 4){
        TransposeMicrokernelTester()
          .height(i)
          .width(j)
          .h_start(0)
          .h_end(i)
          .w_start(0)
          .w_end(j)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_sse);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
