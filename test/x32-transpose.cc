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


TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_2_bw_1_4) {
  for(size_t i = 1; i <= 2; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(1)
      .block_width(i)
      .block_height(1)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_2_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_2_2_bw_2){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_2_2_bw_4){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_2_2_bw_3_4) {
  for(size_t i = 2; i < 2; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2_is_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_int);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_2_bw_1_8) {
  for(size_t i = 1; i <= 2; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(8)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(1)
      .block_width(i)
      .block_height(1)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_2_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_2_2_bw_4){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_2_2_bw_8){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_2_2_bw_5_8) {
  for(size_t i = 2; i < 2; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4_is_8_os_2) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_int);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_1_4_bw_1_2) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_4_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_3_4_bw_1){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(1)
      .output_stride(i)
      .block_width(1)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_3_4_bw_2){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_3_4_bw_2_2) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 2; j < 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1_os_4) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1_is_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_int);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_1_4_bw_1_4) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_3_4_bw_2){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_3_4_bw_4){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_3_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2_is_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_int);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_1_4_bw_1_8) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(8)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_4_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_3_4_bw_4){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_3_4_bw_8){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_3_4_bw_5_8) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4_is_8_os_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_int);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_1_8_bw_1_2) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_8_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_8_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_5_8_bw_1){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(1)
      .output_stride(i)
      .block_width(1)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_5_8_bw_2){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_5_8_bw_2_2) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 2; j < 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1_os_8) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1_is_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_int);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_1_8_bw_1_4) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_8_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_8_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_5_8_bw_2){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_5_8_bw_4){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_5_8_bw_3_4) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2_is_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_int);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_1_8_bw_1_8) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(8)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_8_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_8_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_5_8_bw_4){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_5_8_bw_8){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_5_8_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
    }
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4_is_8_os_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_int);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_2_bw_1_4) {
  for(size_t i = 1; i <= 2; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(1)
      .block_width(i)
      .block_height(1)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_2_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_2_2_bw_2){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_2_2_bw_4){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_2_2_bw_3_4) {
  for(size_t i = 2; i < 2; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_bw_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
}

TEST(X32_TRANSPOSE__1X2_SCALAR_FLOAT, bh_1_bw_2_is_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x2_scalar_float);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_2_bw_1_8) {
  for(size_t i = 1; i <= 2; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(8)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(1)
      .block_width(i)
      .block_height(1)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_2_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_2_2_bw_4){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_2_2_bw_8){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_2_2_bw_5_8) {
  for(size_t i = 2; i < 2; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_bw_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
}

TEST(X32_TRANSPOSE__1X4_SCALAR_FLOAT, bh_1_bw_4_is_8_os_2) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__1x4_scalar_float);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_1_4_bw_1_2) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_2_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_4_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_3_4_bw_1){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(1)
      .output_stride(i)
      .block_width(1)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_3_4_bw_2){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_3_4_bw_2_2) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 2; j < 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_2_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_2_bw_1_os_4) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
}

TEST(X32_TRANSPOSE__2X1_SCALAR_FLOAT, bh_2_bw_1_is_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x1_scalar_float);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_1_4_bw_1_4) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_2_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_3_4_bw_2){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_3_4_bw_4){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_3_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_2_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_2_bw_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
}

TEST(X32_TRANSPOSE__2X2_SCALAR_FLOAT, bh_2_bw_2_is_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x2_scalar_float);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_1_4_bw_1_8) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_2_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(8)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_2_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_4_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_3_4_bw_4){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_3_4_bw_8){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_3_4_bw_5_8) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_2_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_2_bw_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
}

TEST(X32_TRANSPOSE__2X4_SCALAR_FLOAT, bh_2_bw_4_is_8_os_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__2x4_scalar_float);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_1_8_bw_1_2) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_4_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_8_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_8_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_5_8_bw_1){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(1)
      .output_stride(i)
      .block_width(1)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_5_8_bw_2){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_5_8_bw_2_2) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 2; j < 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_4_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_4_bw_1_os_8) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
}

TEST(X32_TRANSPOSE__4X1_SCALAR_FLOAT, bh_4_bw_1_is_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x1_scalar_float);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_1_8_bw_1_4) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_8_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_8_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_5_8_bw_2){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_5_8_bw_4){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_5_8_bw_3_4) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_4_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_4_bw_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
}

TEST(X32_TRANSPOSE__4X2_SCALAR_FLOAT, bh_4_bw_2_is_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x2_scalar_float);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_1_8_bw_1_8) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_4_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(8)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_4_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_8_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_8_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_5_8_bw_4){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_5_8_bw_8){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_5_8_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
    }
  }
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_4_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_4_bw_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
}

TEST(X32_TRANSPOSE__4X4_SCALAR_FLOAT, bh_4_bw_4_is_8_os_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x32_transpose_ukernel__4x4_scalar_float);
}

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


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_SSE2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_SSE2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_SSE2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_SSE2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_SSE2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_5_8_bw_4){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_5_8_bw_8){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_SSE2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


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


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_4_bw_4) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_1_8_bw_1_8) {
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_4_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_4_bw_5_8) {
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
    }
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_8_bw_5_8) {
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
    }
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_8_bw_4) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_5_8_bw_4){
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
    }
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_5_8_bw_8){
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
    }
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_5_8_bw_5_8) {
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_4_bw_4_is_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_4_bw_4_os_8) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
  }

  TEST(X32_TRANSPOSE__4X4_WASMSIMD, bh_4_bw_4_is_8_os_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_DEC_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MOV_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_MULTI_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_MULTI_SWITCH_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_DEC_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MOV_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_MULTI_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 8; ++i){
      for(size_t j = 1; j <= 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_5_8_bw_4){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_5_8_bw_8){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 5; i < 8; ++i){
      for(size_t j = 5; j < 8; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X32_TRANSPOSE__4X4_REUSE_SWITCH_ZIP_NEON, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
