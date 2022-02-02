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


TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_2_bw_1_4) {
  for(size_t i = 1; i <= 2; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(1)
      .block_width(i)
      .block_height(1)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_2_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_2_2_bw_2){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_2_2_bw_4){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_2_2_bw_3_4) {
  for(size_t i = 2; i < 2; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
}

TEST(X8_TRANSPOSE__1X2_SCALAR_INT, bh_1_bw_2_is_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x2_scalar_int);
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_2_bw_1_8) {
  for(size_t i = 1; i <= 2; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(8)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(1)
      .block_width(i)
      .block_height(1)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_2_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_2_2_bw_4){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_2_2_bw_8){
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_2_2_bw_5_8) {
  for(size_t i = 2; i < 2; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
}

TEST(X8_TRANSPOSE__1X4_SCALAR_INT, bh_1_bw_4_is_8_os_2) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__1x4_scalar_int);
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_1_4_bw_1_2) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_4_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_3_4_bw_1){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(1)
      .output_stride(i)
      .block_width(1)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_3_4_bw_2){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_3_4_bw_2_2) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 2; j < 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1_os_4) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
}

TEST(X8_TRANSPOSE__2X1_SCALAR_INT, bh_2_bw_1_is_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x1_scalar_int);
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_1_4_bw_1_4) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_3_4_bw_2){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_3_4_bw_4){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_3_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
}

TEST(X8_TRANSPOSE__2X2_SCALAR_INT, bh_2_bw_2_is_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x2_scalar_int);
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_1_4_bw_1_8) {
  for(size_t i = 1; i <= 4; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(8)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_4_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_3_4_bw_4){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_3_4_bw_8){
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_3_4_bw_5_8) {
  for(size_t i = 3; i < 4; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
}

TEST(X8_TRANSPOSE__2X4_SCALAR_INT, bh_2_bw_4_is_8_os_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__2x4_scalar_int);
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_1_8_bw_1_2) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_8_bw_2_2) {
  for(size_t i = 2; i < 2; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_8_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_5_8_bw_1){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(1)
      .output_stride(i)
      .block_width(1)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_5_8_bw_2){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_5_8_bw_2_2) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 2; j < 2; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1_os_8) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
}

TEST(X8_TRANSPOSE__4X1_SCALAR_INT, bh_4_bw_1_is_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x1_scalar_int);
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_1_8_bw_1_4) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_8_bw_3_4) {
  for(size_t i = 3; i < 4; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_8_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_5_8_bw_2){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_5_8_bw_4){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_5_8_bw_3_4) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 3; j < 4; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
}

TEST(X8_TRANSPOSE__4X2_SCALAR_INT, bh_4_bw_2_is_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x2_scalar_int);
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_1_8_bw_1_8) {
  for(size_t i = 1; i <= 8; ++i){
    for(size_t j = 1; j <= 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(8)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_8_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_8_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(8)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_5_8_bw_4){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_5_8_bw_8){
  for(size_t i = 5; i < 8; ++i){
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
  }
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_5_8_bw_5_8) {
  for(size_t i = 5; i < 8; ++i){
    for(size_t j = 5; j < 8; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
    }
  }
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
}

TEST(X8_TRANSPOSE__4X4_SCALAR_INT, bh_4_bw_4_is_8_os_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .iterations(1)
    .Test(xnn_x8_transpose_ukernel__4x4_scalar_int);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_1_32_bw_1_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 1; i <= 32; ++i){
      for(size_t j = 1; j <= 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_16_bw_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_16_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_32_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_32_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(32)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_17_32_bw_16){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_17_32_bw_32){
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_17_32_bw_17_32) {
    TEST_REQUIRES_X86_SSE2;
    for(size_t i = 17; i < 32; ++i){
      for(size_t j = 17; j < 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_16_bw_16_is_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_16_bw_16_os_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_SSE2, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2);
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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_1_32_bw_1_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 32; ++i){
      for(size_t j = 1; j <= 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_16_bw_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_16_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_32_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_32_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(32)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_17_32_bw_16){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_17_32_bw_32){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_17_32_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      for(size_t j = 17; j < 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_16_bw_16_is_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_16_bw_16_os_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_DEC_ZIP_NEON, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_1_32_bw_1_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 32; ++i){
      for(size_t j = 1; j <= 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_16_bw_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_16_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_32_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_32_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(32)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_17_32_bw_16){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_17_32_bw_32){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_17_32_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      for(size_t j = 17; j < 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_16_bw_16_is_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_16_bw_16_os_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_MOV_ZIP_NEON, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_1_32_bw_1_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 1; i <= 32; ++i){
      for(size_t j = 1; j <= 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_16_bw_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_16_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_32_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_32_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(32)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_17_32_bw_16){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_17_32_bw_32){
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .iterations(1)
        .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_17_32_bw_17_32) {
    TEST_REQUIRES_ARM_NEON;
    for(size_t i = 17; i < 32; ++i){
      for(size_t j = 17; j < 32; ++j){
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .iterations(1)
          .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_16_bw_16_is_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_16_bw_16_os_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
  }

  TEST(X8_TRANSPOSE__16X16_REUSE_SWITCH_ZIP_NEON, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .iterations(1)
      .Test(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
