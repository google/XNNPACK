// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/lut.h>
#include "lut-norm-microkernel-tester.h"


TEST(U8_LUT32NORM__SCALAR, n_eq_1) {
  LUTNormMicrokernelTester()
    .n(1)
    .Test(xnn_u8_lut32norm_ukernel__scalar);
}

TEST(U8_LUT32NORM__SCALAR, small_n) {
  for (size_t n = 2; n <= 16; n++) {
    LUTNormMicrokernelTester()
      .n(n)
      .Test(xnn_u8_lut32norm_ukernel__scalar);
  }
}

TEST(U8_LUT32NORM__SCALAR, large_n) {
  for (size_t n = 16; n <= 128; n+=2) {
    LUTNormMicrokernelTester()
      .n(n)
      .Test(xnn_u8_lut32norm_ukernel__scalar);
  }
}

TEST(U8_LUT32NORM__SCALAR, n_eq_1_inplace) {
  LUTNormMicrokernelTester()
    .n(1)
    .inplace(true)
    .Test(xnn_u8_lut32norm_ukernel__scalar);
}

TEST(U8_LUT32NORM__SCALAR, small_n_inplace) {
  for (size_t n = 2; n <= 16; n++) {
    LUTNormMicrokernelTester()
      .n(n)
      .inplace(true)
      .Test(xnn_u8_lut32norm_ukernel__scalar);
  }
}

TEST(U8_LUT32NORM__SCALAR, large_n_inplace) {
  for (size_t n = 16; n <= 128; n+=2) {
    LUTNormMicrokernelTester()
      .n(n)
      .inplace(true)
      .Test(xnn_u8_lut32norm_ukernel__scalar);
  }
}
