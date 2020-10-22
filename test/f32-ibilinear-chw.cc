// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/ibilinear.h>
#include "ibilinear-chw-microkernel-tester.h"


TEST(F32_IBILINEAR__SCALAR_P1, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p1);
  }
}

TEST(F32_IBILINEAR__SCALAR_P1, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .input_offset(7)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p1);
  }
}

TEST(F32_IBILINEAR__SCALAR_P1, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p1);
  }
}

TEST(F32_IBILINEAR__SCALAR_P2, pixels_eq_2) {
  IBilinearCHWMicrokernelTester()
    .pixels(2)
    .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p2);
}

TEST(F32_IBILINEAR__SCALAR_P2, pixels_div_2) {
  for (size_t pixels = 4; pixels < 20; pixels += 2) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p2);
  }
}

TEST(F32_IBILINEAR__SCALAR_P2, pixels_lt_2) {
  for (size_t pixels = 1; pixels < 2; pixels++) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p2);
  }
}

TEST(F32_IBILINEAR__SCALAR_P2, pixels_gt_2) {
  for (size_t pixels = 3; pixels < 4; pixels++) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p2);
  }
}

TEST(F32_IBILINEAR__SCALAR_P2, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .input_offset(13)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p2);
  }
}

TEST(F32_IBILINEAR__SCALAR_P4, pixels_eq_4) {
  IBilinearCHWMicrokernelTester()
    .pixels(4)
    .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p4);
}

TEST(F32_IBILINEAR__SCALAR_P4, pixels_div_4) {
  for (size_t pixels = 8; pixels < 40; pixels += 4) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p4);
  }
}

TEST(F32_IBILINEAR__SCALAR_P4, pixels_lt_4) {
  for (size_t pixels = 1; pixels < 4; pixels++) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p4);
  }
}

TEST(F32_IBILINEAR__SCALAR_P4, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    IBilinearCHWMicrokernelTester()
      .pixels(pixels)
      .input_offset(23)
      .Test(xnn_f32_ibilinear_chw_ukernel__scalar_p4);
  }
}
