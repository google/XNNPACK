// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/cs16-vsquareabs.yaml
//   Generator: tools/generate-vsquareabs-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vsquareabs.h>
#include "vsquareabs-microkernel-tester.h"


TEST(CS16_VSQUAREABS__SCALAR_X1, channels_eq_1) {
  VSquareAbsMicrokernelTester()
    .channels(1)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x1);
}

TEST(CS16_VSQUAREABS__SCALAR_X1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x1);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X2, channels_eq_2) {
  VSquareAbsMicrokernelTester()
    .channels(2)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
}

TEST(CS16_VSQUAREABS__SCALAR_X2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X3, channels_eq_3) {
  VSquareAbsMicrokernelTester()
    .channels(3)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
}

TEST(CS16_VSQUAREABS__SCALAR_X3, channels_div_3) {
  for (size_t channels = 6; channels < 30; channels += 3) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X3, channels_lt_3) {
  for (size_t channels = 1; channels < 3; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X3, channels_gt_3) {
  for (size_t channels = 4; channels < 6; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X4, channels_eq_4) {
  VSquareAbsMicrokernelTester()
    .channels(4)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
}

TEST(CS16_VSQUAREABS__SCALAR_X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    VSquareAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}
