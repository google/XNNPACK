// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/cs16-fftr.yaml
//   Generator: tools/generate-fftr-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/fft.h>
#include "fftr-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_FFTR__NEON_X4, samples_eq_256) {
    TEST_REQUIRES_ARM_NEON;
    FftrMicrokernelTester()
      .samples(256)
      .Test(xnn_cs16_fftr_ukernel__neon_x4);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
