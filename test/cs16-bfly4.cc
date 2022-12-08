// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/cs16-bfly4.yaml
//   Generator: tools/generate-bfly4-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/fft.h>
#include "bfly4-microkernel-tester.h"


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X1, samples_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(1)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x1);
  }

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X1, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(4)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x1);
  }

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X1, batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 16; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(1)
        .stride(64)
        .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x1);
    }
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X2, samples_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(1)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x2);
  }

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X2, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(4)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x2);
  }

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X2, batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 16; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(1)
        .stride(64)
        .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x2);
    }
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X4, samples_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(1)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4);
  }

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(4)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4);
  }

  TEST(CS16_BFLY4_SAMPLES1__ASM_AARCH32_NEON_X4, batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 16; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(1)
        .stride(64)
        .Test(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4);
    }
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64

  TEST(CS16_BFLY4_SAMPLES1__NEON, samples_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(1)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__neon);
  }

  TEST(CS16_BFLY4_SAMPLES1__NEON, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .batch(4)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__neon);
  }

  TEST(CS16_BFLY4_SAMPLES1__NEON, batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 16; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(1)
        .stride(64)
        .Test(xnn_cs16_bfly4_samples1_ukernel__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64

  TEST(CS16_BFLY4_SAMPLES4__NEON, samples_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_samples4_ukernel__neon);
  }

  TEST(CS16_BFLY4_SAMPLES4__NEON, samples_eq_4_batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 4; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(4)
        .stride(16)
        .Test(xnn_cs16_bfly4_samples4_ukernel__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64

  TEST(CS16_BFLY4__NEON_X1, samples_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_ukernel__neon_x1);
  }

  TEST(CS16_BFLY4__NEON_X1, samples_eq_4_batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 4; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(4)
        .stride(16)
        .Test(xnn_cs16_bfly4_ukernel__neon_x1);
    }
  }

  TEST(CS16_BFLY4__NEON_X1, samples_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(16)
      .stride(4)
      .Test(xnn_cs16_bfly4_ukernel__neon_x1);
  }

  TEST(CS16_BFLY4__NEON_X1, samples_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(64)
      .stride(1)
      .Test(xnn_cs16_bfly4_ukernel__neon_x1);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64

  TEST(CS16_BFLY4__NEON_X4, samples_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_ukernel__neon_x4);
  }

  TEST(CS16_BFLY4__NEON_X4, samples_eq_4_batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch <= 4; batch++) {
      BFly4MicrokernelTester()
        .batch(batch)
        .samples(4)
        .stride(16)
        .Test(xnn_cs16_bfly4_ukernel__neon_x4);
    }
  }

  TEST(CS16_BFLY4__NEON_X4, samples_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(16)
      .stride(4)
      .Test(xnn_cs16_bfly4_ukernel__neon_x4);
  }

  TEST(CS16_BFLY4__NEON_X4, samples_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    BFly4MicrokernelTester()
      .samples(64)
      .stride(1)
      .Test(xnn_cs16_bfly4_ukernel__neon_x4);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64



TEST(CS16_BFLY4_SAMPLES1__SCALAR, samples_eq_1) {
  BFly4MicrokernelTester()
    .batch(1)
    .samples(1)
    .stride(64)
    .Test(xnn_cs16_bfly4_samples1_ukernel__scalar);
}

TEST(CS16_BFLY4_SAMPLES1__SCALAR, batch_eq_4) {
  BFly4MicrokernelTester()
    .batch(4)
    .samples(1)
    .stride(64)
    .Test(xnn_cs16_bfly4_samples1_ukernel__scalar);
}

TEST(CS16_BFLY4_SAMPLES1__SCALAR, batch_gt_1) {
  for (size_t batch = 2; batch <= 16; batch++) {
    BFly4MicrokernelTester()
      .batch(batch)
      .samples(1)
      .stride(64)
      .Test(xnn_cs16_bfly4_samples1_ukernel__scalar);
  }
}



TEST(CS16_BFLY4_SAMPLES4__SCALAR, samples_eq_4) {
  BFly4MicrokernelTester()
    .samples(4)
    .stride(16)
    .Test(xnn_cs16_bfly4_samples4_ukernel__scalar);
}

TEST(CS16_BFLY4_SAMPLES4__SCALAR, samples_eq_4_batch_gt_1) {
  for (size_t batch = 2; batch <= 4; batch++) {
    BFly4MicrokernelTester()
      .batch(batch)
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_samples4_ukernel__scalar);
  }
}



TEST(CS16_BFLY4__SCALAR_X1, samples_eq_4) {
  BFly4MicrokernelTester()
    .samples(4)
    .stride(16)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x1);
}

TEST(CS16_BFLY4__SCALAR_X1, samples_eq_4_batch_gt_1) {
  for (size_t batch = 2; batch <= 4; batch++) {
    BFly4MicrokernelTester()
      .batch(batch)
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_ukernel__scalar_x1);
  }
}

TEST(CS16_BFLY4__SCALAR_X1, samples_eq_16) {
  BFly4MicrokernelTester()
    .samples(16)
    .stride(4)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x1);
}

TEST(CS16_BFLY4__SCALAR_X1, samples_eq_64) {
  BFly4MicrokernelTester()
    .samples(64)
    .stride(1)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x1);
}



TEST(CS16_BFLY4__SCALAR_X2, samples_eq_4) {
  BFly4MicrokernelTester()
    .samples(4)
    .stride(16)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x2);
}

TEST(CS16_BFLY4__SCALAR_X2, samples_eq_4_batch_gt_1) {
  for (size_t batch = 2; batch <= 4; batch++) {
    BFly4MicrokernelTester()
      .batch(batch)
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_ukernel__scalar_x2);
  }
}

TEST(CS16_BFLY4__SCALAR_X2, samples_eq_16) {
  BFly4MicrokernelTester()
    .samples(16)
    .stride(4)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x2);
}

TEST(CS16_BFLY4__SCALAR_X2, samples_eq_64) {
  BFly4MicrokernelTester()
    .samples(64)
    .stride(1)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x2);
}



TEST(CS16_BFLY4__SCALAR_X4, samples_eq_4) {
  BFly4MicrokernelTester()
    .samples(4)
    .stride(16)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x4);
}

TEST(CS16_BFLY4__SCALAR_X4, samples_eq_4_batch_gt_1) {
  for (size_t batch = 2; batch <= 4; batch++) {
    BFly4MicrokernelTester()
      .batch(batch)
      .samples(4)
      .stride(16)
      .Test(xnn_cs16_bfly4_ukernel__scalar_x4);
  }
}

TEST(CS16_BFLY4__SCALAR_X4, samples_eq_16) {
  BFly4MicrokernelTester()
    .samples(16)
    .stride(4)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x4);
}

TEST(CS16_BFLY4__SCALAR_X4, samples_eq_64) {
  BFly4MicrokernelTester()
    .samples(64)
    .stride(1)
    .Test(xnn_cs16_bfly4_ukernel__scalar_x4);
}
