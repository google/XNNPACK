// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vadd.h>
#include "vadd-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VADD_MINMAX__SSE2, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VAddMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
  }

  TEST(QU8_VADD_MINMAX__SSE2, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 128; batch_size += 24) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, inplace_a) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, inplace_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (float a_scale = 1.0e-2; a_scale < 1.0e+2; a_scale *= 1.7f) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (float b_scale = 1.0e-2; b_scale < 1.0e+2; b_scale *= 1.7f) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (float y_scale = 1.0e-2; y_scale < 1.0e+2; y_scale *= 1.7f) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (int32_t a_zero_point = 0; a_zero_point <= 255; a_zero_point += 51) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .a_zero_point(uint8_t(a_zero_point))
          .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (int32_t b_zero_point = 0; b_zero_point <= 255; b_zero_point += 51) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .b_zero_point(uint8_t(b_zero_point))
          .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .y_zero_point(uint8_t(y_zero_point))
          .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }

  TEST(QU8_VADD_MINMAX__SSE2, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__sse2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VADD_MINMAX__NEON, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VAddMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vadd_minmax_ukernel__neon);
  }

  TEST(QU8_VADD_MINMAX__NEON, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 128; batch_size += 24) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (float a_scale = 1.0e-2; a_scale < 1.0e+2; a_scale *= 1.7f) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qu8_vadd_minmax_ukernel__neon);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (float b_scale = 1.0e-2; b_scale < 1.0e+2; b_scale *= 1.7f) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qu8_vadd_minmax_ukernel__neon);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (float y_scale = 1.0e-2; y_scale < 1.0e+2; y_scale *= 1.7f) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qu8_vadd_minmax_ukernel__neon);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (int32_t a_zero_point = 0; a_zero_point <= 255; a_zero_point += 51) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .a_zero_point(uint8_t(a_zero_point))
          .Test(xnn_qu8_vadd_minmax_ukernel__neon);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (int32_t b_zero_point = 0; b_zero_point <= 255; b_zero_point += 51) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .b_zero_point(uint8_t(b_zero_point))
          .Test(xnn_qu8_vadd_minmax_ukernel__neon);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
        VAddMicrokernelTester()
          .iterations(1)
          .batch_size(batch_size)
          .y_zero_point(uint8_t(y_zero_point))
          .Test(xnn_qu8_vadd_minmax_ukernel__neon);
      }
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }

  TEST(QU8_VADD_MINMAX__NEON, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 128; batch_size += 11) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vadd_minmax_ukernel__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

TEST(QU8_VADD_MINMAX__SCALAR, n_eq_1) {
  VAddMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
}

TEST(QU8_VADD_MINMAX__SCALAR, n_gt_1) {
  for (size_t batch_size = 2; batch_size < 8; batch_size++) {
    VAddMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, inplace_a) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    VAddMicrokernelTester()
      .iterations(1)
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, inplace_b) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    VAddMicrokernelTester()
      .iterations(1)
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    VAddMicrokernelTester()
      .iterations(1)
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, a_scale) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    for (float a_scale = 1.0e-2; a_scale < 1.0e+2; a_scale *= 1.7f) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, b_scale) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    for (float b_scale = 1.0e-2; b_scale < 1.0e+2; b_scale *= 1.7f) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, y_scale) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    for (float y_scale = 1.0e-2; y_scale < 1.0e+2; y_scale *= 1.7f) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, a_zero_point) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    for (int32_t a_zero_point = 0; a_zero_point <= 255; a_zero_point += 51) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .a_zero_point(uint8_t(a_zero_point))
        .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, b_zero_point) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    for (int32_t b_zero_point = 0; b_zero_point <= 255; b_zero_point += 51) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .b_zero_point(uint8_t(b_zero_point))
        .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, y_zero_point) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      VAddMicrokernelTester()
        .iterations(1)
        .batch_size(batch_size)
        .y_zero_point(uint8_t(y_zero_point))
        .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, qmin) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    VAddMicrokernelTester()
      .iterations(1)
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_VADD_MINMAX__SCALAR, qmax) {
  for (size_t batch_size = 1; batch_size < 16; batch_size += 3) {
    VAddMicrokernelTester()
      .iterations(1)
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qu8_vadd_minmax_ukernel__scalar, VAddMicrokernelTester::Variant::Scalar);
  }
}
