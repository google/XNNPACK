// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/pad.h>
#include "pad-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PAD_X2__NEON, fulltile_copy_n_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PadMicrokernelTester()
      .m(2)
      .n(4)
      .Test(xnn_x32_pad_x2__neon);
  }

  TEST(X32_PAD_X2__NEON, fulltile_copy_n_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 32; n += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(n)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_copy_n_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      PadMicrokernelTester()
        .m(2)
        .n(n)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_copy_n_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      PadMicrokernelTester()
        .m(2)
        .n(4)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, subtile_copy) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 2; m++) {
      for (size_t n = 1; n < 10; n++) {
        PadMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_x32_pad_x2__neon);
      }
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_lpad_l_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .l(4)
      .Test(xnn_x32_pad_x2__neon);
  }

  TEST(X32_PAD_X2__NEON, fulltile_lpad_l_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t l = 8; l < 32; l += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_lpad_l_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t l = 1; l < 4; l++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_lpad_l_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t l = 5; l < 8; l++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, subtile_lpad) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 2; m++) {
      for (size_t l = 1; l < 10; l++) {
        PadMicrokernelTester()
          .m(m)
          .n(1)
          .l(l)
          .Test(xnn_x32_pad_x2__neon);
      }
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_rpad_r_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .r(4)
      .Test(xnn_x32_pad_x2__neon);
  }

  TEST(X32_PAD_X2__NEON, fulltile_rpad_r_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t r = 8; r < 32; r += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_rpad_r_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t r = 1; r < 4; r++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, fulltile_rpad_r_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t r = 5; r < 8; r++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(r)
        .Test(xnn_x32_pad_x2__neon);
    }
  }

  TEST(X32_PAD_X2__NEON, subtile_rpad) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 2; m++) {
      for (size_t r = 1; r < 10; r++) {
        PadMicrokernelTester()
          .m(m)
          .n(1)
          .r(r)
          .Test(xnn_x32_pad_x2__neon);
      }
    }
  }

  TEST(X32_PAD_X2__NEON, x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m <= 2; m++) {
      for (size_t k = 1; k < 10; k++) {
        PadMicrokernelTester()
          .m(m)
          .n(k)
          .l(k)
          .r(k)
          .x_stride(2 * k + 1)
          .Test(xnn_x32_pad_x2__neon);
      }
    }
  }

  TEST(X32_PAD_X2__NEON, y_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m <= 2; m++) {
      for (size_t k = 1; k < 10; k++) {
        PadMicrokernelTester()
          .m(m)
          .n(2 * k)
          .l(k)
          .r(k)
          .y_stride(5 * k + 3)
          .Test(xnn_x32_pad_x2__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PAD_X2__SSE2, fulltile_copy_n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PadMicrokernelTester()
      .m(2)
      .n(4)
      .Test(xnn_x32_pad_x2__sse2);
  }

  TEST(X32_PAD_X2__SSE2, fulltile_copy_n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 32; n += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(n)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_copy_n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      PadMicrokernelTester()
        .m(2)
        .n(n)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_copy_n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      PadMicrokernelTester()
        .m(2)
        .n(4)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, subtile_copy) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 2; m++) {
      for (size_t n = 1; n < 10; n++) {
        PadMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_x32_pad_x2__sse2);
      }
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_lpad_l_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .l(4)
      .Test(xnn_x32_pad_x2__sse2);
  }

  TEST(X32_PAD_X2__SSE2, fulltile_lpad_l_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t l = 8; l < 32; l += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_lpad_l_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t l = 1; l < 4; l++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_lpad_l_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t l = 5; l < 8; l++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, subtile_lpad) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 2; m++) {
      for (size_t l = 1; l < 10; l++) {
        PadMicrokernelTester()
          .m(m)
          .n(1)
          .l(l)
          .Test(xnn_x32_pad_x2__sse2);
      }
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_rpad_r_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .r(4)
      .Test(xnn_x32_pad_x2__sse2);
  }

  TEST(X32_PAD_X2__SSE2, fulltile_rpad_r_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t r = 8; r < 32; r += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_rpad_r_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t r = 1; r < 4; r++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, fulltile_rpad_r_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t r = 5; r < 8; r++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(r)
        .Test(xnn_x32_pad_x2__sse2);
    }
  }

  TEST(X32_PAD_X2__SSE2, subtile_rpad) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 2; m++) {
      for (size_t r = 1; r < 10; r++) {
        PadMicrokernelTester()
          .m(m)
          .n(1)
          .r(r)
          .Test(xnn_x32_pad_x2__sse2);
      }
    }
  }

  TEST(X32_PAD_X2__SSE2, x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 2; m++) {
      for (size_t k = 1; k < 10; k++) {
        PadMicrokernelTester()
          .m(m)
          .n(k)
          .l(k)
          .r(k)
          .x_stride(2 * k + 1)
          .Test(xnn_x32_pad_x2__sse2);
      }
    }
  }

  TEST(X32_PAD_X2__SSE2, y_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 2; m++) {
      for (size_t k = 1; k < 10; k++) {
        PadMicrokernelTester()
          .m(m)
          .n(2 * k)
          .l(k)
          .r(k)
          .y_stride(5 * k + 3)
          .Test(xnn_x32_pad_x2__sse2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(X32_PAD_X2__PSIMD, fulltile_copy_n_eq_4) {
    TEST_REQUIRES_PSIMD;
    PadMicrokernelTester()
      .m(2)
      .n(4)
      .Test(xnn_x32_pad_x2__psimd);
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_copy_n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 8; n < 32; n += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(n)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_copy_n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      PadMicrokernelTester()
        .m(2)
        .n(n)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_copy_n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      PadMicrokernelTester()
        .m(2)
        .n(4)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, subtile_copy) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 2; m++) {
      for (size_t n = 1; n < 10; n++) {
        PadMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_x32_pad_x2__psimd);
      }
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_lpad_l_eq_4) {
    TEST_REQUIRES_PSIMD;
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .l(4)
      .Test(xnn_x32_pad_x2__psimd);
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_lpad_l_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t l = 8; l < 32; l += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_lpad_l_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t l = 1; l < 4; l++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_lpad_l_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t l = 5; l < 8; l++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, subtile_lpad) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 2; m++) {
      for (size_t l = 1; l < 10; l++) {
        PadMicrokernelTester()
          .m(m)
          .n(1)
          .l(l)
          .Test(xnn_x32_pad_x2__psimd);
      }
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_rpad_r_eq_4) {
    TEST_REQUIRES_PSIMD;
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .r(4)
      .Test(xnn_x32_pad_x2__psimd);
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_rpad_r_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t r = 8; r < 32; r += 4) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_rpad_r_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t r = 1; r < 4; r++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, fulltile_rpad_r_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t r = 5; r < 8; r++) {
      PadMicrokernelTester()
        .m(2)
        .n(1)
        .l(r)
        .Test(xnn_x32_pad_x2__psimd);
    }
  }

  TEST(X32_PAD_X2__PSIMD, subtile_rpad) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 2; m++) {
      for (size_t r = 1; r < 10; r++) {
        PadMicrokernelTester()
          .m(m)
          .n(1)
          .r(r)
          .Test(xnn_x32_pad_x2__psimd);
      }
    }
  }

  TEST(X32_PAD_X2__PSIMD, x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 2; m++) {
      for (size_t k = 1; k < 10; k++) {
        PadMicrokernelTester()
          .m(m)
          .n(k)
          .l(k)
          .r(k)
          .x_stride(2 * k + 1)
          .Test(xnn_x32_pad_x2__psimd);
      }
    }
  }

  TEST(X32_PAD_X2__PSIMD, y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 2; m++) {
      for (size_t k = 1; k < 10; k++) {
        PadMicrokernelTester()
          .m(m)
          .n(2 * k)
          .l(k)
          .r(k)
          .y_stride(5 * k + 3)
          .Test(xnn_x32_pad_x2__psimd);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(X32_PAD_X2__SCALAR, fulltile_copy_n_eq_1) {
  PadMicrokernelTester()
    .m(2)
    .n(1)
    .Test(xnn_x32_pad_x2__scalar);
}

TEST(X32_PAD_X2__SCALAR, fulltile_copy_n_gt_1) {
  for (size_t n = 2; n < 8; n++) {
    PadMicrokernelTester()
      .m(2)
      .n(n)
      .Test(xnn_x32_pad_x2__scalar);
  }
}

TEST(X32_PAD_X2__SCALAR, subtile_copy) {
  for (size_t m = 1; m < 2; m++) {
    for (size_t n = 1; n < 5; n++) {
      PadMicrokernelTester()
        .m(m)
        .n(n)
        .Test(xnn_x32_pad_x2__scalar);
    }
  }
}

TEST(X32_PAD_X2__SCALAR, fulltile_lpad_l_eq_1) {
  PadMicrokernelTester()
    .m(2)
    .n(1)
    .l(1)
    .Test(xnn_x32_pad_x2__scalar);
}

TEST(X32_PAD_X2__SCALAR, fulltile_lpad_l_gt_1) {
  for (size_t l = 2; l < 8; l++) {
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .l(l)
      .Test(xnn_x32_pad_x2__scalar);
  }
}

TEST(X32_PAD_X2__SCALAR, subtile_lpad) {
  for (size_t m = 1; m < 2; m++) {
    for (size_t l = 1; l < 5; l++) {
      PadMicrokernelTester()
        .m(m)
        .n(1)
        .l(l)
        .Test(xnn_x32_pad_x2__scalar);
    }
  }
}

TEST(X32_PAD_X2__SCALAR, fulltile_rpad_r_eq_1) {
  PadMicrokernelTester()
    .m(2)
    .n(1)
    .r(1)
    .Test(xnn_x32_pad_x2__scalar);
}

TEST(X32_PAD_X2__SCALAR, fulltile_rpad_r_gt_1) {
  for (size_t r = 1; r < 8; r++) {
    PadMicrokernelTester()
      .m(2)
      .n(1)
      .l(r)
      .Test(xnn_x32_pad_x2__scalar);
  }
}

TEST(X32_PAD_X2__SCALAR, subtile_rpad) {
  for (size_t m = 1; m < 2; m++) {
    for (size_t r = 1; r < 5; r++) {
      PadMicrokernelTester()
        .m(m)
        .n(1)
        .r(r)
        .Test(xnn_x32_pad_x2__scalar);
    }
  }
}

TEST(X32_PAD_X2__SCALAR, x_stride) {
  for (size_t m = 1; m <= 2; m++) {
    for (size_t k = 1; k < 5; k++) {
      PadMicrokernelTester()
        .m(m)
        .n(k)
        .l(k)
        .r(k)
        .x_stride(2 * k + 1)
        .Test(xnn_x32_pad_x2__scalar);
    }
  }
}

TEST(X32_PAD_X2__SCALAR, y_stride) {
  for (size_t m = 1; m <= 2; m++) {
    for (size_t k = 1; k < 5; k++) {
      PadMicrokernelTester()
        .m(m)
        .n(2 * k)
        .l(k)
        .r(k)
        .y_stride(5 * k + 3)
        .Test(xnn_x32_pad_x2__scalar);
    }
  }
}
