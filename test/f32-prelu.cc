// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <xnnpack/isa-checks.h>
#include <xnnpack/prelu.h>

#include "prelu-microkernel-tester.h"


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_PRELU_X4__SSE2, fulltile_n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PReLUMicrokernelTester()
      .m(4)
      .n(4)
      .Test(xnn_f32_prelu_ukernel_x4__sse);
  }

  TEST(F32_PRELU_X4__SSE2, fulltile_n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      PReLUMicrokernelTester()
        .m(4)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__sse);
    }
  }

  TEST(F32_PRELU_X4__SSE2, fulltile_n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      PReLUMicrokernelTester()
        .m(4)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__sse);
    }
  }

  TEST(F32_PRELU_X4__SSE2, fulltile_n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      PReLUMicrokernelTester()
        .m(4)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__sse);
    }
  }

  TEST(F32_PRELU_X4__SSE2, subtile_n_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 4; m++) {
      PReLUMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_prelu_ukernel_x4__sse);
    }
  }

  TEST(F32_PRELU_X4__SSE2, subtile_n_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 4; m++) {
      for (size_t n = 4; n < 64; n += 12) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, subtile_n_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 4; m++) {
      for (size_t n = 5; n < 8; n++) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, subtile_n_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 4; m++) {
      for (size_t n = 1; n < 4; n++) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(n * 3)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, y_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .y_stride(n * 5)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, x_stride_and_y_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(n * 3)
          .y_stride(n * 5)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }

  TEST(F32_PRELU_X4__SSE2, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__sse);
      }
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64


#if !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS
  TEST(F32_PRELU_X4__PSIMD, fulltile_n_eq_4) {
    TEST_REQUIRES_PSIMD;
    PReLUMicrokernelTester()
      .m(4)
      .n(4)
      .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PRELU_X4__PSIMD, fulltile_n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      PReLUMicrokernelTester()
        .m(4)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU_X4__PSIMD, fulltile_n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      PReLUMicrokernelTester()
        .m(4)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU_X4__PSIMD, fulltile_n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      PReLUMicrokernelTester()
        .m(4)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU_X4__PSIMD, subtile_n_eq_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 4; m++) {
      PReLUMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU_X4__PSIMD, subtile_n_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 4; m++) {
      for (size_t n = 4; n < 64; n += 12) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, subtile_n_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 4; m++) {
      for (size_t n = 5; n < 8; n++) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, subtile_n_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 4; m++) {
      for (size_t n = 1; n < 4; n++) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(n * 3)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .y_stride(n * 5)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, x_stride_and_y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(n * 3)
          .y_stride(n * 5)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU_X4__PSIMD, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n < 64; n += 5) {
        PReLUMicrokernelTester()
          .m(m)
          .n(n)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel_x4__psimd, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS


TEST(F32_PRELU_X4__SCALAR, fulltile_n_eq_1) {
  PReLUMicrokernelTester()
    .m(4)
    .n(1)
    .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
}

TEST(F32_PRELU_X4__SCALAR, fulltile_n_gt_1) {
  for (size_t n = 2; n < 16; n++) {
    PReLUMicrokernelTester()
      .m(4)
      .n(n)
      .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PRELU_X4__SCALAR, subtile_n_eq_1) {
  for (size_t m = 1; m < 4; m++) {
    PReLUMicrokernelTester()
      .m(m)
      .n(1)
      .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PRELU_X4__SCALAR, subtile_n_gt_1) {
  for (size_t m = 1; m < 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU_X4__SCALAR, x_stride) {
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .x_stride(n * 3)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU_X4__SCALAR, y_stride) {
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .y_stride(n * 5)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU_X4__SCALAR, x_stride_and_y_stride) {
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .x_stride(n * 3)
        .y_stride(n * 5)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU_X4__SCALAR, inplace) {
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .inplace(true)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU_X4__SCALAR, qmin) {
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .qmin(128)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU_X4__SCALAR, qmax) {
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 2; n < 16; n += 3) {
      PReLUMicrokernelTester()
        .m(m)
        .n(n)
        .qmax(128)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel_x4__scalar, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}
