// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vsub.h>
#include "vsub-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB__SSE, n_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VSubMicrokernelTester()
      .n(4)
      .Test(xnn_f32_vsub_ukernel__sse);
  }

  TEST(F32_VSUB__SSE, n_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VSubMicrokernelTester()
      .n(8)
      .Test(xnn_f32_vsub_ukernel__sse);
  }

  TEST(F32_VSUB__SSE, n_eq_12) {
    TEST_REQUIRES_X86_SSE;
    VSubMicrokernelTester()
      .n(12)
      .Test(xnn_f32_vsub_ukernel__sse);
  }

  TEST(F32_VSUB__SSE, n_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 8; n < 128; n += 24) {
      VSubMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }

  TEST(F32_VSUB__SSE, n_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 9; n < 16; n++) {
      VSubMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }

  TEST(F32_VSUB__SSE, n_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 8; n++) {
      VSubMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }

  TEST(F32_VSUB__SSE, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_a(true)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }

  TEST(F32_VSUB__SSE, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_b(true)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }

  TEST(F32_VSUB__SSE, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }

  TEST(F32_VSUB__SSE, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_vsub_ukernel__sse);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  TEST(F32_VSUB__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    VSubMicrokernelTester()
      .n(4)
      .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VSUB__PSIMD, n_eq_8) {
    TEST_REQUIRES_PSIMD;
    VSubMicrokernelTester()
      .n(8)
      .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VSUB__PSIMD, n_eq_12) {
    TEST_REQUIRES_PSIMD;
    VSubMicrokernelTester()
      .n(12)
      .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VSUB__PSIMD, n_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 8; n < 128; n += 24) {
      VSubMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VSUB__PSIMD, n_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 9; n < 16; n++) {
      VSubMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VSUB__PSIMD, n_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 8; n++) {
      VSubMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VSUB__PSIMD, inplace_a) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_a(true)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VSUB__PSIMD, inplace_b) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_b(true)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VSUB__PSIMD, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VSUB__PSIMD, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VSubMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_vsub_ukernel__psimd, VSubMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS


TEST(F32_VSUB__SCALAR, n_eq_1) {
  VSubMicrokernelTester()
    .n(1)
    .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
}

TEST(F32_VSUB__SCALAR, n_eq_2) {
  VSubMicrokernelTester()
    .n(2)
    .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
}

TEST(F32_VSUB__SCALAR, n_eq_3) {
  VSubMicrokernelTester()
    .n(3)
    .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
}

TEST(F32_VSUB__SCALAR, n_div_2) {
  for (size_t n = 2; n < 16; n += 6) {
    VSubMicrokernelTester()
      .n(n)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSUB__SCALAR, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    VSubMicrokernelTester()
      .n(n)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSUB__SCALAR, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    VSubMicrokernelTester()
      .n(n)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSUB__SCALAR, inplace_a) {
  for (size_t n = 1; n < 16; n += 3) {
    VSubMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace_a(true)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSUB__SCALAR, inplace_b) {
  for (size_t n = 1; n < 16; n += 3) {
    VSubMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace_b(true)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSUB__SCALAR, qmin) {
  for (size_t n = 1; n < 16; n += 3) {
    VSubMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmin(128)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VSUB__SCALAR, qmax) {
  for (size_t n = 1; n < 16; n += 3) {
    VSubMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmax(128)
      .Test(xnn_f32_vsub_ukernel__scalar, VSubMicrokernelTester::Variant::Scalar);
  }
}
