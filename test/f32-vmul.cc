// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <xnnpack/isa-checks.h>
#include <xnnpack/vmul.h>

#include "vmul-microkernel-tester.h"


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_VMUL__SSE, n_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VMulMicrokernelTester()
      .n(4)
      .Test(xnn_f32_vmul_ukernel__sse);
  }

  TEST(F32_VMUL__SSE, n_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VMulMicrokernelTester()
      .n(8)
      .Test(xnn_f32_vmul_ukernel__sse);
  }

  TEST(F32_VMUL__SSE, n_eq_12) {
    TEST_REQUIRES_X86_SSE;
    VMulMicrokernelTester()
      .n(12)
      .Test(xnn_f32_vmul_ukernel__sse);
  }

  TEST(F32_VMUL__SSE, n_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 8; n < 128; n += 24) {
      VMulMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, n_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 9; n < 16; n++) {
      VMulMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, n_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 8; n++) {
      VMulMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_a(true)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_b(true)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }

  TEST(F32_VMUL__SSE, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_vmul_ukernel__sse);
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64


#if !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS
  TEST(F32_VMUL__PSIMD, n_eq_4) {
    TEST_REQUIRES_PSIMD;
    VMulMicrokernelTester()
      .n(4)
      .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VMUL__PSIMD, n_eq_8) {
    TEST_REQUIRES_PSIMD;
    VMulMicrokernelTester()
      .n(8)
      .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VMUL__PSIMD, n_eq_12) {
    TEST_REQUIRES_PSIMD;
    VMulMicrokernelTester()
      .n(12)
      .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VMUL__PSIMD, n_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 8; n < 128; n += 24) {
      VMulMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, n_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 9; n < 16; n++) {
      VMulMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, n_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 8; n++) {
      VMulMicrokernelTester()
        .n(n)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, inplace_a) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_a(true)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, inplace_b) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_b(true)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, inplace_a_and_b) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMUL__PSIMD, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 128; n += 11) {
      VMulMicrokernelTester()
        .iterations(1)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_vmul_ukernel__psimd, VMulMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS


TEST(F32_VMUL__SCALAR, n_eq_1) {
  VMulMicrokernelTester()
    .n(1)
    .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
}

TEST(F32_VMUL__SCALAR, n_eq_2) {
  VMulMicrokernelTester()
    .n(2)
    .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
}

TEST(F32_VMUL__SCALAR, n_eq_3) {
  VMulMicrokernelTester()
    .n(3)
    .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
}

TEST(F32_VMUL__SCALAR, n_div_2) {
  for (size_t n = 2; n < 16; n += 6) {
    VMulMicrokernelTester()
      .n(n)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, n_gt_2) {
  for (size_t n = 3; n < 4; n++) {
    VMulMicrokernelTester()
      .n(n)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, n_lt_2) {
  for (size_t n = 1; n < 2; n++) {
    VMulMicrokernelTester()
      .n(n)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, inplace_a) {
  for (size_t n = 1; n < 16; n += 3) {
    VMulMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace_a(true)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, inplace_b) {
  for (size_t n = 1; n < 16; n += 3) {
    VMulMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace_b(true)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, inplace_a_and_b) {
  for (size_t n = 1; n < 16; n += 3) {
    VMulMicrokernelTester()
      .iterations(1)
      .n(n)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, qmin) {
  for (size_t n = 1; n < 16; n += 3) {
    VMulMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmin(128)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMUL__SCALAR, qmax) {
  for (size_t n = 1; n < 16; n += 3) {
    VMulMicrokernelTester()
      .iterations(1)
      .n(n)
      .qmax(128)
      .Test(xnn_f32_vmul_ukernel__scalar, VMulMicrokernelTester::Variant::Scalar);
  }
}
