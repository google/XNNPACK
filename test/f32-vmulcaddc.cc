// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vmulcaddc.yaml
//   Generator: tools/generate-vmulcaddc-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vmulcaddc.h>
#include "vmulcaddc-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VMULCADDC_C4__NEONFMA_X2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VMulCAddCMicrokernelTester()
      .cr(4)
      .c(4)
      .m(2)
      .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t c = 8; c < 64; c += 12) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t c = 4; c < 8; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t c = 1; c < 4; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, subtile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 1; m < 2; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, multitile) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 3; m < 8; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, x_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .x_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, y_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .y_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEONFMA_X2, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neonfma_x2);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VMULCADDC_C4__NEON_X2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VMulCAddCMicrokernelTester()
      .cr(4)
      .c(4)
      .m(2)
      .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 8; c < 64; c += 12) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 4; c < 8; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t c = 1; c < 4; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 2; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, multitile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 3; m < 8; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .x_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, y_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .y_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__NEON_X2, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__neon_x2);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VMULCADDC_C4__SSE_X2, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VMulCAddCMicrokernelTester()
      .cr(4)
      .c(4)
      .m(2)
      .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t c = 8; c < 64; c += 12) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t c = 4; c < 8; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t c = 1; c < 4; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m < 2; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, multitile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 3; m < 8; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, x_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .x_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, y_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .y_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__SSE_X2, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__sse_x2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  TEST(F32_VMULCADDC_C4__PSIMD_X2, c_eq_4) {
    TEST_REQUIRES_PSIMD;
    VMulCAddCMicrokernelTester()
      .cr(4)
      .c(4)
      .m(2)
      .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, c_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 8; c < 64; c += 12) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, c_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 4; c < 8; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, c_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t c = 1; c < 4; c++) {
      VMulCAddCMicrokernelTester()
        .cr(4)
        .c(c)
        .m(2)
        .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 2; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, multitile) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 3; m < 8; m++) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .x_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .y_stride(23)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_VMULCADDC_C4__PSIMD_X2, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 6; m += 1) {
      for (size_t c = 1; c <= 20; c += 3) {
        VMulCAddCMicrokernelTester()
          .cr(4)
          .c(c)
          .m(m)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_ukernel_c4__psimd_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM


TEST(F32_VMULCADDC_C1__SCALAR_X2, c_eq_1) {
  VMulCAddCMicrokernelTester()
    .cr(1)
    .c(1)
    .m(2)
    .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, c_gt_1) {
  for (size_t c = 1; c < 10; c++) {
    VMulCAddCMicrokernelTester()
      .cr(1)
      .c(c)
      .m(2)
      .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, subtile) {
  for (size_t m = 1; m < 2; m++) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, multitile) {
  for (size_t m = 3; m < 8; m++) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, x_stride) {
  for (size_t m = 1; m < 6; m += 1) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .x_stride(7)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, y_stride) {
  for (size_t m = 1; m < 6; m += 1) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .y_stride(7)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, inplace) {
  for (size_t m = 1; m < 6; m += 1) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .inplace(true)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, qmin) {
  for (size_t m = 1; m < 6; m += 1) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .qmin(128)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_VMULCADDC_C1__SCALAR_X2, qmax) {
  for (size_t m = 1; m < 6; m += 1) {
    for (size_t c = 1; c <= 5; c += 1) {
      VMulCAddCMicrokernelTester()
        .cr(1)
        .c(c)
        .m(m)
        .qmax(128)
        .Test(xnn_f32_vmulcaddc_ukernel_c1__scalar_x2, VMulCAddCMicrokernelTester::Variant::Scalar);
    }
  }
}