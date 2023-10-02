// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gavgpool-cw.yaml
//   Generator: tools/generate-gavgpool-cw-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-cw-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolCWMicrokernelTester()
      .elements(4)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 20; elements += 4) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, channels_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(4)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_eq_4) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolCWMicrokernelTester()
      .elements(4)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 5; elements < 8; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 1; elements < 4; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 8; elements < 20; elements += 4) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, channels_gt_1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(4)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_eq_4) {
    GAvgPoolCWMicrokernelTester()
      .elements(4)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_gt_4) {
    for (size_t elements = 5; elements < 8; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_lt_4) {
    for (size_t elements = 1; elements < 4; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_div_4) {
    for (size_t elements = 8; elements < 20; elements += 4) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, channels_gt_1) {
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(4)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, qmin) {
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, qmax) {
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_eq_4) {
    GAvgPoolCWMicrokernelTester()
      .elements(4)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_gt_4) {
    for (size_t elements = 5; elements < 8; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_lt_4) {
    for (size_t elements = 1; elements < 4; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_div_4) {
    for (size_t elements = 8; elements < 20; elements += 4) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, channels_gt_1) {
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(4)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, qmin) {
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, qmax) {
    for (size_t elements = 1; elements < 8; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_GAVGPOOL_CW__SCALAR_U1, elements_eq_1) {
  GAvgPoolCWMicrokernelTester()
    .elements(1)
    .channels(1)
    .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, elements_gt_1) {
  for (size_t elements = 2; elements < 2; elements++) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, elements_div_1) {
  for (size_t elements = 2; elements < 5; elements += 1) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, channels_gt_1) {
  for (size_t channels = 2; channels < 4; channels++) {
    GAvgPoolCWMicrokernelTester()
      .elements(1)
      .channels(channels)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, qmin) {
  for (size_t elements = 1; elements < 2; elements += 1) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, qmax) {
  for (size_t elements = 1; elements < 2; elements += 1) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}