// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gavgpool-cw.yaml
//   Generator: tools/generate-gavgpool-cw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "gavgpool-cw-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, channels_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__neon_u4, xnn_init_f32_gavgpool_neon_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__NEON_U4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
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
    const size_t element_tile = 4;
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_gt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t element_tile = 4;
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_lt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, elements_div_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t element_tile = 4;
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, channels_gt_1) {
    TEST_REQUIRES_X86_SSE;
    const size_t element_tile = 4;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, qmin) {
    TEST_REQUIRES_X86_SSE;
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__sse_u4, xnn_init_f32_gavgpool_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__SSE_U4, qmax) {
    TEST_REQUIRES_X86_SSE;
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
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
    const size_t element_tile = 4;
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_gt_4) {
    const size_t element_tile = 4;
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_lt_4) {
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, elements_div_4) {
    const size_t element_tile = 4;
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, channels_gt_1) {
    const size_t element_tile = 4;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, qmin) {
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_arm_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_ARM_U4, qmax) {
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
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
    const size_t element_tile = 4;
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_gt_4) {
    const size_t element_tile = 4;
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_lt_4) {
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, elements_div_4) {
    const size_t element_tile = 4;
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, channels_gt_1) {
    const size_t element_tile = 4;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, qmin) {
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__WASMSIMD_X86_U4, qmax) {
    const size_t element_tile = 4;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_GAVGPOOL_CW__RVV_U1V, elements_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__RVV_U1V, elements_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U1V, elements_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U1V, elements_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U1V, channels_gt_1) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U1V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U1V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (1*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u1v, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_GAVGPOOL_CW__RVV_U2V, elements_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__RVV_U2V, elements_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U2V, elements_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U2V, elements_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U2V, channels_gt_1) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U2V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U2V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (2*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u2v, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_GAVGPOOL_CW__RVV_U4V, elements_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__RVV_U4V, elements_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U4V, elements_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U4V, elements_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U4V, channels_gt_1) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U4V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U4V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (4*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u4v, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_GAVGPOOL_CW__RVV_U8V, elements_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
  }

  TEST(F32_GAVGPOOL_CW__RVV_U8V, elements_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U8V, elements_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U8V, elements_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U8V, channels_gt_1) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U8V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_CW__RVV_U8V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t element_tile = (8*xnn_init_hardware_config()->vlenb/sizeof(float));
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_cw_ukernel__rvv_u8v, xnn_init_f32_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


TEST(F32_GAVGPOOL_CW__SCALAR_U1, elements_eq_1) {
  const size_t element_tile = 1;
  GAvgPoolCWMicrokernelTester()
    .elements(element_tile)
    .channels(1)
    .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, elements_gt_1) {
  const size_t element_tile = 1;
  for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, elements_div_1) {
  const size_t element_tile = 1;
  for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, channels_gt_1) {
  const size_t element_tile = 1;
  for (size_t channels = 2; channels < 4; channels++) {
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(channels)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, qmin) {
  const size_t element_tile = 1;
  for (size_t elements = 1; elements < element_tile*2; elements += 3) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}

TEST(F32_GAVGPOOL_CW__SCALAR_U1, qmax) {
  const size_t element_tile = 1;
  for (size_t elements = 1; elements < element_tile*2; elements += 3) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_cw_ukernel__scalar_u1, xnn_init_f32_gavgpool_scalar_params);
  }
}