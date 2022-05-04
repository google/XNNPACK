// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vmulcaddc-minmax.yaml
//   Generator: tools/generate-vmulcaddc-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vmulcaddc.h>
#include "vmulcaddc-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEON_2X, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, channels_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, rows_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, input_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__NEONFMA_2X, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEON_2X, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, channels_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, rows_div_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, input_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__NEONFMA_2X, qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, channels_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, channels_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, channels_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, channels_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, rows_lt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, rows_div_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, rows_gt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__SSE_2X, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, channels_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, channels_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, channels_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, channels_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, rows_lt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, rows_div_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, rows_gt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__SSE_2X, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, channels_eq_4) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_ARM_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, channels_eq_4) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMSIMD_X86_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, channels_eq_8) {
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_ARM_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, channels_eq_8) {
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMSIMD_X86_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, channels_eq_4) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_FMA_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, channels_eq_4) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASMRELAXEDSIMD_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, channels_eq_8) {
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_FMA_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, channels_eq_8) {
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C8__WASMRELAXEDSIMD_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, channels_eq_1) {
    VMulCAddCMicrokernelTester()
      .channel_tile(1)
      .channels(1)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, channels_gt_1) {
    for (size_t channels = 2; channels < 10; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .input_stride(7)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .output_stride(7)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C1__WASM_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(1)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, channels_eq_2) {
    VMulCAddCMicrokernelTester()
      .channel_tile(2)
      .channels(2)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, channels_div_2) {
    for (size_t channels = 4; channels < 20; channels += 2) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, channels_lt_2) {
    for (size_t channels = 1; channels < 2; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, channels_gt_2) {
    for (size_t channels = 3; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .input_stride(13)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .output_stride(13)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C2__WASM_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 10; channels += 1) {
        VMulCAddCMicrokernelTester()
          .channel_tile(2)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, channels_eq_4) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(4)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(2)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .input_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .output_stride(23)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VMULCADDC_MINMAX_C4__WASM_2X, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        VMulCAddCMicrokernelTester()
          .channel_tile(4)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, channels_eq_1) {
  VMulCAddCMicrokernelTester()
    .channel_tile(1)
    .channels(1)
    .rows(2)
    .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    VMulCAddCMicrokernelTester()
      .channel_tile(1)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .input_stride(7)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .output_stride(7)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .inplace(true)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, qmin) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .qmin(128)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C1__SCALAR_2X, qmax) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(1)
        .channels(channels)
        .rows(rows)
        .qmax(128)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, channels_eq_2) {
  VMulCAddCMicrokernelTester()
    .channel_tile(2)
    .channels(2)
    .rows(2)
    .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    VMulCAddCMicrokernelTester()
      .channel_tile(2)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    VMulCAddCMicrokernelTester()
      .channel_tile(2)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    VMulCAddCMicrokernelTester()
      .channel_tile(2)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .input_stride(13)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .output_stride(13)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .inplace(true)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, qmin) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .qmin(128)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C2__SCALAR_2X, qmax) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      VMulCAddCMicrokernelTester()
        .channel_tile(2)
        .channels(channels)
        .rows(rows)
        .qmax(128)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, channels_eq_4) {
  VMulCAddCMicrokernelTester()
    .channel_tile(4)
    .channels(4)
    .rows(2)
    .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    VMulCAddCMicrokernelTester()
      .channel_tile(4)
      .channels(channels)
      .rows(2)
      .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .input_stride(23)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .output_stride(23)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .inplace(true)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, qmin) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .qmin(128)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VMULCADDC_MINMAX_C4__SCALAR_2X, qmax) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      VMulCAddCMicrokernelTester()
        .channel_tile(4)
        .channels(channels)
        .rows(rows)
        .qmax(128)
        .Test(xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, xnn_init_f32_minmax_scalar_params);
    }
  }
}