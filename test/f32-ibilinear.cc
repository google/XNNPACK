// clang-format off
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-ibilinear.yaml
//   Generator: tools/generate-ibilinear-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/isa-checks.h"
#include "test/ibilinear-microkernel-tester.h"


TEST(F32_IBILINEAR__SCALAR_U1, channels_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  IBilinearMicrokernelTester()
    .pixels(1)
    .channels(1)
    .Test(xnn_f32_ibilinear_ukernel__scalar_u1);
}

TEST(F32_IBILINEAR__SCALAR_U1, channels_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 2; channels < 10; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u1);
  }
}

TEST(F32_IBILINEAR__SCALAR_U1, pixels_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u1);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U1, input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(7)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u1);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U1, output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(7)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u1);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U2, channels_eq_2) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  IBilinearMicrokernelTester()
    .pixels(1)
    .channels(2)
    .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
}

TEST(F32_IBILINEAR__SCALAR_U2, channels_div_2) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 4; channels < 20; channels += 2) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
  }
}

TEST(F32_IBILINEAR__SCALAR_U2, channels_lt_2) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 1; channels < 2; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
  }
}

TEST(F32_IBILINEAR__SCALAR_U2, channels_gt_2) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 3; channels < 4; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
  }
}

TEST(F32_IBILINEAR__SCALAR_U2, pixels_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U2, input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(13)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U2, output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(13)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u2);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U4, channels_eq_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  IBilinearMicrokernelTester()
    .pixels(1)
    .channels(4)
    .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
}

TEST(F32_IBILINEAR__SCALAR_U4, channels_div_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 8; channels < 40; channels += 4) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
  }
}

TEST(F32_IBILINEAR__SCALAR_U4, channels_lt_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 1; channels < 4; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
  }
}

TEST(F32_IBILINEAR__SCALAR_U4, channels_gt_4) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t channels = 5; channels < 8; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
  }
}

TEST(F32_IBILINEAR__SCALAR_U4, pixels_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U4, input_offset) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
    }
  }
}

TEST(F32_IBILINEAR__SCALAR_U4, output_stride) {
  TEST_REQUIRES_ARCH_FLAGS(0);
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(23)
        .Test(xnn_f32_ibilinear_ukernel__scalar_u4);
    }
  }
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_IBILINEAR__NEON_U4, channels_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_ibilinear_ukernel__neon_u4);
  }

  TEST(F32_IBILINEAR__NEON_U4, channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t channels = 8; channels < 40; channels += 4) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neon_u4);
    }
  }

  TEST(F32_IBILINEAR__NEON_U4, channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t channels = 1; channels < 4; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neon_u4);
    }
  }

  TEST(F32_IBILINEAR__NEON_U4, channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t channels = 5; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neon_u4);
    }
  }

  TEST(F32_IBILINEAR__NEON_U4, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__neon_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__NEON_U4, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_ibilinear_ukernel__neon_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__NEON_U4, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_ibilinear_ukernel__neon_u4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_IBILINEAR__NEON_U8, channels_eq_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_ibilinear_ukernel__neon_u8);
  }

  TEST(F32_IBILINEAR__NEON_U8, channels_div_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neon_u8);
    }
  }

  TEST(F32_IBILINEAR__NEON_U8, channels_lt_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neon_u8);
    }
  }

  TEST(F32_IBILINEAR__NEON_U8, channels_gt_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neon_u8);
    }
  }

  TEST(F32_IBILINEAR__NEON_U8, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__neon_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__NEON_U8, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_ibilinear_ukernel__neon_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__NEON_U8, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_ibilinear_ukernel__neon_u8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_IBILINEAR__NEONFMA_U4, channels_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
  }

  TEST(F32_IBILINEAR__NEONFMA_U4, channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t channels = 8; channels < 40; channels += 4) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U4, channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t channels = 1; channels < 4; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U4, channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t channels = 5; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U4, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U4, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U4, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_ibilinear_ukernel__neonfma_u4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_IBILINEAR__NEONFMA_U8, channels_eq_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
  }

  TEST(F32_IBILINEAR__NEONFMA_U8, channels_div_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U8, channels_lt_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U8, channels_gt_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U8, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U8, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__NEONFMA_U8, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_neon_fma);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_ibilinear_ukernel__neonfma_u8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_SSE && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_IBILINEAR__SSE_U4, channels_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_ibilinear_ukernel__sse_u4);
  }

  TEST(F32_IBILINEAR__SSE_U4, channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t channels = 8; channels < 40; channels += 4) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__sse_u4);
    }
  }

  TEST(F32_IBILINEAR__SSE_U4, channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t channels = 1; channels < 4; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__sse_u4);
    }
  }

  TEST(F32_IBILINEAR__SSE_U4, channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t channels = 5; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__sse_u4);
    }
  }

  TEST(F32_IBILINEAR__SSE_U4, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__sse_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__SSE_U4, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_ibilinear_ukernel__sse_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__SSE_U4, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_ibilinear_ukernel__sse_u4);
      }
    }
  }
#endif  // XNN_ENABLE_SSE && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_SSE && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_IBILINEAR__SSE_U8, channels_eq_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_ibilinear_ukernel__sse_u8);
  }

  TEST(F32_IBILINEAR__SSE_U8, channels_div_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__sse_u8);
    }
  }

  TEST(F32_IBILINEAR__SSE_U8, channels_lt_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__sse_u8);
    }
  }

  TEST(F32_IBILINEAR__SSE_U8, channels_gt_8) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__sse_u8);
    }
  }

  TEST(F32_IBILINEAR__SSE_U8, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__sse_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__SSE_U8, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_ibilinear_ukernel__sse_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__SSE_U8, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_ibilinear_ukernel__sse_u8);
      }
    }
  }
#endif  // XNN_ENABLE_SSE && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_IBILINEAR__WASMSIMD_U4, channels_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
  }

  TEST(F32_IBILINEAR__WASMSIMD_U4, channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 8; channels < 40; channels += 4) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U4, channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 1; channels < 4; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U4, channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 5; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U4, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U4, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U4, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_IBILINEAR__WASMSIMD_U8, channels_eq_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
  }

  TEST(F32_IBILINEAR__WASMSIMD_U8, channels_div_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U8, channels_lt_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U8, channels_gt_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U8, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U8, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMSIMD_U8, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_ibilinear_ukernel__wasmsimd_u8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, channels_eq_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, channels_div_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 8; channels < 40; channels += 4) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, channels_lt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 1; channels < 4; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, channels_gt_4) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 5; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U4, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, channels_eq_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, channels_div_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, channels_lt_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, channels_gt_8) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
      }
    }
  }

  TEST(F32_IBILINEAR__WASMRELAXEDSIMD_U8, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(0);
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_u8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_IBILINEAR__RVV_U1V, channels_eq_1v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels((1 * xnn_init_hardware_config()->vlenb / sizeof(float)))
      .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
  }

  TEST(F32_IBILINEAR__RVV_U1V, channels_div_1v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (1 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = channel_tile * 2; channels < channel_tile * 10; channels += channel_tile) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U1V, channels_lt_1v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (1 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = 1; channels < channel_tile; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U1V, channels_gt_1v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (1 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = channel_tile+1; channels < channel_tile * 2; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U1V, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (1 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
      }
    }
  }

  TEST(F32_IBILINEAR__RVV_U1V, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (1 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(channel_tile * 5 + 1)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
      }
    }
  }

  TEST(F32_IBILINEAR__RVV_U1V, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (1 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(channel_tile * 5 + 1)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u1v);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_IBILINEAR__RVV_U2V, channels_eq_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels((2 * xnn_init_hardware_config()->vlenb / sizeof(float)))
      .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
  }

  TEST(F32_IBILINEAR__RVV_U2V, channels_div_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (2 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = channel_tile * 2; channels < channel_tile * 10; channels += channel_tile) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U2V, channels_lt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (2 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = 1; channels < channel_tile; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U2V, channels_gt_2v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (2 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = channel_tile+1; channels < channel_tile * 2; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U2V, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (2 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
      }
    }
  }

  TEST(F32_IBILINEAR__RVV_U2V, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (2 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(channel_tile * 5 + 1)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
      }
    }
  }

  TEST(F32_IBILINEAR__RVV_U2V, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (2 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(channel_tile * 5 + 1)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u2v);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_IBILINEAR__RVV_U4V, channels_eq_4v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels((4 * xnn_init_hardware_config()->vlenb / sizeof(float)))
      .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
  }

  TEST(F32_IBILINEAR__RVV_U4V, channels_div_4v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (4 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = channel_tile * 2; channels < channel_tile * 10; channels += channel_tile) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U4V, channels_lt_4v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (4 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = 1; channels < channel_tile; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U4V, channels_gt_4v) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (4 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t channels = channel_tile+1; channels < channel_tile * 2; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
    }
  }

  TEST(F32_IBILINEAR__RVV_U4V, pixels_gt_1) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (4 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
      }
    }
  }

  TEST(F32_IBILINEAR__RVV_U4V, input_offset) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (4 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(channel_tile * 5 + 1)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
      }
    }
  }

  TEST(F32_IBILINEAR__RVV_U4V, output_stride) {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector);
    const size_t channel_tile = (4 * xnn_init_hardware_config()->vlenb / sizeof(float));
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= channel_tile * 5; channels += channel_tile - 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(channel_tile * 5 + 1)
          .Test(xnn_f32_ibilinear_ukernel__rvv_u4v);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
