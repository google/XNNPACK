// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-bilinear.yaml
//   Generator: tools/generate-bilinear-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/bilinear.h>
#include "bilinear-microkernel-tester.h"


TEST(F32_BILINEAR__SCALAR_C1, channels_eq_1) {
  BilinearMicrokernelTester()
    .pixels(1)
    .channels(1)
    .Test(xnn_f32_bilinear_ukernel__scalar_c1);
}

TEST(F32_BILINEAR__SCALAR_C1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c1);
  }
}

TEST(F32_BILINEAR__SCALAR_C1, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__scalar_c1);
    }
  }
}

TEST(F32_BILINEAR__SCALAR_C1, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(7)
        .Test(xnn_f32_bilinear_ukernel__scalar_c1);
    }
  }
}
TEST(F32_BILINEAR__SCALAR_C1, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(7)
        .Test(xnn_f32_bilinear_ukernel__scalar_c1);
    }
  }
}

TEST(F32_BILINEAR__SCALAR_C2, channels_eq_2) {
  BilinearMicrokernelTester()
    .pixels(1)
    .channels(2)
    .Test(xnn_f32_bilinear_ukernel__scalar_c2);
}

TEST(F32_BILINEAR__SCALAR_C2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c2);
  }
}

TEST(F32_BILINEAR__SCALAR_C2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c2);
  }
}

TEST(F32_BILINEAR__SCALAR_C2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c2);
  }
}

TEST(F32_BILINEAR__SCALAR_C2, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__scalar_c2);
    }
  }
}

TEST(F32_BILINEAR__SCALAR_C2, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(13)
        .Test(xnn_f32_bilinear_ukernel__scalar_c2);
    }
  }
}
TEST(F32_BILINEAR__SCALAR_C2, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(13)
        .Test(xnn_f32_bilinear_ukernel__scalar_c2);
    }
  }
}

TEST(F32_BILINEAR__SCALAR_C4, channels_eq_4) {
  BilinearMicrokernelTester()
    .pixels(1)
    .channels(4)
    .Test(xnn_f32_bilinear_ukernel__scalar_c4);
}

TEST(F32_BILINEAR__SCALAR_C4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c4);
  }
}

TEST(F32_BILINEAR__SCALAR_C4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c4);
  }
}

TEST(F32_BILINEAR__SCALAR_C4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_f32_bilinear_ukernel__scalar_c4);
  }
}

TEST(F32_BILINEAR__SCALAR_C4, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__scalar_c4);
    }
  }
}

TEST(F32_BILINEAR__SCALAR_C4, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_bilinear_ukernel__scalar_c4);
    }
  }
}
TEST(F32_BILINEAR__SCALAR_C4, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      BilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(23)
        .Test(xnn_f32_bilinear_ukernel__scalar_c4);
    }
  }
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_BILINEAR__NEON_C4, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_bilinear_ukernel__neon_c4);
  }

  TEST(F32_BILINEAR__NEON_C4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 40; channels += 4) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neon_c4);
    }
  }

  TEST(F32_BILINEAR__NEON_C4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neon_c4);
    }
  }

  TEST(F32_BILINEAR__NEON_C4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neon_c4);
    }
  }

  TEST(F32_BILINEAR__NEON_C4, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__neon_c4);
      }
    }
  }

  TEST(F32_BILINEAR__NEON_C4, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_bilinear_ukernel__neon_c4);
      }
    }
  }
  TEST(F32_BILINEAR__NEON_C4, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_bilinear_ukernel__neon_c4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_BILINEAR__NEON_C8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_bilinear_ukernel__neon_c8);
  }

  TEST(F32_BILINEAR__NEON_C8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neon_c8);
    }
  }

  TEST(F32_BILINEAR__NEON_C8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neon_c8);
    }
  }

  TEST(F32_BILINEAR__NEON_C8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neon_c8);
    }
  }

  TEST(F32_BILINEAR__NEON_C8, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__neon_c8);
      }
    }
  }

  TEST(F32_BILINEAR__NEON_C8, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_bilinear_ukernel__neon_c8);
      }
    }
  }
  TEST(F32_BILINEAR__NEON_C8, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_bilinear_ukernel__neon_c8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_BILINEAR__NEONFMA_C4, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
  }

  TEST(F32_BILINEAR__NEONFMA_C4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 8; channels < 40; channels += 4) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels < 4; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 5; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C4, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
      }
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C4, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
      }
    }
  }
  TEST(F32_BILINEAR__NEONFMA_C4, output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_bilinear_ukernel__neonfma_c4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_BILINEAR__NEONFMA_C8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
  }

  TEST(F32_BILINEAR__NEONFMA_C8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 16; channels < 80; channels += 8) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 9; channels < 16; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C8, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
      }
    }
  }

  TEST(F32_BILINEAR__NEONFMA_C8, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
      }
    }
  }
  TEST(F32_BILINEAR__NEONFMA_C8, output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_bilinear_ukernel__neonfma_c8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_BILINEAR__SSE_C4, channels_eq_4) {
    TEST_REQUIRES_X86_SSE;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_bilinear_ukernel__sse_c4);
  }

  TEST(F32_BILINEAR__SSE_C4, channels_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 40; channels += 4) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__sse_c4);
    }
  }

  TEST(F32_BILINEAR__SSE_C4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__sse_c4);
    }
  }

  TEST(F32_BILINEAR__SSE_C4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__sse_c4);
    }
  }

  TEST(F32_BILINEAR__SSE_C4, pixels_gt_1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__sse_c4);
      }
    }
  }

  TEST(F32_BILINEAR__SSE_C4, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_bilinear_ukernel__sse_c4);
      }
    }
  }
  TEST(F32_BILINEAR__SSE_C4, output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_bilinear_ukernel__sse_c4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_BILINEAR__SSE_C8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_bilinear_ukernel__sse_c8);
  }

  TEST(F32_BILINEAR__SSE_C8, channels_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 16; channels < 80; channels += 8) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__sse_c8);
    }
  }

  TEST(F32_BILINEAR__SSE_C8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__sse_c8);
    }
  }

  TEST(F32_BILINEAR__SSE_C8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 9; channels < 16; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__sse_c8);
    }
  }

  TEST(F32_BILINEAR__SSE_C8, pixels_gt_1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__sse_c8);
      }
    }
  }

  TEST(F32_BILINEAR__SSE_C8, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_bilinear_ukernel__sse_c8);
      }
    }
  }
  TEST(F32_BILINEAR__SSE_C8, output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_bilinear_ukernel__sse_c8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  TEST(F32_BILINEAR__PSIMD_C4, channels_eq_4) {
    TEST_REQUIRES_PSIMD;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(4)
      .Test(xnn_f32_bilinear_ukernel__psimd_c4);
  }

  TEST(F32_BILINEAR__PSIMD_C4, channels_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 40; channels += 4) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__psimd_c4);
    }
  }

  TEST(F32_BILINEAR__PSIMD_C4, channels_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__psimd_c4);
    }
  }

  TEST(F32_BILINEAR__PSIMD_C4, channels_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__psimd_c4);
    }
  }

  TEST(F32_BILINEAR__PSIMD_C4, pixels_gt_1) {
    TEST_REQUIRES_PSIMD;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__psimd_c4);
      }
    }
  }

  TEST(F32_BILINEAR__PSIMD_C4, input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(23)
          .Test(xnn_f32_bilinear_ukernel__psimd_c4);
      }
    }
  }
  TEST(F32_BILINEAR__PSIMD_C4, output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(23)
          .Test(xnn_f32_bilinear_ukernel__psimd_c4);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  TEST(F32_BILINEAR__PSIMD_C8, channels_eq_8) {
    TEST_REQUIRES_PSIMD;
    BilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f32_bilinear_ukernel__psimd_c8);
  }

  TEST(F32_BILINEAR__PSIMD_C8, channels_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 16; channels < 80; channels += 8) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__psimd_c8);
    }
  }

  TEST(F32_BILINEAR__PSIMD_C8, channels_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 8; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__psimd_c8);
    }
  }

  TEST(F32_BILINEAR__PSIMD_C8, channels_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 9; channels < 16; channels++) {
      BilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f32_bilinear_ukernel__psimd_c8);
    }
  }

  TEST(F32_BILINEAR__PSIMD_C8, pixels_gt_1) {
    TEST_REQUIRES_PSIMD;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f32_bilinear_ukernel__psimd_c8);
      }
    }
  }

  TEST(F32_BILINEAR__PSIMD_C8, input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f32_bilinear_ukernel__psimd_c8);
      }
    }
  }
  TEST(F32_BILINEAR__PSIMD_C8, output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        BilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f32_bilinear_ukernel__psimd_c8);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
