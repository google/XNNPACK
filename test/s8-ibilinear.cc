// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s8-ibilinear.yaml
//   Generator: tools/generate-ibilinear-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/isa-checks.h"
#include "ibilinear-microkernel-tester.h"


TEST(S8_IBILINEAR__SCALAR_C1, channels_eq_1) {
  IBilinearMicrokernelTester()
    .pixels(1)
    .channels(1)
    .Test(xnn_s8_ibilinear_ukernel__scalar_c1);
}

TEST(S8_IBILINEAR__SCALAR_C1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c1);
  }
}

TEST(S8_IBILINEAR__SCALAR_C1, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c1);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C1, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(7)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c1);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C1, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(7)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c1);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C2, channels_eq_2) {
  IBilinearMicrokernelTester()
    .pixels(1)
    .channels(2)
    .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
}

TEST(S8_IBILINEAR__SCALAR_C2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
  }
}

TEST(S8_IBILINEAR__SCALAR_C2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
  }
}

TEST(S8_IBILINEAR__SCALAR_C2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
  }
}

TEST(S8_IBILINEAR__SCALAR_C2, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C2, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(13)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C2, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(13)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c2);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C4, channels_eq_4) {
  IBilinearMicrokernelTester()
    .pixels(1)
    .channels(4)
    .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
}

TEST(S8_IBILINEAR__SCALAR_C4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
  }
}

TEST(S8_IBILINEAR__SCALAR_C4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
  }
}

TEST(S8_IBILINEAR__SCALAR_C4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(channels)
      .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
  }
}

TEST(S8_IBILINEAR__SCALAR_C4, pixels_gt_1) {
  for (size_t pixels = 2; pixels < 3; pixels++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C4, input_offset) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
    }
  }
}

TEST(S8_IBILINEAR__SCALAR_C4, output_stride) {
  for (size_t pixels = 1; pixels < 5; pixels += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(23)
        .Test(xnn_s8_ibilinear_ukernel__scalar_c4);
    }
  }
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S8_IBILINEAR__NEON_C8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_s8_ibilinear_ukernel__neon_c8);
  }

  TEST(S8_IBILINEAR__NEON_C8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__neon_c8);
    }
  }

  TEST(S8_IBILINEAR__NEON_C8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__neon_c8);
    }
  }

  TEST(S8_IBILINEAR__NEON_C8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__neon_c8);
    }
  }

  TEST(S8_IBILINEAR__NEON_C8, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__neon_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__NEON_C8, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_s8_ibilinear_ukernel__neon_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__NEON_C8, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_s8_ibilinear_ukernel__neon_c8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S8_IBILINEAR__NEON_C16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_s8_ibilinear_ukernel__neon_c16);
  }

  TEST(S8_IBILINEAR__NEON_C16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__neon_c16);
    }
  }

  TEST(S8_IBILINEAR__NEON_C16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__neon_c16);
    }
  }

  TEST(S8_IBILINEAR__NEON_C16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__neon_c16);
    }
  }

  TEST(S8_IBILINEAR__NEON_C16, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__neon_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__NEON_C16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_s8_ibilinear_ukernel__neon_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__NEON_C16, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_s8_ibilinear_ukernel__neon_c16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S8_IBILINEAR__SSE2_C8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
  }

  TEST(S8_IBILINEAR__SSE2_C8, channels_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
    }
  }

  TEST(S8_IBILINEAR__SSE2_C8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
    }
  }

  TEST(S8_IBILINEAR__SSE2_C8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
    }
  }

  TEST(S8_IBILINEAR__SSE2_C8, pixels_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE2_C8, input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE2_C8, output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_s8_ibilinear_ukernel__sse2_c8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S8_IBILINEAR__SSE2_C16, channels_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
  }

  TEST(S8_IBILINEAR__SSE2_C16, channels_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
    }
  }

  TEST(S8_IBILINEAR__SSE2_C16, channels_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
    }
  }

  TEST(S8_IBILINEAR__SSE2_C16, channels_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
    }
  }

  TEST(S8_IBILINEAR__SSE2_C16, pixels_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE2_C16, input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE2_C16, output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_s8_ibilinear_ukernel__sse2_c16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S8_IBILINEAR__SSE41_C8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
  }

  TEST(S8_IBILINEAR__SSE41_C8, channels_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
    }
  }

  TEST(S8_IBILINEAR__SSE41_C8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
    }
  }

  TEST(S8_IBILINEAR__SSE41_C8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
    }
  }

  TEST(S8_IBILINEAR__SSE41_C8, pixels_gt_1) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE41_C8, input_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE41_C8, output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_s8_ibilinear_ukernel__sse41_c8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S8_IBILINEAR__SSE41_C16, channels_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
  }

  TEST(S8_IBILINEAR__SSE41_C16, channels_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
    }
  }

  TEST(S8_IBILINEAR__SSE41_C16, channels_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
    }
  }

  TEST(S8_IBILINEAR__SSE41_C16, channels_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
    }
  }

  TEST(S8_IBILINEAR__SSE41_C16, pixels_gt_1) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE41_C16, input_offset) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__SSE41_C16, output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_s8_ibilinear_ukernel__sse41_c16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, channels_eq_8) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, pixels_gt_1) {
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, input_offset) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C8, output_stride) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, channels_eq_16) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, pixels_gt_1) {
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, input_offset) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_DOT16X2_C16, output_stride) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, channels_eq_8) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, pixels_gt_1) {
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, input_offset) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C8, output_stride) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, channels_eq_16) {
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, pixels_gt_1) {
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, input_offset) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
      }
    }
  }

  TEST(S8_IBILINEAR__WASMSIMD_MUL32_C16, output_stride) {
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
