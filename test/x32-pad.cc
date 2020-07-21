// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/pad.h>
#include "pad-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PAD__NEON, fulltile_copy_channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PadMicrokernelTester()
      .rows(1)
      .input_channels(4)
      .Test(xnn_x32_pad_ukernel__neon);
  }

  TEST(X32_PAD__NEON, fulltile_copy_channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_copy_channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_copy_channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(4)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_pre_padding_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(4)
      .Test(xnn_x32_pad_ukernel__neon);
  }

  TEST(X32_PAD__NEON, fulltile_pre_padding_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pre_padding = 8; pre_padding < 32; pre_padding += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_pre_padding_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pre_padding = 1; pre_padding < 4; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_pre_padding_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pre_padding = 5; pre_padding < 8; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_post_padding_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .post_padding(4)
      .Test(xnn_x32_pad_ukernel__neon);
  }

  TEST(X32_PAD__NEON, fulltile_post_padding_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t post_padding = 8; post_padding < 32; post_padding += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_post_padding_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t post_padding = 1; post_padding < 4; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, fulltile_post_padding_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t post_padding = 5; post_padding < 8; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__neon);
    }
  }

  TEST(X32_PAD__NEON, multitile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .Test(xnn_x32_pad_ukernel__neon);
      }
    }
  }

  TEST(X32_PAD__NEON, multitile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .input_stride(2 * channels + 1)
          .Test(xnn_x32_pad_ukernel__neon);
      }
    }
  }

  TEST(X32_PAD__NEON, multitile_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(2 * channels)
          .pre_padding(channels)
          .post_padding(channels)
          .output_stride(5 * channels + 3)
          .Test(xnn_x32_pad_ukernel__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PAD__SSE, fulltile_copy_channels_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PadMicrokernelTester()
      .rows(1)
      .input_channels(4)
      .Test(xnn_x32_pad_ukernel__sse);
  }

  TEST(X32_PAD__SSE, fulltile_copy_channels_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_copy_channels_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_copy_channels_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(4)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_pre_padding_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(4)
      .Test(xnn_x32_pad_ukernel__sse);
  }

  TEST(X32_PAD__SSE, fulltile_pre_padding_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pre_padding = 8; pre_padding < 32; pre_padding += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_pre_padding_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pre_padding = 1; pre_padding < 4; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_pre_padding_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pre_padding = 5; pre_padding < 8; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_post_padding_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .post_padding(4)
      .Test(xnn_x32_pad_ukernel__sse);
  }

  TEST(X32_PAD__SSE, fulltile_post_padding_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t post_padding = 8; post_padding < 32; post_padding += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_post_padding_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t post_padding = 1; post_padding < 4; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, fulltile_post_padding_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t post_padding = 5; post_padding < 8; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__sse);
    }
  }

  TEST(X32_PAD__SSE, multitile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .Test(xnn_x32_pad_ukernel__sse);
      }
    }
  }

  TEST(X32_PAD__SSE, multitile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .input_stride(2 * channels + 1)
          .Test(xnn_x32_pad_ukernel__sse);
      }
    }
  }

  TEST(X32_PAD__SSE, multitile_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(2 * channels)
          .pre_padding(channels)
          .post_padding(channels)
          .output_stride(5 * channels + 3)
          .Test(xnn_x32_pad_ukernel__sse);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(X32_PAD__WASMSIMD, fulltile_copy_channels_eq_4) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(4)
      .Test(xnn_x32_pad_ukernel__wasmsimd);
  }

  TEST(X32_PAD__WASMSIMD, fulltile_copy_channels_div_4) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_copy_channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_copy_channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(4)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_pre_padding_eq_4) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(4)
      .Test(xnn_x32_pad_ukernel__wasmsimd);
  }

  TEST(X32_PAD__WASMSIMD, fulltile_pre_padding_div_4) {
    for (size_t pre_padding = 8; pre_padding < 32; pre_padding += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_pre_padding_lt_4) {
    for (size_t pre_padding = 1; pre_padding < 4; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_pre_padding_gt_4) {
    for (size_t pre_padding = 5; pre_padding < 8; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_post_padding_eq_4) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .post_padding(4)
      .Test(xnn_x32_pad_ukernel__wasmsimd);
  }

  TEST(X32_PAD__WASMSIMD, fulltile_post_padding_div_4) {
    for (size_t post_padding = 8; post_padding < 32; post_padding += 4) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_post_padding_lt_4) {
    for (size_t post_padding = 1; post_padding < 4; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, fulltile_post_padding_gt_4) {
    for (size_t post_padding = 5; post_padding < 8; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(post_padding)
        .Test(xnn_x32_pad_ukernel__wasmsimd);
    }
  }

  TEST(X32_PAD__WASMSIMD, multitile) {
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .Test(xnn_x32_pad_ukernel__wasmsimd);
      }
    }
  }

  TEST(X32_PAD__WASMSIMD, multitile_with_input_stride) {
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(channels)
          .pre_padding(channels)
          .post_padding(channels)
          .input_stride(2 * channels + 1)
          .Test(xnn_x32_pad_ukernel__wasmsimd);
      }
    }
  }

  TEST(X32_PAD__WASMSIMD, multitile_with_output_stride) {
    for (size_t rows = 2; rows <= 5; rows++) {
      for (size_t channels = 1; channels < 10; channels++) {
        PadMicrokernelTester()
          .rows(rows)
          .input_channels(2 * channels)
          .pre_padding(channels)
          .post_padding(channels)
          .output_stride(5 * channels + 3)
          .Test(xnn_x32_pad_ukernel__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


TEST(X32_PAD__SCALAR_INT, fulltile_copy_channels_eq_1) {
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .Test(xnn_x32_pad_ukernel__scalar_int);
}

TEST(X32_PAD__SCALAR_INT, fulltile_copy_channels_gt_1) {
  for (size_t channels = 2; channels < 8; channels++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(channels)
      .Test(xnn_x32_pad_ukernel__scalar_int);
  }
}

TEST(X32_PAD__SCALAR_INT, fulltile_pre_padding_eq_1) {
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .pre_padding(1)
    .Test(xnn_x32_pad_ukernel__scalar_int);
}

TEST(X32_PAD__SCALAR_INT, fulltile_pre_padding_gt_1) {
  for (size_t pre_padding = 2; pre_padding < 8; pre_padding++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(pre_padding)
      .Test(xnn_x32_pad_ukernel__scalar_int);
  }
}

TEST(X32_PAD__SCALAR_INT, fulltile_post_padding_eq_1) {
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .post_padding(1)
    .Test(xnn_x32_pad_ukernel__scalar_int);
}

TEST(X32_PAD__SCALAR_INT, fulltile_post_padding_gt_1) {
  for (size_t post_padding = 1; post_padding < 8; post_padding++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(post_padding)
      .Test(xnn_x32_pad_ukernel__scalar_int);
  }
}

TEST(X32_PAD__SCALAR_INT, multitile) {
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < 10; channels++) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(channels)
        .pre_padding(channels)
        .post_padding(channels)
        .Test(xnn_x32_pad_ukernel__scalar_int);
    }
  }
}

TEST(X32_PAD__SCALAR_INT, multitile_with_input_stride) {
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < 10; channels++) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(channels)
        .pre_padding(channels)
        .post_padding(channels)
        .input_stride(2 * channels + 1)
        .Test(xnn_x32_pad_ukernel__scalar_int);
    }
  }
}

TEST(X32_PAD__SCALAR_INT, multitile_with_output_stride) {
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < 10; channels++) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(2 * channels)
        .pre_padding(channels)
        .post_padding(channels)
        .output_stride(5 * channels + 3)
        .Test(xnn_x32_pad_ukernel__scalar_int);
    }
  }
}


TEST(X32_PAD__SCALAR_FLOAT, fulltile_copy_channels_eq_1) {
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .Test(xnn_x32_pad_ukernel__scalar_float);
}

TEST(X32_PAD__SCALAR_FLOAT, fulltile_copy_channels_gt_1) {
  for (size_t channels = 2; channels < 8; channels++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(channels)
      .Test(xnn_x32_pad_ukernel__scalar_float);
  }
}

TEST(X32_PAD__SCALAR_FLOAT, fulltile_pre_padding_eq_1) {
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .pre_padding(1)
    .Test(xnn_x32_pad_ukernel__scalar_float);
}

TEST(X32_PAD__SCALAR_FLOAT, fulltile_pre_padding_gt_1) {
  for (size_t pre_padding = 2; pre_padding < 8; pre_padding++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(pre_padding)
      .Test(xnn_x32_pad_ukernel__scalar_float);
  }
}

TEST(X32_PAD__SCALAR_FLOAT, fulltile_post_padding_eq_1) {
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .post_padding(1)
    .Test(xnn_x32_pad_ukernel__scalar_float);
}

TEST(X32_PAD__SCALAR_FLOAT, fulltile_post_padding_gt_1) {
  for (size_t post_padding = 1; post_padding < 8; post_padding++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(post_padding)
      .Test(xnn_x32_pad_ukernel__scalar_float);
  }
}

TEST(X32_PAD__SCALAR_FLOAT, multitile) {
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < 10; channels++) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(channels)
        .pre_padding(channels)
        .post_padding(channels)
        .Test(xnn_x32_pad_ukernel__scalar_float);
    }
  }
}

TEST(X32_PAD__SCALAR_FLOAT, multitile_with_input_stride) {
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < 10; channels++) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(channels)
        .pre_padding(channels)
        .post_padding(channels)
        .input_stride(2 * channels + 1)
        .Test(xnn_x32_pad_ukernel__scalar_float);
    }
  }
}

TEST(X32_PAD__SCALAR_FLOAT, multitile_with_output_stride) {
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < 10; channels++) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(2 * channels)
        .pre_padding(channels)
        .post_padding(channels)
        .output_stride(5 * channels + 3)
        .Test(xnn_x32_pad_ukernel__scalar_float);
    }
  }
}
