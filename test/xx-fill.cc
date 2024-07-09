// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/xx-fill.yaml
//   Generator: tools/generate-fill-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/fill.h"
#include "xnnpack/isa-checks.h"
#include "fill-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(XX_FILL__NEON_U64, channels_eq_64) {
    TEST_REQUIRES_ARM_NEON;
    FillMicrokernelTester()
      .channels(64)
      .Test(xnn_xx_fill_ukernel__neon_u64);
  }

  TEST(XX_FILL__NEON_U64, channels_div_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels <= 192; channels += 64) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__neon_u64);
    }
  }

  TEST(XX_FILL__NEON_U64, channels_lt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 64; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__neon_u64);
    }
  }

  TEST(XX_FILL__NEON_U64, channels_gt_64) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__neon_u64);
    }
  }

  TEST(XX_FILL__NEON_U64, multiple_rows) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 192; channels += 15) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .Test(xnn_xx_fill_ukernel__neon_u64);
      }
    }
  }

  TEST(XX_FILL__NEON_U64, multiple_rows_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 192; channels += 15) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(193)
          .Test(xnn_xx_fill_ukernel__neon_u64);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(XX_FILL__SSE2_U64, channels_eq_64) {
    TEST_REQUIRES_X86_SSE2;
    FillMicrokernelTester()
      .channels(64)
      .Test(xnn_xx_fill_ukernel__sse2_u64);
  }

  TEST(XX_FILL__SSE2_U64, channels_div_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 128; channels <= 192; channels += 64) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__sse2_u64);
    }
  }

  TEST(XX_FILL__SSE2_U64, channels_lt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 64; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__sse2_u64);
    }
  }

  TEST(XX_FILL__SSE2_U64, channels_gt_64) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 65; channels < 128; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__sse2_u64);
    }
  }

  TEST(XX_FILL__SSE2_U64, multiple_rows) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 192; channels += 15) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .Test(xnn_xx_fill_ukernel__sse2_u64);
      }
    }
  }

  TEST(XX_FILL__SSE2_U64, multiple_rows_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 192; channels += 15) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(193)
          .Test(xnn_xx_fill_ukernel__sse2_u64);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(XX_FILL__WASMSIMD_U64, channels_eq_64) {
    FillMicrokernelTester()
      .channels(64)
      .Test(xnn_xx_fill_ukernel__wasmsimd_u64);
  }

  TEST(XX_FILL__WASMSIMD_U64, channels_div_64) {
    for (size_t channels = 128; channels <= 192; channels += 64) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__wasmsimd_u64);
    }
  }

  TEST(XX_FILL__WASMSIMD_U64, channels_lt_64) {
    for (size_t channels = 1; channels < 64; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__wasmsimd_u64);
    }
  }

  TEST(XX_FILL__WASMSIMD_U64, channels_gt_64) {
    for (size_t channels = 65; channels < 128; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_xx_fill_ukernel__wasmsimd_u64);
    }
  }

  TEST(XX_FILL__WASMSIMD_U64, multiple_rows) {
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 192; channels += 15) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .Test(xnn_xx_fill_ukernel__wasmsimd_u64);
      }
    }
  }

  TEST(XX_FILL__WASMSIMD_U64, multiple_rows_with_output_stride) {
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 192; channels += 15) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(193)
          .Test(xnn_xx_fill_ukernel__wasmsimd_u64);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(XX_FILL__SCALAR_U16, channels_eq_16) {
  FillMicrokernelTester()
    .channels(16)
    .Test(xnn_xx_fill_ukernel__scalar_u16);
}

TEST(XX_FILL__SCALAR_U16, channels_div_16) {
  for (size_t channels = 32; channels <= 48; channels += 16) {
    FillMicrokernelTester()
      .channels(channels)
      .Test(xnn_xx_fill_ukernel__scalar_u16);
  }
}

TEST(XX_FILL__SCALAR_U16, channels_lt_16) {
  for (size_t channels = 1; channels < 16; channels++) {
    FillMicrokernelTester()
      .channels(channels)
      .Test(xnn_xx_fill_ukernel__scalar_u16);
  }
}

TEST(XX_FILL__SCALAR_U16, channels_gt_16) {
  for (size_t channels = 17; channels < 32; channels++) {
    FillMicrokernelTester()
      .channels(channels)
      .Test(xnn_xx_fill_ukernel__scalar_u16);
  }
}

TEST(XX_FILL__SCALAR_U16, multiple_rows) {
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 48; channels += 3) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .Test(xnn_xx_fill_ukernel__scalar_u16);
    }
  }
}

TEST(XX_FILL__SCALAR_U16, multiple_rows_with_output_stride) {
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 48; channels += 3) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .output_stride(49)
        .Test(xnn_xx_fill_ukernel__scalar_u16);
    }
  }
}