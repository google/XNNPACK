// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/fill.h>
#include "fill-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_FILL__NEON, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    FillMicrokernelTester()
      .channels(4)
      .Test(xnn_x32_fill_ukernel__neon);
  }

  TEST(X32_FILL__NEON, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__neon);
    }
  }

  TEST(X32_FILL__NEON, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__neon);
    }
  }

  TEST(X32_FILL__NEON, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__neon);
    }
  }

  TEST(X32_FILL__NEON, multiple_rows) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .Test(xnn_x32_fill_ukernel__neon);
      }
    }
  }

  TEST(X32_FILL__NEON, multiple_rows_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(17)
          .Test(xnn_x32_fill_ukernel__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_FILL__SSE, channels_eq_4) {
    TEST_REQUIRES_X86_SSE;
    FillMicrokernelTester()
      .channels(4)
      .Test(xnn_x32_fill_ukernel__sse);
  }

  TEST(X32_FILL__SSE, channels_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__sse);
    }
  }

  TEST(X32_FILL__SSE, channels_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__sse);
    }
  }

  TEST(X32_FILL__SSE, channels_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__sse);
    }
  }

  TEST(X32_FILL__SSE, multiple_rows) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .Test(xnn_x32_fill_ukernel__sse);
      }
    }
  }

  TEST(X32_FILL__SSE, multiple_rows_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(17)
          .Test(xnn_x32_fill_ukernel__sse);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(X32_FILL__WASMSIMD, channels_eq_4) {
    FillMicrokernelTester()
      .channels(4)
      .Test(xnn_x32_fill_ukernel__wasmsimd);
  }

  TEST(X32_FILL__WASMSIMD, channels_div_4) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__wasmsimd);
    }
  }

  TEST(X32_FILL__WASMSIMD, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__wasmsimd);
    }
  }

  TEST(X32_FILL__WASMSIMD, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(xnn_x32_fill_ukernel__wasmsimd);
    }
  }

  TEST(X32_FILL__WASMSIMD, multiple_rows) {
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .Test(xnn_x32_fill_ukernel__wasmsimd);
      }
    }
  }

  TEST(X32_FILL__WASMSIMD, multiple_rows_with_output_stride) {
    for (size_t rows = 2; rows < 5; rows++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        FillMicrokernelTester()
          .channels(channels)
          .rows(rows)
          .output_stride(17)
          .Test(xnn_x32_fill_ukernel__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


TEST(X32_FILL__SCALAR_FLOAT, eq_1) {
  FillMicrokernelTester()
    .channels(1)
    .Test(xnn_x32_fill_ukernel__scalar_float);
}

TEST(X32_FILL__SCALAR_FLOAT, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    FillMicrokernelTester()
      .channels(channels)
      .Test(xnn_x32_fill_ukernel__scalar_float);
  }
}

TEST(X32_FILL__SCALAR_FLOAT, multiple_rows) {
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 16; channels += 3) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .Test(xnn_x32_fill_ukernel__scalar_float);
    }
  }
}

TEST(X32_FILL__SCALAR_FLOAT, multiple_rows_with_output_stride) {
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 16; channels += 3) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .output_stride(17)
        .Test(xnn_x32_fill_ukernel__scalar_float);
    }
  }
}


TEST(X32_FILL__SCALAR_INT, eq_1) {
  FillMicrokernelTester()
    .channels(1)
    .Test(xnn_x32_fill_ukernel__scalar_int);
}

TEST(X32_FILL__SCALAR_INT, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    FillMicrokernelTester()
      .channels(channels)
      .Test(xnn_x32_fill_ukernel__scalar_int);
  }
}

TEST(X32_FILL__SCALAR_INT, multiple_rows) {
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 16; channels += 3) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .Test(xnn_x32_fill_ukernel__scalar_int);
    }
  }
}

TEST(X32_FILL__SCALAR_INT, multiple_rows_with_output_stride) {
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < 16; channels += 3) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .output_stride(17)
        .Test(xnn_x32_fill_ukernel__scalar_int);
    }
  }
}
