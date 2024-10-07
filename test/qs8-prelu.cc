// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-prelu.yaml
//   Generator: tools/generate-prelu-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/prelu.h"
#include "prelu-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_PRELU__AVX2_2X16, channels_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
  }

  TEST(QS8_PRELU__AVX2_2X16, channels_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, channels_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, channels_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, rows_lt_2) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
      }
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, rows_div_2) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
      }
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, rows_gt_2) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
      }
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, input_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
      }
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, output_stride) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
      }
    }
  }

  TEST(QS8_PRELU__AVX2_2X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_qs8_prelu_ukernel__avx2_2x16, xnn_init_qs8_prelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(QS8_PRELU__SCALAR_2X4, channels_eq_4) {
  PReLUMicrokernelTester()
    .rows(2)
    .channels(4)
    .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
}

TEST(QS8_PRELU__SCALAR_2X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
  }
}

TEST(QS8_PRELU__SCALAR_2X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
  }
}

TEST(QS8_PRELU__SCALAR_2X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
  }
}

TEST(QS8_PRELU__SCALAR_2X4, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
    }
  }
}

TEST(QS8_PRELU__SCALAR_2X4, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
    }
  }
}

TEST(QS8_PRELU__SCALAR_2X4, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
    }
  }
}

TEST(QS8_PRELU__SCALAR_2X4, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .iterations(1)
        .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
    }
  }
}

TEST(QS8_PRELU__SCALAR_2X4, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .output_stride(23)
        .iterations(1)
        .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
    }
  }
}

TEST(QS8_PRELU__SCALAR_2X4, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_qs8_prelu_ukernel__scalar_2x4, xnn_init_qs8_prelu_scalar_params);
    }
  }
}