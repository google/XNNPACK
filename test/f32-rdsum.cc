// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rdsum.yaml
//   Generator: tools/generate-rdsum-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rdsum-microkernel-tester.h"


TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile) {
  RDSumMicrokernelTester()
    .rows(14)
    .channels(4)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  RDSumMicrokernelTester()
    .rows(14)
    .channels(4)
    .input_stride(7)
    .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile) {
  for (size_t rows = 1; rows < 14; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  for (size_t rows = 1; rows < 14; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile) {
  for (size_t rows = 1; rows <= 35; rows += 7) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  for (size_t rows = 1; rows <= 35; rows += 7) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_2pass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_2pass_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_2pass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_2pass_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_2pass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_2pass_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 35; rows += 14) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 35; rows += 14) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_RDSUM_7P7X__SCALAR_C4, overflow_accumulator) {
  for (size_t channels = 1; channels < 8; ++channels) {
     RDSumMicrokernelTester()
       .rows(264)
       .channels(channels)
       .Test(xnn_f32_rdsum_ukernel_7p7x__scalar_c4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C16, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C32, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 64; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__NEON_C64, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 128; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__neon_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 16; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 32; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 64; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__SSE_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 128; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__sse_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 16; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 32; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 64; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 128; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__avx_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 16; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 32; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 64; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__AVX512F_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 128; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__avx512f_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_subtile) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_multipass_fulltile) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_2pass_fulltile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_2pass_subtile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_multipass_fulltile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_2pass_fulltile) {
    for (size_t channels = 1; channels < 16; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_2pass_subtile) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_multipass_fulltile) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_2pass_fulltile) {
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_2pass_subtile) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_multipass_fulltile) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C16, overflow_accumulator) {
    for (size_t channels = 1; channels < 32; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_subtile) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_multipass_fulltile) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_2pass_fulltile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_2pass_subtile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_multipass_fulltile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_2pass_fulltile) {
    for (size_t channels = 1; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_2pass_subtile) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_multipass_fulltile) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_2pass_fulltile) {
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_2pass_subtile) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_multipass_fulltile) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C32, overflow_accumulator) {
    for (size_t channels = 1; channels < 64; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_fulltile) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(64)
      .input_stride(67)
      .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_subtile) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_multipass_fulltile) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(64)
        .input_stride(67)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_2pass_fulltile) {
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_2pass_subtile) {
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_multipass_fulltile) {
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_2pass_fulltile) {
    for (size_t channels = 1; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_2pass_subtile) {
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_multipass_fulltile) {
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 64; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_2pass_fulltile) {
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_2pass_subtile) {
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_multipass_fulltile) {
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_RDSUM_7P7X__WASMSIMD_C64, overflow_accumulator) {
    for (size_t channels = 1; channels < 128; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c64, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
