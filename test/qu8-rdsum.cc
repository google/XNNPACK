// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-rdsum.yaml
//   Generator: tools/generate-rdsum-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rdsum-microkernel-tester.h"


TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows(14)
    .channels(channel_tile)
    .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  const size_t channel_tile = 4;
  RDSumMicrokernelTester()
    .rows(14)
    .channels(channel_tile)
    .input_stride(7)
    .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile) {
  const size_t channel_tile = 4;
  for (size_t rows = 1; rows < 14; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  const size_t channel_tile = 4;
  for (size_t rows = 1; rows < 14; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(channel_tile)
      .input_stride(7)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile) {
  const size_t channel_tile = 4;
  for (size_t rows = 1; rows <= 35; rows += 7) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 4;
  for (size_t rows = 1; rows <= 35; rows += 7) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(channel_tile)
      .input_stride(7)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_div_4_2pass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_div_4_2pass_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(67)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_lt_4_2pass_fulltile) {
  const size_t channel_tile = 4;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_lt_4_2pass_subtile) {
  const size_t channel_tile = 4;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile) {
  const size_t channel_tile = 4;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 4;
  for (size_t channels = 1; channels < channel_tile; channels++) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_gt_4_2pass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_gt_4_2pass_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 35; rows += 14) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 35; rows += 14) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
    }
  }
}

TEST(QU8_RDSUM_7P7X__SCALAR_C4, overflow_accumulator) {
  const size_t channel_tile = 4;
  for (size_t channels = 1; channels < channel_tile*2; ++channels) {
     RDSumMicrokernelTester()
       .rows(264)
       .channels(channels)
       .Test(xnn_qu8_rdsum_ukernel_7p7x__scalar_c4);
  }
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(19)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(19)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U16, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile*2; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(37)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(37)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U32, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile*2; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(67)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(67)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__NEON_U64, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile*2; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_qu8_rdsum_ukernel_7p7x__neon_u64);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(19)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(19)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 16;
    for (size_t channels = 1; channels < channel_tile*2; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(37)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(37)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 64; channels < 256; channels += 32) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 33; channels < 64; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 32;
    for (size_t channels = 1; channels < channel_tile*2; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(67)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channel_tile)
        .input_stride(67)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 128; channels < 512; channels += 64) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 128; channels < 512; channels += 64) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(1031)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t rows = 1; rows <= 35; rows += 7) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 65; channels < 128; channels++) {
      RDSumMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 14; rows++) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 65; channels < 128; channels++) {
      for (size_t rows = 1; rows < 35; rows += 14) {
        RDSumMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(149)
          .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
      }
    }
  }

  TEST(QU8_RDSUM_7P7X__SSSE3_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_SSSE3;
    const size_t channel_tile = 64;
    for (size_t channels = 1; channels < channel_tile*2; ++channels) {
       RDSumMicrokernelTester()
         .rows(264)
         .channels(channels)
         .Test(xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
