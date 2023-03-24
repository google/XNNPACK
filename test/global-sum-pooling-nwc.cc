// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "global-sum-pooling-operator-tester.h"

#include <xnnpack/params.h>

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_small_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_small_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_large_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, unit_batch_large_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, small_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, small_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, small_batch_small_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, small_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, small_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F16, small_batch_large_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_small_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_small_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_large_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, unit_batch_large_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, small_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, small_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, small_batch_small_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, small_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, small_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_SUM_POOLING_NWC_F32, small_batch_large_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalSumPoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}
