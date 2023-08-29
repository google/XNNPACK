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

#include "global-average-pooling-operator-tester.h"

#include <xnnpack/params.h>

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_scale(input_scale)
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_zero_point(uint8_t(input_zero_point))
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_scale(output_scale)
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_zero_point(uint8_t(output_zero_point))
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_scale(input_scale)
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_zero_point(uint8_t(input_zero_point))
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_scale(output_scale)
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_zero_point(uint8_t(output_zero_point))
          .TestNWCxQU8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_small_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_large_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, large_width_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(spatial_tile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxQU8();
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_scale(input_scale)
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_zero_point(uint8_t(input_zero_point))
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_scale(output_scale)
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_zero_point(uint8_t(output_zero_point))
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_scale(input_scale)
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_zero_point(uint8_t(input_zero_point))
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_scale(output_scale)
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_zero_point(uint8_t(output_zero_point))
          .TestNWCxQS8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_small_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_large_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, large_width_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qs8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(spatial_tile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxQS8();
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_small_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_small_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_large_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_large_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_small_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_large_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, large_width_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  if (gavgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(spatial_tile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxF16();
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_small_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_small_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_small_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_large_width) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_large_width_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_large_width_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = spatial_tile; width <= 4 * spatial_tile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, large_width_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const uint32_t spatial_tile = std::max<uint32_t>(gavgpool_config->row_tile, 1);
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(spatial_tile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxF32();
}
