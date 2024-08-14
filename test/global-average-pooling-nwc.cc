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
#include "xnnpack/config.h"
#include "global-average-pooling-operator-tester.h"

static const uint32_t kSpatialTile = 1;

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, unit_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQU8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QU8, small_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(kSpatialTile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxQU8();
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, unit_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQS8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_QS8, small_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(kSpatialTile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxQS8();
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_large_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, unit_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF16();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F16, small_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(kSpatialTile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxF16();
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_small_width) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_small_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_large_width_with_input_stride) {
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = kSpatialTile; width <= 4 * kSpatialTile; width++) {
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
  GlobalAveragePoolingOperatorTester()
    .batch_size(5)
    .width(kSpatialTile * 3)
    .channels(15)
    .multithreaded(true)
    .TestNWCxF32();
}
