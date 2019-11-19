// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "global-average-pooling-operator-tester.h"

#include <xnnpack/params.h>


TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_input_scale) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_scale(input_scale)
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_input_zero_point) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_zero_point(uint8_t(input_zero_point))
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_output_scale) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_scale(output_scale)
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_output_zero_point) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_zero_point(uint8_t(output_zero_point))
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_small_width_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_input_scale) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_scale(input_scale)
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_input_zero_point) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .input_zero_point(uint8_t(input_zero_point))
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_output_scale) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_scale(output_scale)
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_output_zero_point) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GlobalAveragePoolingOperatorTester()
          .batch_size(1)
          .width(width)
          .channels(channels)
          .output_zero_point(uint8_t(output_zero_point))
          .TestNWCxQ8();
      }
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmin(128)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, unit_batch_large_width_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .qmax(128)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, small_batch_small_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, small_batch_small_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, small_batch_small_width_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, small_batch_large_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, small_batch_large_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .input_stride(5 * channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_Q8, small_batch_large_width_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.q8.gavgpool.mr; width <= 4 * xnn_params.q8.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxQ8();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_small_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(1)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, unit_batch_large_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_small_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = 1; width <= xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .TestNWCxF32();
    }
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NWC_F32, small_batch_large_width_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
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
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t width = xnn_params.f32.gavgpool.mr; width <= 4 * xnn_params.f32.gavgpool.mr; width++) {
      GlobalAveragePoolingOperatorTester()
        .batch_size(3)
        .width(width)
        .channels(channels)
        .output_stride(5 * channels)
        .TestNWCxF32();
    }
  }
}
