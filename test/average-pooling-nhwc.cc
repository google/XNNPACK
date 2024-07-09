// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/config.h"
#include "average-pooling-operator-tester.h"

static std::pair<size_t, size_t> SmallPoolSize(size_t max_elements) {
  const size_t small_side = size_t(std::floor(std::sqrt(double(max_elements))));
  const size_t large_side = small_side + 1;
  if (small_side * large_side < max_elements) {
    return std::make_pair(small_side, large_side);
  } else {
    return std::make_pair(small_side - 1, large_side - 1);
  }
}

static std::pair<size_t, size_t> LargePoolSize(size_t min_elements) {
  const size_t small_side = size_t(std::ceil(std::sqrt(double(min_elements))));
  return std::make_pair(small_side, small_side + 1);
}

/**************************** AVGPOOL path, unipass ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_width_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_height_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_tf_same_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_tf_same_padding_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_input_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_output_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_input_scale) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_input_zero_point) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_output_scale) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_output_zero_point) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_qmin) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_pool_with_qmax) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

/**************************** AVGPOOL path, unipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_width_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_height_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_tf_same_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_input_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_output_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_input_scale) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_input_zero_point) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_output_scale) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_output_zero_point) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_qmin) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_pool_with_qmax) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

/**************************** AVGPOOL path, multipass ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_tf_same_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_pool_with_tf_same_padding_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile + 1);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first)
    .input_width(pooling_size.second)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestQU8();
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestQU8();
}

/**************************** AVGPOOL path, multipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestQU8();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestQU8();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_tf_same_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestQU8();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first + 3)
        .input_width(pooling_size.second + 2)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second + 3)
        .input_width(pooling_size.first + 2)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_qu8_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  AveragePoolingOperatorTester()
    .batch_size(11)
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestQU8();
}

/**************************** GAVGPOOL path, unipass ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, small_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With left padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_left(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_left(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();

    /* With right padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_right(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_right(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With top padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_top(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_top(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();

    /* With bottom padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_bottom(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_bottom(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, small_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

/**************************** GAVGPOOL path, unipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With left padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_left(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_left(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();

    /* With right padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_right(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_right(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With top padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_top(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_top(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();

    /* With bottom padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_bottom(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_bottom(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_small_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

/**************************** GAVGPOOL path, multipass ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, large_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
    for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_top = 1; padding_top <= 2; padding_top++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first - padding_top)
        .input_width(pooling_size.second)
        .padding_top(padding_top)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second - padding_top)
        .input_width(pooling_size.first)
        .padding_top(padding_top)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
    for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first - padding_bottom)
        .input_width(pooling_size.second)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second - padding_bottom)
        .input_width(pooling_size.first)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, large_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

/**************************** GAVGPOOL path, multipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
    for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_top = 1; padding_top <= 2; padding_top++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first - padding_top)
        .input_width(pooling_size.second)
        .padding_top(padding_top)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second - padding_top)
        .input_width(pooling_size.first)
        .padding_top(padding_top)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
    for (size_t padding_bottom = 1; padding_bottom <= 2; padding_bottom++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first - padding_bottom)
        .input_width(pooling_size.second)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second - padding_bottom)
        .input_width(pooling_size.first)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_input_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_scale(input_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_input_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_output_scale) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_scale(output_scale)
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_output_zero_point) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQU8();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestQU8();
  }
}

TEST(AVERAGE_POOLING_NHWC_QU8, batched_large_image_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_qu8_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .multithreaded(true)
      .TestQU8();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .multithreaded(true)
      .TestQU8();
  }
}

/**************************** AVGPOOL path, setup ****************************/

TEST(AVERAGE_POOLING_NHWC_QU8, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(5)
    .next_batch_size(2)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_changing_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_channels(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_channels(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_local_to_global) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(6)
    .input_width(5)
    .next_input_height(5)
    .next_input_width(3)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

TEST(AVERAGE_POOLING_NHWC_QU8, setup_global_to_local) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(5)
    .input_width(3)
    .next_input_height(6)
    .next_input_width(5)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupQU8();
}

/**************************** AVGPOOL path, unipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, small_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_width_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_height_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_tf_same_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_tf_same_padding_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF16();
}


TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_input_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_output_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_qmin) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_pool_with_qmax) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

/**************************** AVGPOOL path, unipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_width_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_height_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_tf_same_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_input_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_output_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_qmin) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_pool_with_qmax) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

/**************************** AVGPOOL path, multipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, large_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .padding_left(padding_left)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .padding_left(padding_left)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .padding_right(padding_right)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .padding_right(padding_right)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_tf_same_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_pool_with_tf_same_padding_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile + 1);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first)
    .input_width(pooling_size.second)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF16();
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF16();
}

/**************************** AVGPOOL path, multipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF16();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF16();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_tf_same_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF16();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f16_avgpool_config();
  if (avgpool_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  AveragePoolingOperatorTester()
    .batch_size(11)
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF16();
}

/**************************** GAVGPOOL path, unipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, small_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With left padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_left(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_left(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();

    /* With right padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_right(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_right(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With top padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_top(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_top(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();

    /* With bottom padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_bottom(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_bottom(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, small_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

/**************************** GAVGPOOL path, unipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With left padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_left(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_left(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();

    /* With right padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_right(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_right(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With top padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_top(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_top(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();

    /* With bottom padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_bottom(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_bottom(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_small_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

/**************************** GAVGPOOL path, multipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, large_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
    for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_top = 1; padding_top <= 2; padding_top++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first - padding_top)
        .input_width(pooling_size.second)
        .padding_top(padding_top)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second - padding_top)
        .input_width(pooling_size.first)
        .padding_top(padding_top)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
    for (size_t padding_bottom = 1; padding_bottom <= 2; padding_bottom++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first - padding_bottom)
        .input_width(pooling_size.second)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second - padding_bottom)
        .input_width(pooling_size.first)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, large_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

/**************************** GAVGPOOL path, multipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
    for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first - padding_top)
        .input_width(pooling_size.second)
        .padding_top(padding_top)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second - padding_top)
        .input_width(pooling_size.first)
        .padding_top(padding_top)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
    for (size_t padding_bottom = 1; padding_bottom <= 2; padding_bottom++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first - padding_bottom)
        .input_width(pooling_size.second)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF16();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second - padding_bottom)
        .input_width(pooling_size.first)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF16();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF16();
  }
}

TEST(AVERAGE_POOLING_NHWC_F16, batched_large_image_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .multithreaded(true)
      .TestF16();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .multithreaded(true)
      .TestF16();
  }
}

/**************************** AVGPOOL path, setup ****************************/

TEST(AVERAGE_POOLING_NHWC_F16, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(5)
    .next_batch_size(2)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_changing_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_channels(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_channels(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_local_to_global) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(6)
    .input_width(5)
    .next_input_height(5)
    .next_input_width(3)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

TEST(AVERAGE_POOLING_NHWC_F16, setup_global_to_local) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(5)
    .input_width(3)
    .next_input_height(6)
    .next_input_width(5)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF16();
}

/**************************** AVGPOOL path, unipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, small_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_width_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_height_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_tf_same_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_tf_same_padding_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_input_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_output_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_qmin) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_pool_with_qmax) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

/**************************** AVGPOOL path, unipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_width_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_height_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_tf_same_padding) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_input_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_output_stride) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_qmin) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_pool_with_qmax) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(avgpool_config->primary_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

/**************************** AVGPOOL path, multipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, large_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .padding_left(padding_left)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .padding_left(padding_left)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .padding_right(padding_right)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .padding_right(padding_right)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_tf_same_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_pool_with_tf_same_padding_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile + 1);
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first)
    .input_width(pooling_size.second)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF32();
  AveragePoolingOperatorTester()
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .padding_tf_same(true)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF32();
}

/**************************** AVGPOOL path, multipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride_width = 1; stride_width <= 2; stride_width++) {
      for (size_t stride_height = 1; stride_height <= 2; stride_height++) {
        if (stride_width == 1 && stride_height == 1) {
          continue;
        }

        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(pooling_size.second + 2)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(pooling_size.first + 2)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride_height(stride_height)
          .stride_width(stride_width)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.first + 3)
            .input_width(pooling_size.second + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.first)
            .pooling_width(pooling_size.second)
            .stride(stride)
            .channels(channels)
            .TestF32();
          AveragePoolingOperatorTester()
            .batch_size(2)
            .input_height(pooling_size.second + 3)
            .input_width(pooling_size.first + 2)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size.second)
            .pooling_width(pooling_size.first)
            .stride(stride)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_tf_same_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t stride = 1; stride <= 2; stride++) {
      for (size_t input_height = pooling_size.first + 3; input_height <= pooling_size.first + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.second + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.second + 2; input_width <= pooling_size.second + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.first + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.first)
          .pooling_width(pooling_size.second)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_height = pooling_size.second + 3; input_height <= pooling_size.second + 4; input_height++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(input_height)
          .input_width(pooling_size.first + 2)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
      for (size_t input_width = pooling_size.first + 2; input_width <= pooling_size.first + 3; input_width++) {
        AveragePoolingOperatorTester()
          .batch_size(2)
          .input_height(pooling_size.second + 3)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(pooling_size.second)
          .pooling_width(pooling_size.first)
          .stride(stride)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first + 3)
      .input_width(pooling_size.second + 2)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second + 3)
      .input_width(pooling_size.first + 2)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_pool_multithreaded) {
  const struct xnn_avgpool_config* avgpool_config = xnn_init_f32_avgpool_config();
  ASSERT_NE(avgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(avgpool_config->primary_tile * 2);
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(pooling_size.first + 3)
    .input_width(pooling_size.second + 2)
    .pooling_height(pooling_size.first)
    .pooling_width(pooling_size.second)
    .channels(15)
    .multithreaded(true)
    .TestF32();
}

/**************************** GAVGPOOL path, unipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, small_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With left padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_left(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_left(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();

    /* With right padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_right(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_right(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With top padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_top(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_top(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();

    /* With bottom padding */
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_bottom(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_bottom(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, small_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

/**************************** GAVGPOOL path, unipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With left padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_left(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_left(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();

    /* With right padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second - 1)
      .padding_right(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first - 1)
      .padding_right(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    /* With top padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_top(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_top(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();

    /* With bottom padding */
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first - 1)
      .input_width(pooling_size.second)
      .padding_bottom(1)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second - 1)
      .input_width(pooling_size.first)
      .padding_bottom(1)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_small_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = SmallPoolSize(gavgpool_config->row_tile);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

/**************************** GAVGPOOL path, multipass ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, large_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
    for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_top = 1; padding_top <= 2; padding_top++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first - padding_top)
        .input_width(pooling_size.second)
        .padding_top(padding_top)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second - padding_top)
        .input_width(pooling_size.first)
        .padding_top(padding_top)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
    for (size_t padding_bottom = 1; padding_bottom <= 2; padding_bottom++) {
      AveragePoolingOperatorTester()
        .input_height(pooling_size.first - padding_bottom)
        .input_width(pooling_size.second)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .input_height(pooling_size.second - padding_bottom)
        .input_width(pooling_size.first)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, large_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

/**************************** GAVGPOOL path, multipass, batched ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_with_width_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 1; padding_left <= 2; padding_left++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_left)
        .padding_left(padding_left)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
    for (size_t padding_right = 1; padding_right <= 2; padding_right++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first)
        .input_width(pooling_size.second - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second)
        .input_width(pooling_size.first - padding_right)
        .padding_right(padding_right)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_with_height_padding) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first - padding_top)
        .input_width(pooling_size.second)
        .padding_top(padding_top)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second - padding_top)
        .input_width(pooling_size.first)
        .padding_top(padding_top)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
    for (size_t padding_bottom = 1; padding_bottom <= 2; padding_bottom++) {
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.first - padding_bottom)
        .input_width(pooling_size.second)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.first)
        .pooling_width(pooling_size.second)
        .channels(channels)
        .TestF32();
      AveragePoolingOperatorTester()
        .batch_size(2)
        .input_height(pooling_size.second - padding_bottom)
        .input_width(pooling_size.first)
        .padding_bottom(padding_bottom)
        .pooling_height(pooling_size.second)
        .pooling_width(pooling_size.first)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_with_input_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .input_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_with_output_stride) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .output_pixel_stride(2 * channels + 3)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_with_qmin) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmin(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmin(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_with_qmax) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .TestF32();
  }
}

TEST(AVERAGE_POOLING_NHWC_F32, batched_large_image_multithreaded) {
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  ASSERT_NE(gavgpool_config, nullptr);
  const std::pair<size_t, size_t> pooling_size = LargePoolSize(gavgpool_config->row_tile * 2);
  for (size_t channels = 1; channels <= 100; channels += 15) {
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.first)
      .input_width(pooling_size.second)
      .pooling_height(pooling_size.first)
      .pooling_width(pooling_size.second)
      .channels(channels)
      .qmax(128)
      .multithreaded(true)
      .TestF32();
    AveragePoolingOperatorTester()
      .batch_size(2)
      .input_height(pooling_size.second)
      .input_width(pooling_size.first)
      .pooling_height(pooling_size.second)
      .pooling_width(pooling_size.first)
      .channels(channels)
      .qmax(128)
      .multithreaded(true)
      .TestF32();
  }
}

/**************************** AVGPOOL path, setup ****************************/

TEST(AVERAGE_POOLING_NHWC_F32, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(5)
    .next_batch_size(2)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_changing_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_channels(25)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(8)
    .input_width(8)
    .next_channels(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_local_to_global) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(6)
    .input_width(5)
    .next_input_height(5)
    .next_input_width(3)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(AVERAGE_POOLING_NHWC_F32, setup_global_to_local) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  AveragePoolingOperatorTester()
    .batch_size(2)
    .input_height(5)
    .input_width(3)
    .next_input_height(6)
    .next_input_width(5)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}
